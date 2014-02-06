"""
    The CellH5 Project
    Copyright (c) 2012 - 2013 Christoph Sommer, Michael Held, Bernd Fischer
    Gerlich Lab, IMBA Vienna, Huber Lab, EMBL Heidelberg
    www.cellh5.org

    CellH5 is distributed under the LGPL License.
    See LICENSE.txt for details.
    See AUTHORS.txt for author contributions.
"""

import os
import zlib
import h5py
import numpy
import base64
import warnings
import unittest
import functools
import collections

import pandas

import matplotlib
matplotlib.use('Qt4Agg', warn=False)
import matplotlib.pyplot as mpl

from matplotlib.colors import hex2color

from version import version

ICON_FILE = os.path.join(os.path.split(__file__)[0], "cellh5_icon.ico")
GALLERY_SIZE = 60

class CH5Const(object):

    # defaults for unpredicted objects, -1 one might be not a
    UNPREDICTED_LABEL = -99
    UNPREDICTED_PROB = numpy.nan

import time
def profile(func):
    """Decorator function for profiling."""
    def wrap(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        print "function: %s, %.2fs" %(func.__name__, (time.time()-t0))
        return result
    return wrap


def hex2rgb(color, mpl=False):
    """Return the rgb color as python int in the range 0-255."""
    assert color.startswith("#")
    if mpl:
        fac = 1.0
    else:
        fac = 255.0

    rgb = [int(i*fac) for i in hex2color(color)]
    return tuple(rgb)


class memoize(object):
    """Cache the return value of a method

    This class is meant to be used as a decorator of methods. The return value
    from a given method invocation will be cached on the instance whose method
    was invoked. All arguments passed to a method decorated with memoize must
    be hashable.

    If a memoized method is invoked directly on its class the result will not
    be cached. Instead the method will be invoked like a static method:
    class Obj(object_):
        @memoize
        def add_to(self, arg):
            return self + arg
    Obj.add_to(1) # not enough arguments
    Obj.add_to(1, 2) # returns 3, result is not cached
    """
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return functools.partial(self, obj)

    def __call__(self, *args, **kw):
        key = (self.func, args, frozenset(kw.items()))

        obj = args[0]
        if not hasattr(object, "_cache"):
            obj._cache = dict()

        try:
            res = obj._cache[key]
        except KeyError:
            res = obj._cache[key] = self.func(*args, **kw)
        except TypeError:
            # if key is not hashable (e.g ndarrays)
            res = self.func(*args, **kw)

        return res


class CH5Position(object):

    def __init__(self, plate, well, pos, grp_pos, parent):
        self.plate = plate
        self.well = well
        self.pos = pos
        self.grp_pos_path = grp_pos
        self.definitions = parent

    def __getitem__(self, key):
        path = "%s/%s" %(self.grp_pos_path, key)
        return self.definitions.get_file_handle()[path]

    def get_tracking(self):
        return self['object']['tracking'].value

    def _get_tracking_lookup(self, obj_idx='obj_idx1'):
        dset_tracking_idx1 = self.get_tracking()[obj_idx]
        tracking_lookup_idx1 = collections.defaultdict()
        for i, o in enumerate(dset_tracking_idx1):
            tracking_lookup_idx1.setdefault(o, []).append(i)
        return tracking_lookup_idx1

    # @profile
    def get_class_prediction(self, object_='primary__primary'):
        path = 'feature/%s/object_classification/prediction' %object_
        return self[path]

    # @profile
    def get_prediction_probabilities(self, indices=None,
                                     object_="primary__primary"):
        path = 'feature/%s/object_classification/probability' %object_

        if indices is None:
            return self[path].value
        else:
            # read probs only once per cell, reading from share is too slow
            ishape = indices.shape
            indices = indices.flatten()
            indices2 = numpy.unique(indices)
            _, nclasses  = self[path].shape
            probs = numpy.empty((indices.size, nclasses))

            for i, j in enumerate(indices2):
                k = numpy.where(indices == j)[0]
                probs[k, :] = self[path][j, :]

            return probs.reshape(ishape+(nclasses, ))

    def has_classification(self, object_):
        return 'object_classification' in self['feature'][object_]

    def get_crack_contour(self, index, object_='primary__primary',
                          bb_corrected=True):
        if not isinstance(index, (list, tuple)):
            index = (index,)
        crack_list = []
        for ind in index:
            crack_str = self['feature'][object_]['crack_contour'][ind]
            crack = numpy.asarray(zlib.decompress( \
                             base64.b64decode(crack_str)).split(','), \
                             dtype=numpy.float32).reshape(-1,2)

            if bb_corrected:
                bb = self['feature'][object_]['center'][ind]
                crack[:,0] -= bb['x'] - GALLERY_SIZE/2
                crack[:,1] -= bb['y'] - GALLERY_SIZE/2
                crack.clip(0, GALLERY_SIZE)

            crack_list.append(crack)

        return crack_list

    def has_object_entries(self, object_='primary__primary'):
        return len(self['object'][object_]) > 0

    def get_object_features(self, object_='primary__primary'):
        if len(self['feature'][object_]['object_features']) > 0:
            return self['feature'] \
                   [object_] \
                   ['object_features'].value
        else:
            return []


    def get_image(self, t, c, z=0):
        return self['image'] \
                    ['channel'] \
                    [c, t, z, :, :]

    def get_gallery_image(self, index, object_='primary__primary'):
        if not isinstance(index, (list, tuple)):
            index = (index,)
        image_list = []
        channel_idx = self.definitions.image_definition['region']['channel_idx'][self.definitions.image_definition['region']['region_name'] == 'region___%s' % object_][0]
        image_width = self['image']['channel'].shape[3]
        image_height = self['image']['channel'].shape[4]


        for ind in index:
            time_idx = self['object'][object_][ind]['time_idx']
            cen1 = self['feature'][object_]['center'][ind]
            image = numpy.zeros((GALLERY_SIZE,GALLERY_SIZE), dtype=numpy.uint8)

            tmp_img = self.get_image(time_idx, channel_idx, 0)[
                              max(0, cen1[1]-GALLERY_SIZE/2):min(image_width,  cen1[1]+GALLERY_SIZE/2),
                              max(0, cen1[0]-GALLERY_SIZE/2):min(image_height, cen1[0]+GALLERY_SIZE/2)]

            image[(image.shape[0]-tmp_img.shape[0]):, :tmp_img.shape[1]] = tmp_img
            image_list.append(image)

        if len(index) > 1:
            return numpy.concatenate(image_list, axis=1)
        return image_list[0]

    def get_gallery_image_rgb(self, index, object_=('primary__primary',)):
        if len(object_) == 1:
            img_ = self.get_gallery_image(index, object_[0])
            rgb_shape = img_.shape + (3,)
            img = numpy.zeros(rgb_shape, img_.dtype)
            for c in range(3): img[:,:,c] = img_
            return img

        for c in range(3):
            if c == 0:
                img_ = self.get_gallery_image(index, object_[c])
                rgb_shape = img_.shape + (3,)
                img = numpy.zeros(rgb_shape, img_.dtype)
                img[:,:, 0] = img_
            if 0 < c < len(object_):
                img[:,:,c] = self.get_gallery_image(index, object_[c])

        return img

    def get_gallery_image_list(self, index, object_='primary__primary'):
        image_list = []
        channel_idx = self.definitions.image_definition['region']['channel_idx'][self.definitions.image_definition['region']['region_name'] == 'region___%s' % object_][0]
        image_width = self['image']['channel'].shape[3]
        image_height = self['image']['channel'].shape[4]


        time_idxs = [self['object'][object_][ind]['time_idx'] for ind in index]
        center_idxs = [self['feature'][object_]['center'][ind] for ind in index]

        for k in xrange(len(index)):
            time_idx = time_idxs[k]
            cen1 = center_idxs[k]
            image = numpy.zeros((GALLERY_SIZE,GALLERY_SIZE), dtype=numpy.uint8)

            tmp_img = self['image/channel'][channel_idx, time_idx, 0,
                                 max(0, cen1[1]-GALLERY_SIZE/2):min(image_width,  cen1[1]+GALLERY_SIZE/2),
                                 max(0, cen1[0]-GALLERY_SIZE/2):min(image_height, cen1[0]+GALLERY_SIZE/2)]

            image[(image.shape[0]-tmp_img.shape[0]):, :tmp_img.shape[1]] = tmp_img
            image_list.append(image)

        return image_list

    def get_gallery_image_generator(self, index, object_='primary__primary'):
        image_list = []
        channel_idx = self.definitions.image_definition['region']['channel_idx'][self.definitions.image_definition['region']['region_name'] == 'region___%s' % object_][0]
        image_width = self['image']['channel'].shape[3]
        image_height = self['image']['channel'].shape[4]

        for ind in index:
            time_idx = self['object'][object_][ind]['time_idx']
            cen1 = self['feature'][object_]['center'][ind]
            image = numpy.zeros((GALLERY_SIZE,GALLERY_SIZE), dtype=numpy.uint8)

            tmp_img = self['image/channel'][channel_idx, time_idx, 0,
                                 max(0, cen1[1]-GALLERY_SIZE/2):min(image_width,  cen1[1]+GALLERY_SIZE/2),
                                 max(0, cen1[0]-GALLERY_SIZE/2):min(image_height, cen1[0]+GALLERY_SIZE/2)]

            image[(image.shape[0]-tmp_img.shape[0]):, :tmp_img.shape[1]] = tmp_img
            yield image

    def get_gallery_image_matrix(self, index, shape, object_='primary__primary'):
        image = numpy.zeros((GALLERY_SIZE * shape[0],
                             GALLERY_SIZE * shape[1]), dtype=numpy.uint8)
        i,j = 0, 0
        img_gen = self.get_gallery_image_generator(index, object_)

        for i in range(shape[0]):
            for j in range(shape[1]):
                try:
                    img = img_gen.next()
                except StopIteration:
                    break
                a = i * GALLERY_SIZE
                b = j * GALLERY_SIZE
                c = a + GALLERY_SIZE
                d = b + GALLERY_SIZE

                if (c,d) > image.shape:
                    break
                image[a:c, b:d] = img

        return image

    def get_gallery_image_contour(self, index, object_=('primary__primary',), color=None, scale=None):
        img = self.get_gallery_image_rgb(index, object_)
        if scale is not None:
            img = numpy.clip(img.astype(numpy.float32)*scale, 0, 255).astype(numpy.uint8)
        for obj_id in object_:
            crack = self.get_crack_contour(index, obj_id)


            if color is None:
                class_color = self.get_class_color(index, obj_id)
                if class_color is None:
                    class_color = ['#FFFFFF']*len(crack)

                if not isinstance(class_color, (list, tuple)):
                    class_color = [class_color]
            else:
                class_color = [color]*len(crack)

            for i, (cr, col) in enumerate(zip(crack, class_color)):
                col_tmp = hex2rgb(col)
                for x, y in cr:
                    for c in range(3):
                        img[y, x + i* GALLERY_SIZE, c] = col_tmp[c]
        return img

    def get_class_label(self, index, object_='primary__primary'):
        """Map prediction indices according to the class definition and
        return an array with the shape of index."""

        if isinstance(index, int):
            index = numpy.array([index])
        elif not isinstance(index, numpy.ndarray):
            index = numpy.array(index)

        index2label = self.definitions.class_definition(object_)["label"]
        predidx = self.get_class_prediction(object_)['label_idx']
        labels = numpy.ones(index.size, dtype=int)*CH5Const.UNPREDICTED_LABEL

        for i, j in enumerate(index.flatten()):
            try:
                labels[i] = index2label[predidx[j]]
            except IndexError as e:
                # unlabled objects
                pass

        return labels.reshape(index.shape)

    def get_center(self, index, object_='primary__primary'):
        if not isinstance(index, (list, tuple)):
            index = [index]
        center_list = self.get_feature_table(object_, 'center')[index]
        return center_list

    def get_class_color(self, index, object_='primary__primary'):
        if not self.has_classification(object_):
            return
        res = map(str, self.class_color_def(tuple(self.get_class_label(index, object_)), object_))
        if len(res) == 1:
            return res[0]
        return res

    def get_time_idx(self, index, object_='primary__primary'):
        return self['object'][object_][index]['time_idx']

    def get_time_idx2(self, index, object_='primary__primary'):
        return self['object'][object_]['time_idx'][index]

    def get_obj_label_id(self, index, object_='primary__primary'):
        return self['object'][object_][index]['obj_label_id']

    def get_time_indecies(self, index, object_='primary__primary'):
        inv_sort = numpy.argsort(numpy.argsort(numpy.array(index)))
        index.sort()
        tmp =  self['object'][object_][index]['time_idx']
        return tmp[inv_sort]

    def get_class_name(self, index, object_='primary__primary'):
        res = map(str, self.class_name_def(tuple(self.get_class_label(index)), object_))
        if len(res) == 1:
            return res[0]
        return res

    def class_color_def(self, class_labels, object_):
        class_mapping = self.definitions.class_definition(object_)
        return [class_mapping['color'][col-1] for col in class_labels]

    def class_name_def(self, class_labels, object_):
        class_mapping = self.definitions.class_definition(object_)
        return [class_mapping['name'][col-1] for col in class_labels]

    def object_feature_def(self, object_='primary__primary'):
        return map(lambda x: str(x[0]), self.definitions.feature_definition['%s/object_features' % object_].value)

    def get_object_table(self, object_):
        if len(self['object'][object_]) > 0:
            return self['object'][object_].value
        else:
            return []

    def get_feature_table(self, object_, feature):
        return self['feature'][object_][feature].value

    # @profile
    def get_events(self, output_second_branch=False, random=None):
        assert isinstance(output_second_branch, bool)
        assert isinstance(random, (type(None), int))

        evtable = self.get_object_table('event')
        event_ids = numpy.unique(evtable['obj_id'])

        if random is not None:
            numpy.random.shuffle(event_ids)
            event_ids = event_ids[:random]

        tracks = list()
        for event_id in event_ids:
            i = numpy.where(evtable['obj_id'] == event_id)[0]
            idx1 = evtable['idx1'][i]
            idx2 = evtable['idx2'][i]

            # find the index of the common elements in the array
            mc, occurence = collections.Counter(idx1).most_common(1)[0]

            if occurence == 1:
                track = numpy.hstack((idx1, idx2[-1]))
                tracks.append(track)
            elif occurence == 2:
                i1, i2 = numpy.where(idx1 == mc)[0]
                track = numpy.hstack((idx1[:i2], idx2[i2-1]))
                tracks.append(track)
                if output_second_branch:
                    track = numpy.hstack((idx1[:(i1+1)], idx2[i2:]))
                    tracks.append(track)
            else:
                raise RuntimeError(("Split events with more than 2 childs are "
                                    "not suppored. How did it get there anyway?"))

        return numpy.array(tracks)


    def get_event_items(self, output_second_branch=False):
        dset_event = self.get_object_table('event')
        events = []
        for event_id in range(dset_event['obj_id'].max()):
            idx = numpy.where(dset_event['obj_id'] == event_id)
            idx1 = dset_event[idx]['idx1']
            idx2 = dset_event[idx]['idx2']
            second_branch_found = False
            event_list = []
            for p1, _ in zip(idx1, idx2):
                if p1 in event_list:
                    # branch ends
                    second_branch_found = True
                    break
                else:
                    event_list.append(p1)
            if second_branch_found and output_second_branch:
                a = list(idx1).index(p1)
                b = len(idx1) - list(idx1)[-1:0:-1].index(p1) - 1
                event_list2 = list(idx1[0:a]) + list(idx1[b:])
                events.append((event_id, event_list, event_list2))
            else:
                events.append((event_id, event_list))
        return events

    def _track_single(self, start_idx, type_):
        track_on_feature = False
        if type_ == 'first':
            sel = 0
        elif type_ == 'last':
            sel = -1
        elif type_ == 'biggest':
            track_on_feature = True
            roisize_ind = [str(feature_name[0]) for feature_name in self.definitions.feature_definition['primary__primary']['object_features']].index('circularity')
            track_feature = self.get_feature_table('primary__primary', 'object_features')[:, roisize_ind]

        else:
            raise NotImplementedError('type not supported')

        dset_tracking = self.get_tracking()
        dset_tracking_idx2 = dset_tracking['obj_idx2']
        tracking_lookup_idx1 = self._get_tracking_lookup()

        idx_list = []
        idx = start_idx
        while True:
            if idx in tracking_lookup_idx1:
                next_p_idx = tracking_lookup_idx1[idx]
            else:
                break
            if track_on_feature:
                sel = numpy.argmax(track_feature[next_p_idx])
            idx = dset_tracking_idx2[next_p_idx[sel]]
            idx_list.append(idx)
        return idx_list


    def track_first(self, start_idx):
        return self._track_single(start_idx, 'first')

    def track_last(self, start_idx):
        return self._track_single(start_idx, 'last')

    def track_biggest(self, start_idx):
        return self._track_single(start_idx, 'biggest')

    def track_all(self, start_idx):
        dset_tracking = self.get_tracking()
        next_p_idx = (dset_tracking['obj_idx1']==start_idx).nonzero()[0]
        if len(next_p_idx) == 0:
            return [None]
        else:
            def all_paths_of_tree(id_):
                found_ids = dset_tracking['obj_idx2'][(dset_tracking['obj_idx1']==id_).nonzero()[0]]

                if len(found_ids) == 0:
                    return [[id_]]
                else:
                    all_paths_ = []
                    for out_id in found_ids:
                        for path_ in all_paths_of_tree(out_id):
                            all_paths_.append([id_] + path_)

                    return all_paths_

            head_ids = dset_tracking['obj_idx2'][(dset_tracking['obj_idx1']==start_idx).nonzero()[0]]
            res = []
            for head_id in head_ids:
                res.extend(all_paths_of_tree(head_id)   )
            return res


class CH5CachedPosition(CH5Position):
    def __init__(self, *args, **kwargs):
        super(CH5CachedPosition, self).__init__(*args, **kwargs)

    @memoize
    def get_prediction_probabilities(self, *args, **kwargs):
        return super(CH5CachedPosition, self).get_prediction_probabilities(
            *args, **kwargs)

    @memoize
    def get_events(self, *args, **kwargs):
        return super(CH5CachedPosition, self).get_events(*args, **kwargs)

    @memoize
    def get_object_table(self, *args, **kwargs):
        return super(CH5CachedPosition, self).get_object_table(*args, **kwargs)

    @memoize
    def get_feature_table(self, *args, **kwargs):
        return super(CH5CachedPosition, self).get_feature_table(*args, **kwargs)

    @memoize
    def get_tracking(self, *args, **kwargs):
        return super(CH5CachedPosition, self).get_tracking(*args, **kwargs)

    @memoize
    def _get_tracking_lookup(self, *args, **kwargs):
        return super(CH5CachedPosition, self)._get_tracking_lookup(*args, **kwargs)

    @memoize
    def get_class_prediction(self, object_='primary__primary'):
        return super(CH5CachedPosition, self).get_class_prediction(object_)

    @memoize
    def get_object_features(self, object_='primary__primary'):
        return super(CH5CachedPosition, self).get_object_features(object_)

    @memoize
    def get_gallery_image(self, *args, **kwargs):
        return super(CH5CachedPosition, self).get_gallery_image(*args, **kwargs)

    @memoize
    def class_name_def(self, class_labels, object_='primary__primary'):
        return super(CH5CachedPosition, self).class_name_def(class_labels, object_)

    @memoize
    def class_color_def(self, class_labels, object_='primary__primary'):
        return super(CH5CachedPosition, self).class_color_def(class_labels, object_)

    @memoize
    def object_feature_def(self, object_='primary__primary'):
        return super(CH5CachedPosition, self).object_feature_def(object_)

    @memoize
    def get_class_name(self, class_labels, object_='primary__primary'):
        return super(CH5CachedPosition, self).get_class_name(class_labels, object_)

    @memoize
    def get_class_color(self, class_labels, object_='primary__primary'):
        return super(CH5CachedPosition, self).get_class_color(class_labels, object_)

    def clear_cache(self):
        if hasattr(self, '_memoize__cache'):
            self._memoize__cache = {}


class CH5File(object):

    def __init__(self, filename, mode='r', cached=True):
        self.filename = filename
        self._cached = cached
        self._file_handle = h5py.File(filename, mode)
        self.plate = self._get_group_members('/sample/0/plate/')[0]
        self.wells = self._get_group_members('/sample/0/plate/%s/experiment/' % self.plate)
        self.positions = collections.OrderedDict()
        for w in sorted(self.wells):
            self.positions[w] = self._get_group_members('/sample/0/plate/%s/experiment/%s/position/' % (self.plate, w))

        self._position_group = {}
        for well, positions in self.positions.iteritems():
            for pos in positions:
                self._position_group[(well, pos)] = self._open_position(
                    self.plate, well, pos)

        self.current_pos = self._position_group.values()[0]

    def _open_position(self, plate, well, position):
        path = ("/sample/0/plate/%s/experiment/%s/position/%s"
                %(plate, well, position))

        try:
            if self._cached:
                return CH5CachedPosition(plate, well, position, path, self)
            else:
                return CH5Position(plate, well, position, path, self)
        except KeyError:
            warnings.warn(("Warning: cellh5 - well, position (%s, %s)"
                           "could not be loaded ") %(well, position))

    def get_position(self, well, pos):
        return self._position_group[(well, str(pos))]

    def get_file_handle(self):
        return self._file_handle

    def iter_positions(self):
        for well, positions in self.positions.items():
            for pos in positions:
                yield self._position_group[(well, pos)]

    def set_current_pos(self, well, pos):
        self.current_pos = self.get_position(well, pos)

    def _get_group_members(self, path):
        return map(str, self._file_handle[path].keys())

    @memoize
    def class_definition(self, object_):
        return self._file_handle['/definition/feature/%s/object_classification/class_labels' % object_].value

    @property
    def feature_definition(self):
        return self._file_handle['/definition/feature']

    @property
    def image_definition(self):
        return self._file_handle['/definition/image']

    @property
    def object_definition(self):
        return self._file_handle['/definition/object']

    def has_classification(self, object_):
        if object_ in self.feature_definition:
            return 'object_classification' in self.feature_definition[object_]
        else:
            return False

    def has_object_features(self, object_):
        if object_ in self.feature_definition:
            return 'object_features' in self.feature_definition[object_]
        else:
            return False

    def object_feature_def(self, object_='primary__primary'):
        return map(lambda x: str(x[0]), self.feature_definition['%s/object_features' % object_].value)

    def get_object_feature_idx_by_name(self, object_, feature_name):
        object_feature_names = self.object_feature_def(object_)
        return list(object_feature_names).index(feature_name)

    def close(self):
        self._file_handle.close()


class CH5MappedFile(CH5File):
    def read_mapping(self, mapping_file, sites=None, rows=None, cols=None, locations=None):
        self.mapping_file = mapping_file
        self.mapping = pandas.read_csv(self.mapping_file, sep='\t')

        if sites is not None:
            self.mapping = self.mapping[self.mapping['Site'].isin(sites)]
        if rows is not None:
            self.mapping = self.mapping[self.mapping['Row'].isin(rows)]
        if cols is not None:
            self.mapping = self.mapping[self.mapping['Column'].isin(cols)]

        if locations is not None:
            self.mapping = self.mapping[reduce(pandas.Series.__or__,
                                               [self.mapping['Row'].isin(c) & \
                                                    (self.mapping['Column'] == r)
                                                for c, r in locations])]

        self.mapping.reset_index(inplace=True)

    def _get_mapping_field_of_pos(self, well, pos, field):
        return self.mapping[(self.mapping['Well'] == str(well)) & \
             (self.mapping['Site'] == int(pos))][field].iloc[0]

    def get_group_of_pos(self, well, pos):
        return self._get_mapping_field_of_pos(well, pos, 'Group')

    def get_treatment_of_pos(self, well, pos, treatment_column=None):
        if treatment_column is None:
            treatment_column = ['siRNA ID', 'Gene Symbol']
        return self._get_mapping_field_of_pos(well, pos, treatment_column)


class CH5TestBase(unittest.TestCase):
    def setUp(self):
        data_filename = '../data/0038.ch5'
        if not os.path.exists(data_filename):
            raise IOError("No CellH5 test data found in 'cellh5/data'. Please refer to the instructions in 'cellh5/data/README'")
        self.fh = CH5File(data_filename)
        self.well_str = '0'
        self.pos_str = self.fh.positions[self.well_str][0]
        self.pos = self.fh.get_position(self.well_str, self.pos_str)

    def tearDown(self):
        self.fh.close()


class TestCH5Basic(CH5TestBase):
    def testGallery(self):
        a1 = self.pos.get_gallery_image(1)
        b1 = self.pos.get_gallery_image(2)
        a2 = self.pos.get_gallery_image(1)

        self.assertTrue(a1.shape == (GALLERY_SIZE, GALLERY_SIZE))
        self.assertTrue(b1.shape == (GALLERY_SIZE, GALLERY_SIZE))
        self.assertTrue(numpy.all(a1 == a2))
        self.assertFalse(numpy.all(a1 == b1))

    def testGallery2(self):
        event = self.pos.track_first(5)
        a1 = self.pos.get_gallery_image(tuple(event))
        a2 = self.pos.get_gallery_image(tuple(event), 'secondary__expanded')

    def testGallery3(self):
        event = self.pos.get_events()[42][0]
        tracks = self.pos.track_all(event)
        w = numpy.array(map(len, tracks)).max()*GALLERY_SIZE
        img = numpy.zeros((GALLERY_SIZE * len(tracks), w), dtype=numpy.uint8)

        for k, t in enumerate(tracks):
            a = self.pos.get_gallery_image(tuple(t))
            img[k*GALLERY_SIZE:(k+1)*GALLERY_SIZE, 0:a.shape[1]] = a

    def testGallery4(self):
        event = self.pos.get_events()[42]
        a1 = self.pos.get_gallery_image(tuple(event))

    def testClassNames(self):
        for x in ['inter', 'pro', 'earlyana']:
            self.assertTrue(x in self.pos.class_name_def((1,2,5)))

    def testClassColors(self):
        for x in ['#FF8000', '#D28DCE', '#FF0000']:
            self.assertTrue(x in self.pos.class_color_def((3,4,8)))

    def testClassColors2(self):
        self.pos.get_class_color((1,221,3233,44244))
        self.pos.get_class_name((1,221,3233,44244))

    def testEvents(self):
        self.assertTrue(len(self.pos.get_events()) > 0)
        self.assertTrue(len(self.pos.get_events()[0]) > 0)

    def testTrack(self):
        self.assertTrue(len(self.pos.track_first(42)) > 0)

    def testTrackFirst(self):
        self.assertListEqual(self.pos.track_first(42),
                             self.pos.track_all(42)[0])

    def testTrackLast(self):
        self.assertListEqual(self.pos.track_last(1111),
                             self.pos.track_all(1111)[-1])

    def testObjectFeature(self):
        self.assertTrue('n2_avg' in  self.pos.object_feature_def())
        self.assertTrue(self.pos.get_object_features().shape[1] == 239)

class TestCH5Examples(CH5TestBase):
    def testGalleryMatrix(self):
        image = self.pos.get_gallery_image_matrix(range(20), (5,6))
        import vigra
        vigra.impex.writeImage(image.swapaxes(1,0), 'img_matrix.png')


    def testReadAnImage(self):
        """Read an raw image an write a sub image to disk"""
        # read the images at time point 1
        h2b = self.pos.get_image(0, 0)
        tub = self.pos.get_image(0, 1)

        # Print part of the images prepare image plot
        fig = mpl.figure(frameon=False)
        ax = mpl.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(h2b[400:600, 400:600], cmap='gray')
        fig.savefig('img1.png', format='png')
        ax.imshow(tub[400:600, 400:600], cmap='gray')

    # unittest.skip('ploting so many lines is very slow in matplotlib')
    def testPrintTrackingTrace(self):
        """Show the cell movement over time by showing the trace of each cell colorcoded
           overlayed on of the first image"""
        h2b = self.pos.get_image(0, 0)

        tracking = self.pos.get_object_table('tracking')
        nucleus = self.pos.get_object_table('primary__primary')
        center = self.pos.get_feature_table('primary__primary', 'center')


        # prepare image plot
        fig = mpl.figure(frameon=False)
        ax = mpl.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(h2b, cmap='gray')

        # on top of the image a white circle is plotted for each center of nucleus.
        I = numpy.nonzero(nucleus[tracking['obj_idx1']]['time_idx'] == 0)[0]
        for x, y in center[I]:
            ax.plot(x,y,'w.', markersize=7.0, scaley=False, scalex=False)

        ax.axis([0, h2b.shape[1], h2b.shape[0], 0])

        # a line is plotted between nucleus center of each pair of connected nuclei. The color is the mitotic phase
        for idx1, idx2 in zip(tracking['obj_idx1'],
                              tracking['obj_idx2']):
            color = self.pos.get_class_color(idx1)
            (x0, y0), (x1, y1) = center[idx1], center[idx2]
            ax.plot([x0, x1],[y0, y1], color=color)

        fig.savefig('tracking.png', format='png')

    # unittest.skip('ploting so many lines is very slow in matplotlib')
    def testComputeTheMitoticIndex(self):
        """Read the classification results and compute the mitotic index"""

        nucleus = self.pos.get_object_table('primary__primary')
        predictions = self.pos.get_class_prediction('primary__primary')

        colors = self.pos.definitions.class_definition('primary__primary')['color']
        names = self.pos.definitions.class_definition('primary__primary')['name']

        n_classes = len(names)
        time_max = nucleus['time_idx'].max()

        # compute mitotic index by counting the number cell per class label over all times
        mitotic_index =  numpy.array(map(lambda x: [len(numpy.nonzero(x==class_idx)[0]) for class_idx in range(n_classes)],
            [predictions[nucleus['time_idx'] == time_idx]['label_idx'] for time_idx in range(time_max)]))

        # plot it
        fig = mpl.figure()
        ax = fig.add_subplot(111)

        for i in range(1, n_classes):
            ax.plot(mitotic_index[:,i], color=colors[i], label=names[i])

        ax.set_xlabel('time')
        ax.set_ylabel('number of cells')
        ax.set_title('Mitotic index')
        ax.set_xlim(0, time_max)
        ax.legend(loc='upper left')
        fig.savefig('mitotic_index.pdf', format='pdf')

    def testShowMitoticEvents(self):
        """Extract the mitotic events and write them as gellery images"""
        events = self.pos.get_events()

        image = []
        for event in events[:5]:
            image.append(self.pos.get_gallery_image(tuple(event)))


if __name__ == '__main__':
    unittest.main()
