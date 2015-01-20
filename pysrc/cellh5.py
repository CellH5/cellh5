"""
    The CellH5 Project
    Copyright (c) 2012 - 2015 Christoph Sommer, Michael Held, Bernd Fischer
    Gerlich Lab, IMBA Vienna, Huber Lab, EMBL Heidelberg
    www.cellh5.org

    CellH5 is distributed under the LGPL License.
    See LICENSE.txt for details.
    See AUTHORS.txt for author contributions.
"""

import sys
import os
import numpy
import h5py
import pandas
import zlib
import matplotlib.pyplot as plt

import base64
import warnings
import unittest
import datetime

import functools
import collections

from itertools import chain, izip
from collections import OrderedDict
from contextlib import contextmanager

from matplotlib.colors import hex2color
from hmm_wrapper import HMMConstraint, HMMAgnosticEstimator, normalize, hmm 

version_num = (1, 2, 0)
version = '.'.join([str(n) for n in version_num])
ICON_FILE = os.path.join(os.path.split(__file__)[0], "cellh5_icon.ico")

GALLERY_SIZE = 80

import logging
log = logging.getLogger(__name__)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

#####################
### Helper functions
#####################

def pandas_apply(df, func):
    """Helper function for pandas DataFrame, when dealing with entries, which are numpy multi-dim arrays.
       Note, pandas integrated apply fun will drive you nuts...
       
       df: pandas DataFrame
       func: function expecting a row a Series over the columns of df
       
       returns list of row results
    """
    res = []
    for i, row in df.iterrows():
        res.append(func(row))
    if isinstance(res[0], (tuple,)):
        return zip(*res)
    else:
        return res
    
def pandas_ms_apply(df, func, n_cores=10):
    from multiprocessing import Pool
    """Helper function for pandas DataFrame, when dealing with entries, which are numpy multi-dim arrays.
       Note, pandas integrated apply fun will drive you nuts...
       
       IMPORTANT: differences to single core: pandas_apply
          1) func must live at the module level
          2) all extra args to func (e.g. used with partial) must be pickable
       
       df: pandas DataFrame
       func: function expecting a row a Series over the columns of df
       
       returns list of row results
    """
    data = []
    for _, row in df.iterrows():
        data.append(row)
    
    pool = Pool(processes=n_cores)
    res = pool.map(func, data)
    pool.close()
    if isinstance(res[0], (tuple,)):
        return zip(*res)
    else:
        return res

def repack_cellh5(cellh5_folder, output_file=None):
    """Copies a cellh5 folder well-based into one single postition file"""
    if output_file is None:
        output_file = '%s/_all_positions_with_data.ch5' % cellh5_folder
  
    import glob, re
    PLATE_PREFIX = '/sample/0/plate/'
    WELL_PREFIX = PLATE_PREFIX + '%s/experiment/'
    POSITION_PREFIX = WELL_PREFIX + '%s/position/'

    def get_plate_and_postion(hf_file):
        plate = hf_file[PLATE_PREFIX].keys()[0]
        well = hf_file[WELL_PREFIX % plate].keys()[0]
        position = hf_file[POSITION_PREFIX % (plate, well)].keys()[0]
        return plate, well, position

    flist = sorted(glob.glob('%s/*.ch5' % cellh5_folder))

    f = h5py.File(output_file, 'w')

    reg = re.compile('^[A-Z]\d{2}_\d{2}')
    cnt = 0
    for fname in flist:
        if reg.search(os.path.split(fname)[1]) is not None:
            print cnt, fname
            if cnt == 0:
                # write definition
                fh = h5py.File(fname, 'r')
                f.copy(fh['/definition'], 'definition')
                fh.close()
            # copy suff
            fh = h5py.File(fname, 'r')
            fplate, fwell, fpos = get_plate_and_postion(fh)
            # print (POSITION_PREFIX + '%s') % (fplate, fwell, fpos)
            f.copy(fh[(POSITION_PREFIX + '%s') % (fplate, fwell, fpos)], (POSITION_PREFIX + '%s') % (fplate, fwell, fpos))
            fh.close()
            cnt += 1
    f.close()

def repack_cellh5_and_combine(cellh5_folder, cellh5_folder_2, rel_path_src, rel_path_dest):
    """Copies a cellh5 folder wellbased into one single postition file
       and copies stuff from another cellh5 into that one (usefull if the same exp
       ran twice)
    """
    import glob, re
    PLATE_PREFIX = '/sample/0/plate/'
    WELL_PREFIX = PLATE_PREFIX + '%s/experiment/'
    POSITION_PREFIX = WELL_PREFIX + '%s/position/'

    def get_plate_and_postion(hf_file):
        plate = hf_file[PLATE_PREFIX].keys()[0]
        well = hf_file[WELL_PREFIX % plate].keys()[0]
        position = hf_file[POSITION_PREFIX % (plate, well)].keys()[0]
        return plate, well, position

    flist = sorted(glob.glob('%s/*.ch5' % cellh5_folder))

    f = h5py.File('%s/_all_positions_with_data_combined.ch5' % cellh5_folder, 'w')

    reg = re.compile('^[A-Z]\d{2}_\d{2}')
    cnt = 0
    for fname in flist:
        if reg.search(os.path.split(fname)[1]) is not None:
            print cnt, fname
            if cnt == 0:
                # write definition
                fh = h5py.File(fname, 'r')
                fh_2 = h5py.File(os.path.join(cellh5_folder_2, os.path.split(fname)[1]), 'r')
                f.copy(fh['/definition'], 'definition')
                for rps, rpd in zip(rel_path_src, rel_path_dest):
                    f.copy(fh_2['/definition/%s' % rps], 'definition/%s' % rpd)

                fh.close()
                fh_2.close()
            # copy suff
            fh = h5py.File(fname, 'r')
            fh_2 = h5py.File(os.path.join(cellh5_folder_2, os.path.split(fname)[1]), 'r')
            fplate, fwell, fpos = get_plate_and_postion(fh)
            # print (POSITION_PREFIX + '%s') % (fplate, fwell, fpos)
            pos_path_in_ch5 = (POSITION_PREFIX + '%s') % (fplate, fwell, fpos)
            f.copy(fh[pos_path_in_ch5], pos_path_in_ch5)
            for rps, rpd in zip(rel_path_src, rel_path_dest):
                f.copy(fh_2[pos_path_in_ch5 + ("/%s" % rps)], pos_path_in_ch5 + ("/%s" % rpd))

            fh.close()
            fh_2.close()
            cnt += 1

    f.close()

def hex2rgb(color, mpl=False):
    """Return the rgb color as python int in the range 0-255."""
    assert color.startswith("#")
    if mpl:
        fac = 1.0
    else:
        fac = 255.0

    rgb = [int(i * fac) for i in hex2color(color)]
    return tuple(rgb)

def to_index_array(value):
    """Convert list or tuple into an numpy.array. If value is integral, its
    packed into a list first.
    """
    if isinstance(value, numpy.ndarray):
        return value
    elif isinstance(value, (list, tuple)):
        return numpy.array(value)
    else:
        return numpy.array([value])

@contextmanager
def ch5open(filename, mode='r', cached=True):
    """Open a cellh5 file using the with statement. The file handle is closed
    automatically.

    >>>with ch5open('datafile.ch5', 'r', cached=False) as ch5:
    >>>   ch5.get_position("0", "0038")
    """

    ch5 = CH5File(filename, mode, cached)
    yield ch5
    ch5.close()

class CH5Const(object):
    """Container class for constants"""
    # defaults for unpredicted objects, -1 one might be not a
    UNPREDICTED_LABEL = -99
    UNPREDICTED_PROB = numpy.nan

    REGION = 'region'
    RELATION = 'relation'

class CH5GroupCoordinate(object):
    """CH5 Coordinates, sample, plate, well, position, region"""
    def __init__(self, region, position, well, plate, sample="0"):
        super(CH5GroupCoordinate, self).__init__()

        self.region = region
        self.position = position
        self.well = well
        self.plate = plate
        self.sample = sample

class memoize(object):
    """Cache the return value of a method
       make sure the arguments are hashable
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
        if not hasattr(obj, "_cache"):
            obj._cache = dict()

        try:
            res = obj._cache[key]
        except KeyError:
            res = self.func(*args, **kw)
            obj._cache[key] = res
        except TypeError:
            # if key is not hashable (e.g ndarrays)
            res = self.func(*args, **kw)

        return res

class CH5Position(object):
    """Main class for interacting with CH5 objects"""
    def __init__(self, plate, well, pos, grp_pos, parent):
        self.plate = plate
        self.well = well
        self.pos = pos
        self.grp_pos_path = grp_pos
        self.definitions = parent

    def __getitem__(self, key):
        path = "%s/%s" % (self.grp_pos_path, key)
        return self.definitions.get_file_handle()[path]

    def channel_color_by_region(self, region):
        """Return the the channel information."""

        path = '/definition/image/region'
        rdef = self.definitions.get_file_handle()[path].value

        i = rdef['channel_idx'][rdef['region_name'] == 'region___%s' % region][0]
        path = '/definition/image/channel'
        return self.definitions.get_file_handle()[path]['color'][i]

    def get_tracking(self):
        return self['object']['tracking'].value

    def _get_tracking_lookup(self, tracking_table, obj_idx='obj_idx1', ):
        dset_tracking_idx1 = tracking_table[obj_idx]
        tracking_lookup_idx1 = collections.defaultdict()
        for i, o in enumerate(dset_tracking_idx1):
            tracking_lookup_idx1.setdefault(o, []).append(i)
        return tracking_lookup_idx1

    def get_class_prediction(self, object_='primary__primary'):
        path = 'feature/%s/object_classification/prediction' % object_
        return self[path]

    def get_prediction_probabilities(self, indices=None,
                                     object_="primary__primary"):
        path = 'feature/%s/object_classification/probability' % object_

        if indices is None:
            return self[path].value
        else:
            # read probs only once per cell, reading from share is too slow
            ishape = indices.shape
            indices = indices.flatten()
            indices2 = numpy.unique(indices)
            _, nclasses = self[path].shape
            probs = numpy.empty((indices.size, nclasses))

            for i, j in enumerate(indices2):
                k = numpy.where(indices == j)[0]
                probs[k, :] = self[path][j, :]

            return probs.reshape(ishape + (nclasses,))

    def has_classification(self, object_):
        fh = self.definitions.get_file_handle()
        path = 'definition/feature/%s/object_classification' % object_
        return path in fh

    def get_crack_contour(self, index, object_='primary__primary',
                          bb_corrected=True, size=GALLERY_SIZE):
        index = to_index_array(index)
        crack_list = []
        for ind in index:
            crack_str = self['feature'][object_]['crack_contour'][ind]
            crack = numpy.asarray(zlib.decompress(\
                             base64.b64decode(crack_str)).split(','), \
                             dtype=numpy.float32).reshape(-1, 2)

            if bb_corrected:
                bb = self['feature'][object_]['center'][ind]
                crack[:, 0] -= bb['x'] - size / 2
                crack[:, 1] -= bb['y'] - size / 2
                crack = crack.clip(0, size-1)

            crack_list.append(crack)

        return crack_list

    def has_object_entries(self, object_='primary__primary'):
        return len(self['object'][object_]) > 0

    def get_object_features(self, object_='primary__primary', index=None):
        if index is not None and len(index) == 0:
            return []
        if len(self['feature'][object_]['object_features']) > 0:
            if index is None:
                return self['feature'][object_]['object_features'].value
            else:
                return self['feature'][object_]['object_features'][index, :]
                
        else:
            return []           
        
    def get_object_feature_by_name(self, name, object_='primary__primary'):   
        if len(self['feature'][object_][name]) > 0:
            return self['feature'] \
                   [object_] \
                   [name].value
        else:
            return []     
        
    def get_time_of_frame(self, frame):
        return self['image/time_lapse']['timestamp_rel'][frame]
                   
    def get_time_lapse(self):
        if 'time_lapse' in self['image']:
            time_stamps = self['image/time_lapse']['timestamp_rel']
            time_lapses = numpy.diff(time_stamps)
            time_lapse = time_lapses.mean()
        else:
            time_lapse = None
        return time_lapse
    
    def get_object_idx(self, object_='primary__primary', frame=None):
        ot = self.get_object_table(object_)
        if frame is None:
            return ot
        else:
            return numpy.nonzero(ot['time_idx'] == frame)[0]

    def get_image(self, t, c, z=0):
        return self['image'] \
                    ['channel'] \
                    [c, t, z, :, :]

    def get_gallery_image(self, index,
                          object_='primary__primary', size=GALLERY_SIZE):
        index = to_index_array(index)
        images = list()

        channel_idx = self.definitions.image_definition['region'] \
            ['channel_idx'][self.definitions.image_definition['region']['region_name'] == 'region___%s' % object_][0]

        image_width = self['image']['channel'].shape[3]
        image_height = self['image']['channel'].shape[4]
        centers = self['feature'][object_]['center'][index.tolist()]
        size_2 = size / 2
        for i, cen in izip(index, centers):
            time_idx = self['object'][object_][i]['time_idx']
            tmp_img = self['image']['channel'][channel_idx, time_idx, 0,
                                               max(0, cen[1] - size_2):min(image_width, cen[1] + size_2),
                                               max(0, cen[0] - size_2):min(image_height, cen[0] + size_2)]
            
            if tmp_img.shape != (size, size):
                image = numpy.zeros((size, size), dtype=numpy.uint8)
                image[(image.shape[0] - tmp_img.shape[0]):, :tmp_img.shape[1]] = tmp_img
                images.append(image)
            else:
                images.append(tmp_img)

        if len(index) > 1:
            return numpy.concatenate(images, axis=1)
        return images[0]
    
    def get_gallery_image_with_class(self, index, object_=('primary__primary',), color=None):
        if len(object_) == 1:
            img_ = self.get_gallery_image(index, object_[0])
            rgb_shape = img_.shape + (3,)
            img = numpy.zeros(rgb_shape, img_.dtype)
            for c in range(3): img[:, :, c] = img_
        else:

            for c in range(3):
                if c == 0:
                    img_ = self.get_gallery_image(index, object_[c])
                    rgb_shape = img_.shape + (3,)
                    img = numpy.zeros(rgb_shape, img_.dtype)
                    img[:, :, 0] = img_
                if 0 < c < len(object_):
                    img[:, :, c] = self.get_gallery_image(index, object_[c])
        
        if color is None:
            class_color = self.get_class_color(index)
        else:
            class_color = color

        col_tmp = hex2rgb(class_color)
        for c in range(3):
            img[:10, :10, c] = col_tmp[c]

        
        return img

    def get_gallery_image_rgb(self, index, object_=('primary__primary',)):
        if len(object_) == 1:
            img_ = self.get_gallery_image(index, object_[0])
            rgb_shape = img_.shape + (3,)
            img = numpy.zeros(rgb_shape, img_.dtype)
            for c in range(3): img[:, :, c] = img_
            return img

        for c in range(3):
            if c == 0:
                img_ = self.get_gallery_image(index, object_[c])
                rgb_shape = img_.shape + (3,)
                img = numpy.zeros(rgb_shape, img_.dtype)
                img[:, :, 0] = img_
            if 0 < c < len(object_):
                img[:, :, c] = self.get_gallery_image(index, object_[c])

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
            image = numpy.zeros((GALLERY_SIZE, GALLERY_SIZE), dtype=numpy.uint8)

            tmp_img = self['image/channel'][channel_idx, time_idx, 0,
                                 max(0, cen1[1] - GALLERY_SIZE / 2):min(image_width, cen1[1] + GALLERY_SIZE / 2),
                                 max(0, cen1[0] - GALLERY_SIZE / 2):min(image_height, cen1[0] + GALLERY_SIZE / 2)]

            image[(image.shape[0] - tmp_img.shape[0]):, :tmp_img.shape[1]] = tmp_img
            image_list.append(image)

        return image_list

    def get_gallery_image_generator(self, index, object_='primary__primary'):
        channel_idx = self.definitions.image_definition['region']['channel_idx'][self.definitions.image_definition['region']['region_name'] == 'region___%s' % object_][0]
        image_width = self['image']['channel'].shape[3]
        image_height = self['image']['channel'].shape[4]
        
        try:
            test_iter = iter(index)
        except TypeError, te:
            index = [index]

        for ind in index:
            time_idx = self['object'][object_][ind]['time_idx']
            cen1 = self['feature'][object_]['center'][ind]
            image = numpy.zeros((GALLERY_SIZE, GALLERY_SIZE, 3), dtype=numpy.uint8)

            tmp_img = self['image/channel'][channel_idx, time_idx, 0,
                                 max(0, cen1[1] - GALLERY_SIZE / 2):min(image_width, cen1[1] + GALLERY_SIZE / 2),
                                 max(0, cen1[0] - GALLERY_SIZE / 2):min(image_height, cen1[0] + GALLERY_SIZE / 2)]

            for c in range(3):
                image[(image.shape[0] - tmp_img.shape[0]):, :tmp_img.shape[1], c] = tmp_img
            
            crack = self.get_crack_contour(ind, object_)
            class_color = ['#FFFF00'] * len(crack)
            for i, (cr, col) in enumerate(zip(crack, class_color)):
                col_tmp = hex2rgb(col)
                for x, y in cr:
                    for c in range(3):
                        y = min(y, GALLERY_SIZE-1)
                        x = min(x, GALLERY_SIZE-1)
                        image[y, x + i * GALLERY_SIZE, c] = col_tmp[c]
            
            
            yield image

    def get_gallery_image_matrix(self, index, shape, object_='primary__primary'):
        image = numpy.zeros((GALLERY_SIZE * shape[0],
                             GALLERY_SIZE * shape[1], 3), dtype=numpy.uint8)
        i, j = 0, 0
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

                if (c, d) > image.shape:
                    break
                image[a:c, b:d, :] = img

        return image

    def get_gallery_image_matrix_with_classification(self, index, shape, object_='primary__primary'):
        image = numpy.zeros((GALLERY_SIZE * shape[0],
                             GALLERY_SIZE * shape[1], 3), dtype=numpy.uint8)
        i, j = 0, 0
        img_gen = self.get_gallery_image_generator(index, object_)
        class_colors = self.get_class_color(index)
        cnt = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                try:
                    img = img_gen.next()
                    color = class_colors[cnt]
                    cnt += 1
                except StopIteration:
                    break
                a = i * GALLERY_SIZE
                b = j * GALLERY_SIZE
                c = a + GALLERY_SIZE
                d = b + GALLERY_SIZE

                if (c, d) > image.shape:
                    break
                for c in range(3): image[a:c, b:d, c] = img
                
                col_rgb = hex2rgb(color)
                for c in range(3): image[a:a + 10, b:b + 10, c] = col_rgb[c]

        return image

    def get_gallery_image_contour(self, index, object_=('primary__primary',), color=None, scale=None):
        img = self.get_gallery_image_rgb(index, object_)
        if scale is not None:
            img = numpy.clip(img.astype(numpy.float32) * scale, 0, 255).astype(numpy.uint8)
        for obj_id in object_:
            crack = self.get_crack_contour(index, obj_id)

            if color is None:
                class_color = self.get_class_color(index, obj_id)
                if class_color is None:
                    class_color = ['#FFFFFF'] * len(crack)

                if not isinstance(class_color, (list, tuple)):
                    class_color = [class_color]
            else:
                class_color = [color] * len(crack)

            for i, (cr, col) in enumerate(zip(crack, class_color)):
                col_tmp = hex2rgb(col)
                for x, y in cr:
                    for c in range(3):
                        img[y, x + i * GALLERY_SIZE, c] = col_tmp[c]
        return img

    def get_class_label(self, index, object_='primary__primary'):
        """Map prediction indices according to the class definition and
        return an array with the shape of index."""
        index = to_index_array(index)

        index2label = self.definitions.class_definition(object_)["label"]
        predidx = self.get_class_prediction(object_)['label_idx']
        labels = numpy.ones(index.size, dtype=int) * CH5Const.UNPREDICTED_LABEL

        for i, j in enumerate(index.flatten()):
            try:
                labels[i] = index2label[predidx[j]]
            except IndexError as e:
                # unlabled objects
                pass

        return labels.reshape(index.shape)
    
    def get_class_label_index(self, index, object_='primary__primary'):
        """return prediction indices """
        index = to_index_array(index)
        predidx = self.get_class_prediction(object_)['label_idx'][index]

        return predidx

    def get_center(self, index, object_='primary__primary'):
        index = to_index_array(index)
        center_list = self.get_feature_table(object_, 'center')[index]
        return center_list
    
    def get_orientation(self, index, object_='primary__primary'):
        index = to_index_array(index)
        angle_list = self.get_feature_table(object_, 'orientation')[index]['angle']
        return angle_list

    def get_class_color(self, index, object_='primary__primary'):
        if not self.has_classification(object_):
            return

        res = map(str, self.class_color_def(tuple(self.get_class_label(index, object_)), object_))
        if len(res) == 1:
            return res[0]
        return res

    def get_all_time_idx(self, object_='primary__primary'):
        return self['object'][object_][:]['time_idx']

    def get_time_idx(self, index, object_='primary__primary'):
        return self['object'][object_][index]['time_idx']

    def get_time_idx2(self, index, object_='primary__primary'):
        return self['object'][object_]['time_idx'][index]

    def get_obj_label_id(self, index, object_='primary__primary'):
        return self['object'][object_][index]['obj_label_id']

    def get_time_indecies(self, index, object_='primary__primary'):
        inv_sort = numpy.argsort(numpy.argsort(numpy.array(index)))
        index.sort()
        tmp = self['object'][object_][index]['time_idx']
        return tmp[inv_sort]

    def get_class_name(self, index, object_='primary__primary'):
        res = map(str, self.class_name_def(tuple(self.get_class_label(index)), object_))
        if len(res) == 1:
            return res[0]
        return res

    def class_color_def(self, class_labels, object_='primary__primary'):
        label2color = OrderedDict()
        class_mapping = self.definitions.class_definition(object_)
        for cm in range(len(class_mapping)):
            label2color[class_mapping["label"][cm]] = class_mapping["color"][cm] 
        return [label2color[cl] for cl in class_labels]

    def class_name_def(self, class_labels, object_):
        label2name = OrderedDict()
        class_mapping = self.definitions.class_definition(object_)
        for cm in range(len(class_mapping)):
            label2name[class_mapping["label"][cm]] = class_mapping["name"][cm] 
        return [label2name[cl] for cl in class_labels]

    def object_feature_def(self, object_='primary__primary'):
        return map(lambda x: str(x[0]),
                   self.definitions.feature_definition['%s/object_features' % object_].value)
        
    def get_object_table(self, object_):
        if len(self['object'][object_]) > 0:
            return self['object'][object_].value
        else:
            return []

    def get_feature_table(self, object_, feature):
        return self['feature'][object_][feature].value

    def has_events(self):
        # either group is emtpy or key does not exist
        try:
            return bool(self['object/event'].size)
        except KeyError:
            return False

    def get_events(self, output_second_branch=False, random=None):
        assert isinstance(output_second_branch, bool)
        assert isinstance(random, (type(None), int))

        evtable = self.get_object_table('event')
        if len(evtable) == 0:
            return numpy.array([]) 
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
                track = numpy.hstack((idx1[:i2], idx2[i2 - 1]))
                tracks.append(track)
                if output_second_branch:
                    track = numpy.hstack((idx1[:(i1 + 1)], idx2[i2:]))
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


    def _track_single(self, start_idx, type_, max_length=None):
        track_on_feature = False
        if type_ == 'first':
            sel = 0
        elif type_ == 'last':
            sel = -1
        elif type_ == 'biggest':
            track_on_feature = True
            roisize_ind = [str(feature_name[0]) for feature_name in self.definitions.feature_definition['primary__primary']['object_features']].index('roisize')
            track_feature = self.get_feature_table('primary__primary', 'object_features')[:, roisize_ind]

        else:
            raise NotImplementedError('type not supported')

        dset_tracking = self.get_tracking()
        dset_tracking_idx2 = dset_tracking['obj_idx2']
        tracking_lookup_idx1 = self._get_tracking_lookup(dset_tracking)

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
            if max_length is not None and len(idx_list) > max_length - 1:
                break

        return idx_list
    
    def _track_backwards_single(self, end_idx, type_, max_length=None):
        track_on_feature = False
        if type_ == 'first':
            sel = 0
        elif type_ == 'last':
            sel = -1
        elif type_ == 'biggest':
            track_on_feature = True
            roisize_ind = [str(feature_name[0]) for feature_name in self.definitions.feature_definition['primary__primary']['object_features']].index('roisize')
            track_feature = self.get_feature_table('primary__primary', 'object_features')[:, roisize_ind]

        else:
            raise NotImplementedError('type not supported')

        dset_tracking = self.get_tracking()
        dset_tracking_idx2 = dset_tracking['obj_idx1']
        tracking_lookup_idx1 = self._get_tracking_lookup(dset_tracking, 'obj_idx2')

        idx_list = []
        idx = end_idx
        while True:
            if idx in tracking_lookup_idx1:
                next_p_idx = tracking_lookup_idx1[idx]
            else:
                break
            if track_on_feature:
                sel = numpy.argmax(track_feature[next_p_idx])
            idx = dset_tracking_idx2[next_p_idx[sel]]
            idx_list.append(idx)
            if max_length is not None and len(idx_list) > max_length - 1:
                break
        
        idx_list.reverse()

        return idx_list
    
    def del_object_feature_data(self, feature_name, object_='primary__primary'):
        # Check file mode
        if self.definitions._file_handle.mode not in ('w', 'a', 'r+'):
            raise IOError('Error: Cannot write to CellH5 file, since it is opened read-only')
        path = 'feature/%s/' % object_
        feature_grp = self[path]
        
        if feature_name in feature_grp:
            del feature_grp[feature_name]
        
    def set_object_feature_data(self, feature_name, data, object_='primary__primary', overwrite=True):
        # Check file mode
        if self.definitions._file_handle.mode not in ('w', 'a', 'r+'):
            raise IOError('Error: Cannot write to CellH5 file, since it is opened read-only')
        
        path = 'feature/%s/' % object_
        feature_grp = self[path]
        
        if feature_name in feature_grp:
            if overwrite:
                "Waring: %s already set in %s... overwrite" % (feature_name, path)
                del feature_grp[feature_name]
            else:
                IOError("Error: %s already set in %s... overwrite" % (feature_name, path))
        
        try:
            feature_grp.create_dataset(feature_name, data=data)
        except:
            print "Error: Creation of %s in %s failed " % (feature_name, path)
            raise

    def track_first(self, start_idx, max_length=None):
        """Track an object based on the tracking information to the end. 
           In case of spits, take the first element randomly"""
        return self._track_single(start_idx, 'first', max_length=max_length)

    def track_last(self, start_idx, max_length=None):
        return self._track_single(start_idx, 'last', max_length=max_length)

    def track_biggest(self, start_idx, max_length=None):
        """Track an object based on the tracking information to the end. 
           In case of spits, take the biggest element"""
        return self._track_single(start_idx, 'biggest', max_length=max_length)
    
    def track_backwards(self, end_idx, max_length=None):
        return self._track_backwards_single(end_idx, 'first', max_length=max_length)

    def track_all(self, start_idx):
        dset_tracking = self.get_tracking()
        next_p_idx = (dset_tracking['obj_idx1'] == start_idx).nonzero()[0]
        if len(next_p_idx) == 0:
            return [None]
        else:
            def all_paths_of_tree(id_):
                found_ids = dset_tracking['obj_idx2'][(dset_tracking['obj_idx1'] == id_).nonzero()[0]]

                if len(found_ids) == 0:
                    return [[id_]]
                else:
                    all_paths_ = []
                    for out_id in found_ids:
                        for path_ in all_paths_of_tree(out_id):
                            all_paths_.append([id_] + path_)

                    return all_paths_

            head_ids = dset_tracking['obj_idx2'][(dset_tracking['obj_idx1'] == start_idx).nonzero()[0]]
            res = []
            for head_id in head_ids:
                res.extend(all_paths_of_tree(head_id))
            return res

class CH5CachedPosition(CH5Position):
    """Same as CH5Position using a cache for all inhereted methods"""
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
    def get_object_idx(self, *args, **kwargs):
        return super(CH5CachedPosition, self).get_object_idx(*args, **kwargs)

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
    def get_object_features(self, *args, **kwargs):
        return super(CH5CachedPosition, self).get_object_features(*args, **kwargs)

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
    """CH5File object to open CH5 files"""
    def __init__(self, filename, mode='a', cached=True):
        self._cached = cached
        if isinstance(filename, basestring):
            self.filename = filename
            self._file_handle = h5py.File(filename, mode)
        else:
            self._file_handle = filename
            self.filename = filename.filename    
        
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
                % (plate, well, position))

        try:
            if self._cached:
                return CH5CachedPosition(plate, well, position, path, self)
            else:
                return CH5Position(plate, well, position, path, self)
        except KeyError:
            warnings.warn(("Warning: cellh5 - well, position (%s, %s)"
                           "could not be loaded ") % (well, position))

    def get_position(self, well, pos):
        return self._position_group[(well, str(pos))]
    
    def has_position(self, well, pos):
        return (well, str(pos)) in self._position_group

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
    def class_definition(self, object_="primary__primary"):
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

    
    def gallery_image_matrix_gen(self, index_tpl, object_='primary__primary'):
        gen_list = []      
        for well, pos, index in index_tpl:
            ch5pos = self.get_position(well, pos)
            gen_list.append(ch5pos.get_gallery_image_generator(index, object_))
        img_gen = chain.from_iterable(gen_list)
        return img_gen
    
    @staticmethod
    def gallery_image_matrix_layouter(img_gen, shape):
        image = numpy.zeros((GALLERY_SIZE * shape[0], GALLERY_SIZE * shape[1]), dtype=numpy.uint8)
        i, j = 0, 0    
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
                
                if (c, d) > image.shape:
                    break
                image[a:c, b:d] = img  
        return image 
    
    @staticmethod
    def gallery_image_matrix_layouter_rgb(img_gen, shape):
        image = numpy.zeros((GALLERY_SIZE * shape[0], GALLERY_SIZE * shape[1], 3), dtype=numpy.uint8)
        i, j = 0, 0    
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
                
                if (c, d) > image.shape:
                    break
                image[a:c, b:d,:] = img  
        return image 
    
    def get_gallery_image_matrix(self, index_tpl, shape, object_='primary__primary'):
        img_gen = self.gallery_image_matrix_gen(index_tpl=index_tpl, object_=object_)
        return CH5File.gallery_image_matrix_layouter(img_gen, shape)
    
    def get_gallery_image_matrix_with_classification(self, index_tpl, shape, object_='primary__primary'):
        img_gen = self.gallery_image_matrix_gen(index_tpl=index_tpl, object_=object_)
        return CH5File.gallery_image_matrix_layouter(img_gen, shape)

    def close(self):
        try:
            self._file_handle.close()
        except:
            pass

class CH5MappedFile(CH5File):
    """Combine a CellH5 file with meta information from a mapping file, which contains information for each well"""
    def read_mapping(self, mapping_file, sites=None, rows=None, cols=None, locations=None, plate_name=''):
        self.mapping_file = mapping_file
        self.mapping = pandas.read_csv(self.mapping_file, sep='\t', dtype={'Well': str})
        self.mapping['Plate'] = plate_name

        if sites is not None:
            self.mapping = self.mapping[self.mapping['Site'].isin(sites)]
        if rows is not None:
            self.mapping = self.mapping[self.mapping['Row'].isin(rows)]
        if cols is not None:
            self.mapping = self.mapping[self.mapping['Column'].isin(cols)]

        if locations is not None:
            self.mapping = self.mapping[reduce(pandas.Series.__or__,
                                               [(self.mapping['Row'] == c) & \
                                                    (self.mapping['Column'] == r)
                                                for c, r in locations])]

        self.mapping.reset_index(inplace=True)
        
    def check_mapping(self, remove=False):
        self.mapping["CellH5"] = self.mapping.apply(lambda x: self.has_position(x["Well"], x["Site"]), axis=1)
        if remove:
            self.mapping = self.mapping[self.mapping["CellH5"]]
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
    
class CH5MappedFileCollection(object):
    """Several CellH5 files together with a mapping"""
    def __init__(self, name="CH5MappedFileCollection", mapping_files=None, cellh5_files=None,
                       sites=None, rows=None, cols=None, locations=None, init=True):
        self.name = name
        self.mapping_files = mapping_files
        self.cellh5_files = cellh5_files
        self.time_lapse = {}
        self.cellh5_handles = {}
        
        self.mapping = None
        if init:
            mappings = []
            for plate_name, mapping_file in mapping_files.items():
                if plate_name not in cellh5_files.keys():
                    raise RuntimeError("Plate name %s not found" % plate_name)
                cellh5_file = cellh5_files[plate_name] 
                
                mapped_ch5 = CH5MappedFile(cellh5_file)
                mapped_ch5.read_mapping(mapping_file, sites=sites, rows=rows, cols=cols, locations=locations, plate_name=plate_name)

                time_lapse = mapped_ch5.current_pos.get_time_lapse()
                if time_lapse is not None:
                    self.time_lapse[plate_name] = time_lapse / 60.0
                    log.info("Found time lapse of plate '%s' = %5.3f min" % (plate_name, self.time_lapse[plate_name]))
                else:
                    self.time_lapse[plate_name] = 0
                    
                self.cellh5_handles[plate_name] = mapped_ch5
                mappings.append(mapped_ch5.mapping)
                
            self.mapping = pandas.concat(mappings, ignore_index=True)
            del mappings
            
    def close(self):
        if self.cellh5_handles is not None:
            for v in self.cellh5_handles.values():
                v.close()
            
    def get_treatment(self, plate, well, site):
        return list(self.mapping[
                                (self.mapping['Plate'] == plate) & 
                                (self.mapping['Well'] == well) & 
                                (self.mapping['Site'] == site)
                                ][['siRNA ID', 'Gene Symbol']].iloc[0])
        
    def get_ch5_position(self, plate, well, site):
        return self.cellh5_handles[plate].get_position(well, site)
    
    def get_object_classificaiton_dict(self, prop="name", object_='primary__primary'):
        res = OrderedDict()
        data = self.cellh5_handles.values()[0].class_definition(object_)
        for i in range(len(data)):
            res[i] = data[prop][i]
        return res
        
####################
### CH5 Analysis ###
####################        

class CH5Analysis(CH5MappedFileCollection):
    """Basic class used for common analysis task using CellH5, e. g. already contains PCA, etc"""
    def __init__(self, name="CH5Analysis", mapping_files=None, cellh5_files=None,
                       sites=None, rows=None, cols=None, locations=None, output_dir=None, init=True):
        CH5MappedFileCollection.__init__(self, name=name, mapping_files=mapping_files, cellh5_files=cellh5_files,
                       sites=sites, rows=rows, cols=cols, locations=locations, init=True)
        
        self.output_dir = output_dir
        self.set_output_dir(output_dir)
        
        self._init_loger()
        
    def _init_loger(self):
        log = logging.getLogger(str(self.__class__.__name__))
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        log.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        log.addHandler(ch)
        self.log = log
        
    def set_output_dir(self, output_dir):
        if self.output_dir is None:
            debug = ""
            try:
                import pydevd  # @UnresolvedImport
                debug = "d_"
            except:
                pass
            self.output_dir = self.name + "/" + debug + datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
            try:
                os.makedirs(self.output_dir)
            except:
                pass
        log.info("Output Directory: " + self.output_dir)
        
    def output(self, file_):
        file_ = self._str_sanatize(file_)
        return os.path.join(self.output_dir, file_) 
        
    @staticmethod    
    def _str_sanatize(input_str):
        input_str = input_str.replace(" ", "_")
        input_str = input_str.replace("/", "_")
        input_str = input_str.replace("#", "_")
        input_str = input_str.replace(")", "_")
        input_str = input_str.replace("(", "_")
        return input_str
    
    def get_treatment(self, plate, w, p):
        return list(self.ch5map.mapping[
                                (self.mapping['Plate'] == plate) & 
                                (self.mapping['Well'] == w) & 
                                (self.mapping['Site'] == p)
                                ][['siRNA ID', 'Gene Symbol']].iloc[0])
    
    def cluster_run(self, cluster_class, cluster_on=('neg', 'target', 'pos'), feature_set="PCA", max_samples=1000, data=None, **clusterargs):
        if data is None:
            data = self.get_data(cluster_on, feature_set)
            
        self.cluster_class_on_all = cluster_class(**clusterargs)
        if data.shape[0] > max_samples:
            idx = range(data.shape[0])
            numpy.random.seed(43)
            numpy.random.shuffle(idx)
            idx = idx[:max_samples]
            data = data[idx, :]
        
        self.cluster_class_on_all.fit(data)
        
        def _cluster_(xxx):
            if xxx["Object count"] > 0:
                data = xxx[feature_set]
                cluster = self.cluster_class_on_all.predict(data)
                return cluster 
            else:
                return []
        
        cluster = pandas_apply(self.mapping, _cluster_)
        self.mapping['Simple clustering'] = pandas.Series(cluster)
        
    def pca_run(self, pca_dims=None, train_on=('neg', 'target', 'pos'), pca_cls=None, **pca_args):
        training_matrix = self.get_data(train_on)
        if pca_cls is None:
            from sklearn.decomposition import PCA
            pca_cls = PCA
        
        log.info('Compute PCA (%s): on matrix shape %r' % (str(pca_cls), training_matrix.shape))
        if pca_dims is None:
            pca_dims = 0.99
        self.pca = pca_cls(pca_dims, pca_args)
        self.pca.fit(training_matrix)
        log.info('Compute PCA (%s): %d dimensions used' % (str(pca_cls), self.pca.n_components_))
        
        def _project_on_pca_(xxx):
            return self.pca.transform(xxx)
        res = pandas.Series(self.mapping['Object features'][self.mapping['Object count'] > 0].map(_project_on_pca_))
        self.mapping['PCA'] = res
         
    def read_feature(self, object_="primary__primary", time_frames=None, remove_feature=(), 
                           read_classification=True, idx_selector_functor=None):

        # TODO
        all_features = self.cellh5_handles.values()[0].object_feature_def()
        features_keep = [na for na in range(len(all_features)) if na not in remove_feature]
        self.all_features = [all_features[na] for na in range(len(all_features)) if na not in remove_feature]
        
        features = []
        classification = []
        counts = []
        c5_object_index = []
        c5_object_index2 = []
        c5_object_index_not = []
        
        for i, row in self.mapping.iterrows():
            plate = row['Plate']
            well = row['Well']
            site = str(row['Site'])
            treatment = "%s %s" % (row['Gene Symbol'], row['siRNA ID'])
            
            ch5_pos = self.get_ch5_position(plate, well, site)            
            all_times = ch5_pos.get_all_time_idx(object_)
            self.log.info('Reading %s %s %s %s for object %s using time %r' % (plate,well,site,treatment,object_,time_frames))

            # there are data points
            if len(all_times) > 0:
                if time_frames is not None:
                    time_idx = numpy.in1d(all_times, time_frames)
                else:
                    time_idx = numpy.ones(len(all_times), dtype=numpy.bool)
                idx_bool = time_idx
                
                # TODO
                if idx_selector_functor is not None:
                    idx_bool = idx_selector_functor(ch5_pos, plate, treatment, self.output_dir)
                
                idx = numpy.nonzero(idx_bool)[0]
                if len(idx) > 0:
                    feature_matrix = ch5_pos.get_object_features(object_=object_, index=tuple(idx))
                    feature_matrix = feature_matrix[:, features_keep]
                    
                    classification_labels = ch5_pos.get_class_prediction(object_=object_)[idx_bool]['label_idx']              
                    object_count = len(feature_matrix)
                else:
                    object_count = 0
            else:
                object_count = 0
                
            counts.append(object_count)
            if object_count > 0:
                features.append(feature_matrix)
                classification.append(classification_labels)
            else:
                features.append(numpy.zeros((0,)))
                classification.append(numpy.zeros((0,)))
            c5_object_index.append(idx_bool)
            c5_object_index2.append(idx)
            c5_object_index_not.append(numpy.logical_not(idx_bool))
            
        self.mapping['Object features'] = features
        self.mapping['Object count'] = counts
        self.mapping['CellH5 object index'] = c5_object_index
        self.mapping['CellH5 object index 2'] = c5_object_index2
        self.mapping['CellH5 object index excluded'] = c5_object_index_not
        self.mapping['Object classification label'] = classification

        self.check_standardaize_features()
        
    def check_standardaize_features(self):
        all_data = self.get_data(('neg', 'target', 'pos'))
        
        nans = numpy.isnan(all_data)
        if nans.any():
            print "Axes 1 features"
            print numpy.nonzero(nans.any(1))
            print "Axes 0 samples"
            print numpy.nonzero(nans.any(0))
            raise RuntimeError("NaNs in data")
        
        self.norm_mean = all_data.mean(0)
        self.norm_stds = all_data.std(0)
        
        if (self.norm_stds < 10e-9).any():
            raise RuntimeError("stds get low")
        
        for matrix_i in  self.mapping[self.mapping["Object count"] > 0]["Object features"]:
            matrix_i -= self.norm_mean
            matrix_i /= self.norm_stds    
            
    def get_column_as_matrix(self, column_name, get_index=False):
        sel = self.mapping["Object count"] > 0
        data = self.mapping[sel][column_name]
        res = numpy.concatenate(list(data))
        if get_index: 
            lens = data.map(len)
            index = [[i]*l for i, l in zip(data.index, lens)]
            return res, numpy.array(list(chain.from_iterable(index)))
        return res
    
    def get_data_sampled(self, in_group, in_classes, n_sample=10000, in_class_type = 'Object classification label', type_='Object features'):
        # Row selection
        assert sum(in_classes.values()) - 1 < 0.000001
        object_count_sel = self.mapping['Object count'] > 0
        in_group_sel = self.mapping['Group'].isin(in_group)
        
        selection = numpy.logical_and(object_count_sel, in_group_sel)
        selected = self.mapping[selection].reset_index()
        res = numpy.concatenate(list(selected[type_]))
        
        # Single cell selection
        
        indicies = []
        
        class_labels = numpy.concatenate(selected[in_class_type])
        for in_class, in_class_ratio in in_classes.items():
            i_samples = int(in_class_ratio * n_sample)
            samples = numpy.nonzero(numpy.in1d(class_labels, in_class))[0]
            numpy.random.shuffle(samples)
            if i_samples > len(samples):
                raise RuntimeError("Not enough classes '%r' for sampled selection (want %d, available %d)" % (in_class, i_samples, len(samples)))
            indicies.append(samples[:i_samples])
            
        indicies = numpy.concatenate(indicies)
            
        return res[indicies, :]
    
    def get_data(self, in_group, type_='Object features', in_classes=None, in_class_type='Object classification label'):
        # Row selection
        object_count_sel = self.mapping['Object count'] > 0
        in_group_sel = self.mapping['Group'].isin(in_group)
        
        selection = numpy.logical_and(object_count_sel, in_group_sel)
        selected = self.mapping[selection].reset_index()
        res = numpy.concatenate(list(selected[type_]))
        
        # Single cell seleciton
        
        if in_classes is not None:
            class_labels = numpy.concatenate(selected[in_class_type])
            single_selection = numpy.in1d(class_labels, in_classes)
            res = res[single_selection, :]

        log.info('get_data for %r positions' % len(selected))
        log.info('get_data for treatment %r with training matrix shape %r' % (list(selected['siRNA ID'].unique()), res.shape))

        return res

class CH5FateAnalysis(CH5Analysis):
    """
    Implementation of Hidden Markov Model Smoothing of class transitions in detected events
    This class is used to pre-process temporal data to foster more precise mitotic timings,
    fate categories (e.g. death in mitosis), and correlations to kinetics in a additional 
    channel / color
    """
    def read_events(self, onset_frame=0, time_limits=(0, numpy.Inf)):    
        event_ids_all = []
        event_labels_all = []
        for _, (plate, well, site) in self.mapping[['Plate', 'Well', 'Site']].iterrows(): 
            self.log.info('Reading %s %s %s for object' % (plate, well, site))
            ch5_pos = self.get_ch5_position(plate, well, site)    

            event_ids = ch5_pos.get_events()            
            event_ids = [e for e in event_ids if ch5_pos.get_time_idx(time_limits[0] <= e[onset_frame]) <= time_limits[1]]
            event_ids_all.append(event_ids)
            
            event_labels = [ch5_pos.get_class_label(e) for e in event_ids]
            event_labels_all.append(event_labels)
            
        self.mapping["Event IDs"] = event_ids_all
        self.mapping["Event labels"] = event_labels_all
        
    
    def __str__(self):
        str_ =  "\nFound events"
        if "Event IDs" in self.mapping:
            str_+= " True\n"
        else:
            str_+= " False\n"    
        str_+= "Found tracks (full trajectories from events)"
        if "Track IDs" in self.mapping:
            str_+= " True\n"
        else:
            str_+= " False\n"
        str_+= "Class description\n"
        str_+= "index\tlabel\tname\n"
        class_dict = self.get_object_classificaiton_dict()
        class_dict_label = self.get_object_classificaiton_dict("label")
        for class_i in class_dict:
            str_+= "%d\t%d\t%s\n" % (class_i, class_dict_label[class_i], class_dict[class_i])
        str_+= "Caution: Class index will be used for all processing\n\n"
        return str_
                      
    def track_events(self):        
        def _track_events_(xxx):
            plate = xxx["Plate"]
            well = xxx["Well"]
            site = xxx["Site"]
            ch5_pos = self.get_ch5_position(plate, well, site)
            self.log.info('Tracking events  %s %s %s' % (plate, well, site))
            
            event_ids = xxx["Event IDs"]
            track_ids = []
            track_labels = []
            for _, event_idx in enumerate(event_ids):   
                start_idx = event_idx[-1]
                track = list(event_idx) + ch5_pos.track_first(start_idx)
                class_labels = ch5_pos.get_class_label_index(track)
                track_labels.append(class_labels)
                track_ids.append(track)
                
            return track_ids, track_labels

        track_ids, track_labels = pandas_apply(self.mapping, _track_events_)
        
        self.mapping["Track IDs"] = pandas.Series(track_ids)
        self.mapping["Track Labels"] = pandas.Series(track_labels)
                                    
    def setup_hmm(self, transmat, constraint_xml, eps=0.001):
        k_classes = transmat.shape[0]
        constraints = HMMConstraint(constraint_xml)
        transmat = normalize(transmat, axis=1, eps=eps)
        est = HMMAgnosticEstimator(k_classes, transmat, numpy.ones((k_classes, k_classes)), numpy.ones((k_classes, )) )
        est.constrain(constraints)
        self.hmm = hmm.MultinomialHMM(n_components=est.nstates, transmat=transmat, startprob=est.startprob, init_params="")
        self.hmm._set_emissionprob(est.emis) 
        
    def predict_hmm(self):
        def _hmm_predict_(xxx):
            plate = xxx["Plate"]
            well = xxx["Well"]
            site = xxx["Site"]
            self.log.info('HMM prediction %s %s %s' % (plate, well, site))
            
            track_ids = xxx["Track IDs"]
            track_labels = xxx["Track Labels"]
            
            hmm_labels_list = []

            
            for _, track_labels in enumerate(track_labels):               
            
                class_labels = track_labels
                if not isinstance(track_labels, (list,)):
                    class_labels = list(track_labels)  
                hmm_class_labels = self.hmm.predict(class_labels)
                hmm_labels_list.append(hmm_class_labels)

            return hmm_labels_list, track_ids

        hmm_track_labels, hmm_track_ids = pandas_apply(self.mapping, _hmm_predict_)
        
        self.mapping["HMM Track IDs"] = pandas.Series(hmm_track_ids)
        self.mapping["HMM Track Labels"] = pandas.Series(hmm_track_labels)
        
    def print_tracks(self, track_name="HMM Track Labels", pattern="%d", ident=" "):
        if track_name not in self.mapping:
            RuntimeError("Tracks have not predicted yet. Use setup_hmm() and predict_hmm() first!")
        for i, xxx in self.mapping.iterrows():
            plate = xxx["Plate"]
            well = xxx["Well"]
            site = xxx["Site"]
            track_labels = xxx[track_name]
            print "Plate '%s'\tWell '%s'\tSite '%s'" % (plate, well, site)
            print "*"*50
            for track_id, track in enumerate(track_labels):
                print ident, "%03d:" % track_id, "".join(map(lambda xxx: pattern % xxx, track))
                
    def report_to_csv(self, output_filename="report.txt", export_images=True):
        import csv
        header = ['Plate', 'Well', 'Site', 'Type', 'Event_id', 'Length', 'Frame_first_appearance', 'Class_first_appearance', 
                  'Frames', 'Raw_classification', 'Hmm_classification']
        
        with open(output_filename, 'wb') as fh:
            writer = csv.DictWriter(fh, header, delimiter="\t", lineterminator='\n')
            writer.writeheader()
            for i, xxx in self.mapping.iterrows():
                plate = xxx["Plate"]
                well = xxx["Well"]
                site = xxx["Site"]
                ch5_pos = self.get_ch5_position(plate, well, site)
                
                index_data = xxx["Track IDs"]
                raw_data = xxx["Track Labels"]
                hmm_data = xxx["HMM Track Labels"]
                
                assert len(index_data) == len(raw_data) == len(hmm_data)
                
                line_dict = {}
                for e_id, (index, raw_class, hmm_class) in enumerate(zip(index_data, raw_data, hmm_data)):
                    frames = ch5_pos.get_time_idx2(index)
                    assert len(frames) == len(hmm_class) == len(raw_class)
                    back_track =  ch5_pos.track_backwards(index[0])
                    if len(back_track) > 0:
                        start_index = back_track[0]
                    else:
                        start_index = index[0]
                        
                    start_frame = ch5_pos.get_time_idx(start_index)
                    start_class = ch5_pos.get_class_name(start_index)
                    
                    line_dict['Event_id'] = e_id
                    line_dict['Plate'] = plate
                    line_dict['Well'] = well
                    line_dict['Site'] = site
                    line_dict['Type'] = "event"
                    line_dict['Length'] = len(index)
                    line_dict['Frame_first_appearance'] = start_frame
                    line_dict['Class_first_appearance'] = start_class
                    line_dict['Frames'] =  " ".join(map(str,frames))
                    line_dict['Raw_classification'] = " ".join(map(str,raw_class))
                    line_dict['Hmm_classification'] = " ".join(map(str,hmm_class))
                    
                    writer.writerow(line_dict)
                    
                    if export_images:
                        import vigra
                        img = self.get_track_image(plate, well, site, e_id)
                        vigra.impex.writeImage(img.swapaxes(1,0), "%s_%s_%s_%d.png" % (plate, well, site, e_id))
                    
    def get_track_image(self, plate, well, site, event_id):
        row = self.mapping[(self.mapping["Plate"] == plate) & (self.mapping["Well"] == well) & (self.mapping["Site"] == site)]
        
        ids_1 = row["Track IDs"][0]
        hmm_labels = row["HMM Track Labels"][0]
        
        ch5_pos = self.get_ch5_position(plate, well, site)
        
        img_1 = numpy.concatenate([ch5_pos.get_gallery_image_with_class(e_id) for e_id in ids_1[event_id]], 1)
        
        hmm_colors = [str(ch5_pos.definitions.class_definition()["color"][k]) for k in hmm_labels[event_id]]
        img_2 = numpy.concatenate([ch5_pos.get_gallery_image_with_class(e_id, color=hmm_colors[k]) for k, e_id in enumerate(ids_1[event_id])], 1)
        
        return numpy.concatenate((img_1, img_2))
             

##################
### Unit tests ###
##################


class CH5TestBase(unittest.TestCase):
    """Unit test base class"""
    def setUp(self):
        data_filename = '../data/0038.ch5'
        if not os.path.exists(data_filename):
            raise IOError(("No CellH5 test data found in 'cellh5/data'."
                           " Please refer to the instructions in "
                           "'cellh5/data/README'"))
        self.fh = CH5File(data_filename, 'a')
        self.well_str = '0'
        self.pos_str = self.fh.positions[self.well_str][0]
        self.pos = self.fh.get_position(self.well_str, self.pos_str)

    def tearDown(self):
        self.fh.close()

class TestCH5Basic(CH5TestBase):
    """Basic unit tests"""
    def testTimeLapse(self):
        time_lapse = self.pos.get_time_lapse()
        assert int(time_lapse) == 276

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
        w = numpy.array(map(len, tracks)).max() * GALLERY_SIZE
        img = numpy.zeros((GALLERY_SIZE * len(tracks), w), dtype=numpy.uint8)

        for k, t in enumerate(tracks):
            a = self.pos.get_gallery_image(tuple(t))
            img[k * GALLERY_SIZE:(k + 1) * GALLERY_SIZE, 0:a.shape[1]] = a

    def testGallery4(self):
        event = self.pos.get_events()[42]
        a1 = self.pos.get_gallery_image(tuple(event))

    def testClassNames(self):
        for x in ['inter', 'pro', 'earlyana']:
            self.assertTrue(x in self.pos.class_name_def((1, 2, 5)))

    def testClassColors(self):
        for x in ['#FF8000', '#D28DCE', '#FF0000']:
            self.assertTrue(x in self.pos.class_color_def((3, 4, 8)))

    def testClassColors2(self):
        self.pos.get_class_color((1, 221, 3233, 44244))
        self.pos.get_class_name((1, 221, 3233, 44244))

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
        
class TestCH5Write(CH5TestBase):
    """Write unit tests"""
    def testSimpleWrite(self):
        data = numpy.random.rand(10, 20)
        self.pos.set_object_feature_data('_test', data)
        data_ = self.pos.get_object_feature_by_name('_test')
        self.pos.del_object_feature_data('_test')
        assert (data == data_).all()

class TestCH5Examples(CH5TestBase):
    """Example application used as unit tests"""
    def testBackwardTracking(self):
        events = self.pos.get_events()
        ev = events[1]
        ev2 = self.pos.track_backwards(ev[-1])[-len(ev)+1:] + [ev[-1]] 
        for a,b in zip(ev,ev2):
            self.assertTrue(a==b)
        
    def testGalleryMatrix(self):
        image = self.pos.get_gallery_image_matrix(range(20), (5, 6))
        import vigra
        vigra.impex.writeImage(image.swapaxes(1, 0), 'img_matrix.png')

    def testReadAnImage(self):
        """Read an raw image an write a sub image to disk"""
        # read the images at time point 1
        h2b = self.pos.get_image(0, 0)
        tub = self.pos.get_image(0, 1)

        # Print part of the images prepare image plot
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(h2b[400:600, 400:600], cmap='gray')
        fig.savefig('img1.png', format='png')
        ax.imshow(tub[400:600, 400:600], cmap='gray')

    # unittest.skip('ploting so many lines is very slow in matplotlib')
    def testPrintTrackingTrace(self):
        """Show the cell movement over time by showing the trace of each cell
        colorcoded overlayed on of the first image
        """
        h2b = self.pos.get_image(0, 0)

        tracking = self.pos.get_object_table('tracking')
        nucleus = self.pos.get_object_table('primary__primary')
        center = self.pos.get_feature_table('primary__primary', 'center')

        # prepare image plot
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(h2b, cmap='gray')

        # on top of the image a white circle is plotted for each
        # center of nucleus.
        I = numpy.nonzero(nucleus[tracking['obj_idx1']]['time_idx'] == 0)[0]
        for x, y in center[I]:
            ax.plot(x, y, 'w.', markersize=7.0, scaley=False, scalex=False)

        ax.axis([0, h2b.shape[1], h2b.shape[0], 0])

        # a line is plotted between nucleus center of each pair of
        # connected nuclei. The color is the mitotic phase
        for idx1, idx2 in zip(tracking['obj_idx1'],
                              tracking['obj_idx2']):
            color = self.pos.get_class_color(idx1)
            (x0, y0), (x1, y1) = center[idx1], center[idx2]
            ax.plot([x0, x1], [y0, y1], color=color)

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

        # compute mitotic index by counting the number cell per
        # class label over all times
        mitotic_index = numpy.array(map(lambda x: [len(numpy.nonzero(x == class_idx)[0]) for class_idx in range(n_classes)],
            [predictions[nucleus['time_idx'] == time_idx]['label_idx'] for time_idx in range(time_max)]))

        # plot it
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for i in range(1, n_classes):
            ax.plot(mitotic_index[:, i], color=colors[i], label=names[i])

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

    
def run_single_test(cls, func):
    """Helper function to run a single test"""
    writing = unittest.TestSuite()
    writing.addTest(cls(func))
    unittest.TextTestRunner().run(writing)


if __name__ == '__main__':
#     run_single_test(TestCH5Write, 'testBackwardTracking')
    unittest.main()

