import numpy
import cellh5
import pandas
import os
import sys
import collections
import functools
import time
import re
import datetime
import pylab
from collections import OrderedDict
from itertools import izip_longest
from matplotlib.mlab import PCA as PCAold
from sklearn.decomposition import PCA, KernelPCA
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import hmm
import csv
from estimator import HMMConstraint, HMMAgnosticEstimator, normalize
hmm.normalize = lambda A, axis=None: normalize(A, axis, eps=10e-99)

import logging
log = logging.getLogger(__name__)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


def hex_to_rgb(hex_string):
    """
    Converts the hex representation of a RGB value (8bit per channel) to
    its integer components.

    Example: hex_to_rgb('#559988') = (85, 153, 136)
             hex_to_rgb('559988') = (85, 153, 136)

    @param hexString: the RGB value
    @type hexString: string
    @return: RGB integer components (tuple)
    """
    if hex_string[:2] == '0x':
        hex_value = eval(hex_string)
    elif hex_string[0] == '#':
        hex_value = eval('0x'+hex_string[1:])
    else:
        hex_value = eval('0x'+hex_string)
    b = hex_value & 0xff
    g = hex_value >> 8 & 0xff
    r = hex_value >> 16 & 0xff
    return (r/255.0, g/255.0, b/255.0)

                         

class CellH5Analysis(object):
    output_formats = ('png',)
    def __init__(self, name="", mapping_files=None, cellh5_files=None, output_dir=None, 
                       sites=None, rows=None, cols=None, locations=None, init=True):
        self.name = name
        self.mapping_files = mapping_files
        self.cellh5_files = cellh5_files
        self.output_dir = output_dir
        self.time_lapse = {}
        self.cellh5_handles = {}
        self.set_output_dir(output_dir)
        
        if init:
            # read mappings for all plates
            mappings = []
            for plate_name, mapping_file in mapping_files.items():
                assert plate_name in cellh5_files.keys()
                cellh5_file = cellh5_files[plate_name]
                plate_cellh5  = cellh5.CH5MappedFile(cellh5_file)
                plate_cellh5.read_mapping(mapping_file, sites=sites, rows=rows, cols=cols, locations=locations, plate_name=plate_name)
                plate_mappings = plate_cellh5.mapping
                time_lapse = plate_cellh5.current_pos.get_time_lapse()
                if time_lapse is not None:
                    self.time_lapse[plate_name] = time_lapse / 60.0
                    log.info("Found time lapse of plate '%s' = %5.3f min" % (plate_name, self.time_lapse[plate_name]))
    #             plate_cellh5.close()
                self.cellh5_handles[plate_name] = plate_cellh5
                
                
                mappings.append(plate_mappings)
            self.mapping = pandas.concat(mappings, ignore_index=True)
            del mappings
        
    def save(self, filename):
        import cPickle as pickle
        export_dict = {}
        for nn in ['name', 'mapping_files', 'cellh5_files', 'output_dir', 'time_lapse', 'pca_dims',]:
            export_dict[nn] = getattr(self, nn)
            
        pickle.dump((export_dict, self.mapping), open(filename, 'wb'))
        
    @staticmethod
    def load(filename):
        import cPickle as pickle
        (export_dic, mapping) = pickle.load(open(filename,'r'))
        ca = CellH5Analysis(init=False)
        for nn in ['name', 'mapping_files', 'cellh5_files', 'output_dir', 'time_lapse', 'pca_dims',]:
            setattr(ca, nn, export_dic[nn])
        ca.mapping = mapping
        return ca
            
        
    def close(self):
        for p, c in self.cellh5_handles.items():
            try:
                c.close()
            except:
                print 'Could not close ch5 file handle of plate', p
        
    def set_output_dir(self, output_dir):
        if self.output_dir is None:
            self.output_dir = self.name +"/" + datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
#             if self.gamma is not None:
#                 self.output_dir += "-o%s-p%d-k%s-n%f-g%f" % (self.feature_set, self.pca_dims, self.kernel, self.nu, self.gamma)
            try:
                os.makedirs(self.output_dir)
            except:
                pass
        log.info("Output Directory: " + self.output_dir)
        

    def output(self, file_):
        file_ = self.str_sanatize(file_)
        return os.path.join(self.output_dir, file_) 
        
    @staticmethod    
    def str_sanatize(input_str):
        input_str = input_str.replace(" ","_")
        input_str = input_str.replace("/","_")
        input_str = input_str.replace("#","_")
        input_str = input_str.replace(")","_")
        input_str = input_str.replace("(","_")
        return input_str
                
    def read_events(self, onset_frame=4, events_before_frame=99999):   
        self.mapping['Event labels'] = -1   
        self.mapping['Event ids'] = -1
        self.onset_frame = onset_frame
        event_label_list = []
        event_id_list = []   
        for _, (plate, w, p) in self.mapping[['Plate', 'Well','Site']].iterrows(): 
            try:
                cellh5file = cellh5.CH5File(self.cellh5_files[plate], 'r')
                cellh5pos = cellh5file.get_position(w, str(p))
            except:
                print "Positon", (w, p), "is corrupt. Process again with CellCognition"
                cellh5file.close()
                continue
            event_ids = cellh5pos.get_events()
            
            event_ids = [e for e in event_ids if cellh5pos.get_time_idx(e[onset_frame]) < events_before_frame]
            
            event_id_list.append(event_ids)
            event_label_list.append([cellh5pos.get_class_label(e) for e in event_ids])
            log.info("Read events from ch5: %r %r %r with %r" % (plate, w, p, list(self.get_treatment(plate, w, p))))
            cellh5file.close()
        self.mapping['Event labels'] = pandas.Series(event_label_list)
        self.mapping['Event ids'] = pandas.Series(event_id_list)
        
    def get_treatment(self, plate, w, p):
        return list(self.mapping[
                                (self.mapping['Plate'] == plate) &
                                (self.mapping['Well'] == w) &
                                (self.mapping['Site'] == p)
                                ][['siRNA ID', 'Gene Symbol']].iloc[0])
    def track_full_events(self):
        self.mapping['Event track labels'] = -1   
        self.mapping['Event track ids'] = -1 
        all_track_ids = []
        all_track_labels = []
        
        for _, (plate, w, p, event_ids) in self.mapping[['Plate', 'Well', 'Site', 'Event ids']].iterrows(): 
            try:
                cellh5file = cellh5.CH5File(self.cellh5_files[plate])
                cellh5pos = cellh5file.get_position(w, str(p))
            except:
                log.error("Positon %s %s is corrupt. Process again with CellCognition" % (w, str(p)))
                cellh5file.close()
                continue
            
            track_labels_list = []
            track_id_list = []
            
            if not isinstance(event_ids, (list,)):
                log.warning("Tracking events from ch5, no events found: %r %r %r %s with %r" % (plate, w, p, "with", list(self.get_treatment(plate, w, p))))
                all_track_ids.append([])
            else:
            
                for _, e_idx in enumerate(event_ids):
                    start_idx = e_idx[-1]
                    track_ids = list(e_idx) + cellh5pos.track_first(start_idx)
                    
                    track_labels_list.append(cellh5pos.get_class_label(track_ids))
                    track_id_list.append(track_ids)
                all_track_ids.append(track_id_list)
                all_track_labels.append(track_labels_list)
                log.info("Tracking events from ch5: %r %r %r %s with %r" % (plate, w, p, "with", list(self.get_treatment(plate, w, p))))
                cellh5file.close()
        self.mapping['Event track labels'] = pandas.Series(all_track_labels)
        self.mapping['Event track ids'] = pandas.Series(all_track_ids)
        
    def setup_hmm(self, hmm_n_classes, hmm_n_obs, hmm_constraint_file):
        self.hmm_n_classes = hmm_n_classes
        self.hmm_n_obs = hmm_n_obs
        self.hmm_constraint_file = hmm_constraint_file
        
        constraints = HMMConstraint(self.hmm_constraint_file)
        
        transmat = numpy.array([
                                [1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
                                [0.0, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                                [0.0, 0.1, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                                [0.0, 0.0, 0.1, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                                
                                [0.0, 0.0, 0.0, 0.0, 90 , 1  , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 60 ],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                                
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 90 , 1  , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 60 ],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.1],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 0.9, 0.0, 0.0, 0.0, 0.1],
                                
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 90 , 1  , 0.0, 0.0, 60 ],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.0, 0.1],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 0.9, 0.1],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 0.1],
                                
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                ])
        transmat = normalize(transmat, axis=1, eps=0 )
        
        assert transmat.shape[0] == self.hmm_n_classes
        assert transmat.shape[1] == self.hmm_n_classes
        
        est = HMMAgnosticEstimator(self.hmm_n_classes, 
                                   transmat, 
                                   numpy.ones((self.hmm_n_classes, self.hmm_n_obs)), 
                                   numpy.ones((self.hmm_n_classes, )))
        
        est.constrain(constraints)
        self.hmm = hmm.MultinomialHMM(n_components=est.nstates, transmat=transmat, startprob=est.startprob, init_params="")
        self.hmm._set_emissionprob(est.emis)
               
    def predict_hmm(self, class_selector='Event track labels'):
        log.info('Predict hmm')
        self.mapping['Event HMM track labels'] = -1 
        all_hmm_labels_list = []
        for _, (plate, w, p, track_labs) in self.mapping[['Plate', 'Well', 'Site', class_selector]].iterrows():
            log.info("  %r %r %r" % (plate, w, p)) 
            hmm_labels_list = []
            for t in track_labs:   
                hmm_class_labels = self.hmm.predict(numpy.array(t-1)) + 1
                hmm_labels_list.append(hmm_class_labels)
            all_hmm_labels_list.append(hmm_labels_list)
        self.mapping['Event HMM track labels'] = pandas.Series(all_hmm_labels_list)
    
    def pca(self, pca_dims=None, train_on=('neg', 'target', 'pos'), pca_type=PCA, whiten=False):
        training_matrix = self.get_data(train_on)
        log.info('Compute PCA (%s): on matrix shape %r' % (str(pca_type), training_matrix.shape))
        if pca_dims is None:
            pca_dims=0.99
        self.pca = pca_type(pca_dims, whiten=whiten)
        self.pca.fit(training_matrix)
        log.info('Compute PCA (%s): %d dimensions used' % (str(pca_type), self.pca.n_components_))
        
        def _project_on_pca_(xxx):
            return self.pca.transform(xxx)
        res = pandas.Series(self.mapping['Object features'][self.mapping['Object count'] > 0].map(_project_on_pca_))
        self.mapping['PCA'] = res
    
    def set_read_feature_time_predicate(self, cmp, value):
        self._rf_time_predicate_cmp = cmp
        self._rf_time_predicate_value = value
            
    def read_feature(self, idx_selector_functor=None, object_="primary__primary", remove_feature=(18, 62, 92, 122, 152)):
        # init new columns
        self.mapping['Object features'] = 0
        self.mapping['Object count'] = 0
        self.mapping['CellH5 object index'] = 0
        self.mapping['Object classification label'] = 0
        # read features from each plate
        selector_output_file = open(self.output('_read_feature_selection.txt'), 'wb')
        
        for plate_name, cellh5_file in self.cellh5_files.items(): 
            ch5_file = cellh5.CH5File(cellh5_file)
            
#             feat_names = map(str, ch5_file.feature_definition[object_]['object_features']['name'])
            features_keep = [na for na in range(239) if na not in remove_feature]
#             self.feature_names = [feat_names[na] for na in features_keep]
            
            features = []
            labels = []
            object_counts = []
            c5_object_index = []
            c5_object_index_not = []
            
            for i, row in self.mapping[self.mapping['Plate']==plate_name].iterrows():
                well = row['Well']
                site = str(row['Site'])
                treatment = "%s %s" % (row['Gene Symbol'], row['siRNA ID'])
                
                ch5_pos = ch5_file.get_position(well, site)
                
                feature_matrix = ch5_pos.get_object_features(object_=object_)
                classi_labels = ch5_pos.get_class_prediction(object_=object_)['label_idx']
                time_idx = ch5_pos['object']["primary__primary"]['time_idx']
                log.info('Reading %s %s %s %d %s for object %s using time %r %r' % (plate_name, 
                                                                     well, 
                                                                     site, 
                                                                     len(feature_matrix),
                                                                     " ".join(self.get_treatment(plate_name, well, int(site))),
                                                                     object_,
                                                                     self._rf_time_predicate_cmp.__name__, 
                                                                     self._rf_time_predicate_value))
                
                if len(time_idx) > 0:
                    if self._rf_time_predicate_cmp is not None:
                        time_idx_2 = self._rf_time_predicate_cmp(time_idx, self._rf_time_predicate_value)
                    else:
                        time_idx_2 = numpy.ones(len(time_idx), dtype=numpy.bool)
                        
                    idx = time_idx_2
                    if idx_selector_functor is not None:
                        idx = idx_selector_functor(ch5_pos, plate_name, treatment, self.output_dir)
                    
                    feature_matrix = feature_matrix[:, features_keep]
                    feature_matrix = feature_matrix[idx, :]
                    
                    label_vector = classi_labels[idx]
                    object_count = len(feature_matrix)
                    selector_output_file.write("%s\t%s\t%s\t%d\t%d\n" % (plate_name, well, treatment, idx.sum(), len(idx)))
                    
                else:
                    feature_matrix = []
                    label_vector = []
                    object_count = 0
                    idx =[]
                
                object_counts.append(object_count)
                
                if object_count > 0:
                    features.append(feature_matrix)
                    labels.append(label_vector)
                else:
                    features.append(numpy.zeros((0, )))
                    labels.append(numpy.zeros((0, )))
                c5_object_index.append(numpy.where(idx)[0])
                c5_object_index_not.append(numpy.where(numpy.logical_not(idx))[0])
                
                
            
            plate_idx = self.mapping['Plate'] == plate_name
            self.mapping.loc[plate_idx, 'Object features'] = features
            self.mapping.loc[plate_idx, 'Object count'] = object_counts
            self.mapping.loc[plate_idx, 'CellH5 object index'] = c5_object_index
            self.mapping.loc[plate_idx, 'CellH5 object index excluded'] = c5_object_index_not
            self.mapping.loc[plate_idx, 'Object classification label'] = labels
        selector_output_file.close()
        self.check_standardaize_features()
        
    def check_standardaize_features(self):
        all_data = self.get_data(('neg', 'target', 'pos'))
        
        nans = numpy.isnan(all_data)
        if nans.any():
            raise RuntimeError("NaNs in data")
        
        self.norm_mean = all_data.mean(0)
        self.norm_stds = all_data.std(0)
        
        if (self.norm_stds < 10e-9).any():
            raise RuntimeError("stds get low")
        
        for matrix_i in  self.mapping[self.mapping["Object count"] > 0]["Object features"]:
            matrix_i -= self.norm_mean
            matrix_i /= self.norm_stds    
            
            
    def get_column_as_matrix(self, column_name):
        sel = self.mapping["Object count"] > 0
        return numpy.concatenate(list(self.mapping[sel][column_name]))
    
    def get_data(self, in_group, type_='Object features', in_classes=None):
        # Row selection
        object_count_sel = self.mapping['Object count'] > 0
        in_group_sel = self.mapping['Group'].isin(in_group)
        
        selection = numpy.logical_and(object_count_sel, in_group_sel)
        selected = self.mapping[selection].reset_index()
        res = numpy.concatenate(list(selected[type_]))
        
        # Single cell seleciton
        
        if in_classes is not None:
            class_labels = numpy.concatenate(selected['Object classification label'])
            single_selection = numpy.in1d(class_labels, in_classes)
            res = res[single_selection, :]
            
        
        
        log.info('get_data for %r positions' % len(selected))
        log.info('get_data for treatment %r with training matrix shape %r' % (list(selected['siRNA ID'].unique()), res.shape))

        if self.fs is not None:
            res = res[:,self.fs]
        return res
                        
        
    def event_curves(self, event_selector, 
                           region_name,
                           feature_name,
                           cmap,
                           xlim,
                           ylim,
                           feature_backgorund
                           ):
        time_unit = 'min'
        try:
            os.makedirs(os.path.join(self.output_dir, 'curves'))
        except:
            pass
        fig = pylab.figure()
        for _, (plate, w, p) in self.mapping[['Plate', 'Well', 'Site']].iterrows(): 
            try:
                cellh5file = cellh5.CH5File(self.cellh5_files[plate])
                cellh5pos = cellh5file.get_position(w, str(p))
            except:
                print "File", (plate, w, p), "is corrupt. Process again with CellCognition"
                cellh5file.close()
                continue

            idx = ((self.mapping['Plate'] == plate) &
                   (self.mapping['Well'] == w) &
                   (self.mapping['Site'] == p))
            treatment = self.get_treatment(plate, w, p)
        
            pylab.clf()
            ax = pylab.subplot(111)          
            
            feature_table = cellh5pos.get_object_features(region_name)   
            feature_idx = cellh5file.get_object_feature_idx_by_name(region_name, feature_name)
            tracks = self.mapping.loc[idx][event_selector].iloc[0]
            track_ids = self.mapping.loc[idx]['Event track ids'].iloc[0]  
            
            all_feature_values = [feature_table[t, feature_idx] for t in track_ids]
                
            for line, feature_values in zip(tracks, all_feature_values):
                x_values = numpy.array(map(int,list(line)))
                if numpy.max(feature_values) < 15:
                    print 'excluding event due to low signal'
                    continue
                values = numpy.array(feature_values - feature_backgorund) 
                values = values / numpy.mean(values[(self.onset_frame-1):(self.onset_frame+2)])
                self._plot_curve(x_values[:len(feature_values)], values, cmap, ax)
  

            pylab.vlines(0, ylim[0], ylim[1], 'k', '--', label='NEBD')
            title = '%s - %s (%s)'% tuple(list(treatment) + [w,])
            ax.set_title(title)
            ax.set_xlabel('Time (%s)' % time_unit)
            ax.set_ylabel('Fluorescence (AU)')
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            
            for fmt in self.output_formats:
                pylab.savefig(self.output('curves_all_%s_%d_.%s' % (title, xlim[1], fmt)))    
            
            cellh5file.close()

    def _plot_curve(self, line, values, cmap, ax):
        old_l = line[0]
        old_l_idx = 0
        for l_idx, l1 in enumerate(line):
            if l1 != old_l:
                x = (numpy.arange(old_l_idx, l_idx+1) - self.onset_frame ) * self.time_lapse.values()[0]
                y = values[old_l_idx:(l_idx+1)]
                #print 'x', x[0], 'to', x[-1]
                #print 'y', y[0], 'to', y[-1]
                ax.plot(x, y, color=cmap(int(old_l)), linewidth=1)
                old_l = l1
                old_l_idx = l_idx
                
        x = (numpy.arange(old_l_idx, len(line)) - self.onset_frame) * self.time_lapse.values()[0]
        y = values[old_l_idx:(len(line))]
        ax.plot(x, y, color=cmap(int(l1)), linewidth=1)
              
    def plot_track_order_map(self, track_selection, color_maps, track_short_crop_in_min=240):
        try:
            os.makedirs(os.path.join(self.output_dir, 'tracks'))
        except:
            pass
        fig = pylab.figure()
        for _, (plate, w, p) in self.mapping[['Plate', 'Well', 'Site']].iterrows(): 
            try:
                cellh5file = cellh5.CH5File(self.cellh5_files[plate])
            except:
                print "File", (plate, w, p), "is corrupt. Process again with CellCognition"
                cellh5file.close()
                continue

            idx = ((self.mapping['Plate'] == plate) &
                   (self.mapping['Well'] == w) &
                   (self.mapping['Site'] == p))
            treatment = self.get_treatment(plate, w, p)
            for k, class_selector in enumerate(track_selection):
                cmap = color_maps[k]
                tracks = self.mapping.loc[idx][class_selector].iloc[0]
                track_lens = map(len, tracks)
                if len(track_lens) > 0:
                    max_track_length = max(track_lens)
                    n_tracks = len(tracks)
                    img = numpy.zeros((n_tracks, max_track_length), dtype=numpy.uint8)
                    
                    track_crop_frame = int(track_short_crop_in_min/self.time_lapse[plate] + 0.5)
                    
                    def my_cmp(x,y):
                        counter = lambda x, items: reduce(lambda a,b:a+b, [list(x).count(xx) for xx in items])
                        tmp =  cmp(counter(x, [2,3,4,5]), counter(y, [2,3,4,5]))
                        return tmp if tmp!=0 else cmp(len(x),len(y)) 
                    
                    for i, t in enumerate([tmp[0] for tmp in sorted(zip(tracks, self.mapping[idx][track_selection[-1]].iloc[0]), cmp=my_cmp, key=lambda x:x[1])]):
                        img[i,:len(t)] = t
                      
                    # plot short
                    pylab.clf()
                    ax = pylab.subplot(111)
                    ax.matshow(img[:,:track_crop_frame], cmap=cmap, vmin=0, vmax=cmap.N-1)
                    title = '%s %s (%s) %s' % tuple(list(treatment) + [w, class_selector]) 
                    ax.set_title(title)
                    pylab.axis('off')
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    for fmt in self.output_formats:
                        fig.savefig(self.output('tracks/short_%s_%d.%s' % (title, track_short_crop_in_min, fmt)), bbox_inches=extent)  

                    # plot long
                    pylab.clf()
                    ax = pylab.subplot(111)
                    ax.matshow(img, cmap=cmap, vmin=0, vmax=cmap.N-1) 
                    ax.set_title(title)
                    pylab.axis('off')
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    for fmt in self.output_formats:
                        fig.savefig(self.output('tracks/full_%s_%d.%s' % (title, int(max_track_length*self.time_lapse[plate]+0.5), fmt)), bbox_inches=extent)
                    

                else:
                    log.warning("%r %r %r 'Track has len zero. Nothing to plot'" % (plate, w, p)) 
                    
    def plot_mitotic_timing(self, track_selector):        
        pp = PdfPages(self.output('_mitotic_timing.pdf'))
        f = pylab.figure(figsize=(20,8))
        mito_timing = OrderedDict()
        mito_class = OrderedDict()
        
        xticklabels=[]
        
        for w, p in sorted(self.tracks.keys()):
            
            exp_content = OrderedDict()
        
            mito_classes = ['mito_int',
                            'mito_int_mito_int_mito',
                            'mito_int_mito_int_apo',
                            'mito_int_apo',
                            'mito_int_mito_apo',
                            'mito_apo',
                            'mito_unclassified']

            mito_timing[(w,p)] = []
            mito_class[(w,p)] = []
            if (w,p) not in self.tracks:
                continue
            
            for m_idx, m_class in enumerate(mito_classes):
                for t_ in self.tracks[(w,p)][m_class]:
                    t = t_[1]
                    track_str = "".join(map(lambda x: "%02d" % x, t))
                    mito_re = re.search(r'01+(?P<mito>(02|03)+)', track_str)
                    if mito_re is not None:
                        span = mito_re.span('mito')
                        mito_timing[(w,p)].append((span[1]/2 - span[0]/2) * self.time_lapse )
                        mito_class[(w,p)].append(m_idx)
                    else:
                        pass
                
            treatment = self.mcellh5.get_treatment(w, p)[1]
            xticklabels.append("%s_%02d" % (w,p))
        
        with open(self.output('__mito_timing.txt'), 'wb') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(xticklabels)
            writer.writerows(izip_longest(*mito_timing.values(), fillvalue=""))
               
        with open(self.output('__mito_classes.txt'), 'wb') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(xticklabels)
            writer.writerows(izip_longest(*mito_class.values(), fillvalue="")) 
        
        colors = ['g', 'g',] + ['r']*5 + ['b']*5 + ["#ffa500"]*3 + ['r']*5 + ['b']*5 + ["#ffa500"]*3 + ['k']*50
#         ax = p(mito_timing.values(), xticklabels=xticklabels, colors=colors, spread_type='g', spread=0.1)
        ax = pylab.boxplot(mito_timing.values())
        ax = pylab.gca()
        ax.set_xticklabels(xticklabels, rotation=90)
        ax.set_title('Mitotic Timing')
        ax.set_ylabel('Mitotic timing (min)')
        pylab.tight_layout()
        pylab.savefig(pp, format='pdf')
        
        pp.close()
    
    def purge_feature(self):
        del self.mapping['Object features']
    
def test_event_tracking():
    pm = CellH5Analysis('test_fate', 
                        {'002338': "M:/experiments/Experiments_002300/002338/002338/_meta/Cecog/Mapping/002338.txt"}, 
                        {'002338': "M:/experiments/Experiments_002300/002338/002338/_meta/Analysis/hdf5/_all_positions.ch5"}, 
                        sites=(1,),
                        rows=("B", "C" ), 
                        cols=(5,9),
                        )
    pm.read_events(5, 90)
    pm.track_full_events()
    pm.setup_hmm(17, 5, 'hmm_constraints/graph_5_to_17_ms_special.xml')
    pm.predict_hmm()
    
    import matplotlib
    cmap = matplotlib.colors.ListedColormap(map(lambda x: hex_to_rgb(x), 
                                                   ['#FFFFFF', 
                                                    '#00ff00', 
                                                    '#ff8000',
                                                    '#d28dce',
                                                    '#0055ff',
                                                    '#aa5500', 
                                                    '#ff0000',
                                                    '#ffff00']), 'classification_cmap')
    CMAP17 = matplotlib.colors.ListedColormap(map(lambda x: hex_to_rgb(x), 
                                                   ['#FFFFFF', 
                                                    '#AAAAAA', 
                                                    '#0000FF',
                                                    '#0000FF',
                                                    '#0000FF',
                                                    '#AAAAAA', 
                                                    '#0000FF',
                                                    '#0000FF',
                                                    '#0000FF',
                                                    '#AAAAAA', 
                                                    '#0000FF',
                                                    '#0000FF',
                                                    '#0000FF',
                                                    '#AAAAAA', 
                                                    '#0000FF',
                                                    '#0000FF',
                                                    '#0000FF',
                                                    '#FF0000',
                                                    '#00FF00']), 'cmap17')  
    pm.plot_track_order_map(['Event track labels', 'Event HMM track labels'], [cmap, CMAP17])
    pm.onset_frame = 5
#     pm.event_curves('Event HMM track labels',
#                     'tertiary__expanded',
#                     'n2_avg',
#                     cmap,
#                     (-20,240),
#                     (0,1.5),
#                     16)
    
#import vigra
from itertools import izip
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV

class CellH5EigenCell(CellH5Analysis):
    def compute_eigen_face_features(self, result_size=100, outer_size=140):
        import vigra
        cellh5.GALLERY_SIZE = outer_size
        rr = (outer_size-result_size) / 2
        rot_images = []
        for _, (plate, w, p) in self.mapping[['Plate', 'Well','Site']].iterrows(): 
            try:
                cellh5file = cellh5.CH5File(self.cellh5_files[plate])
                cellh5pos = cellh5file.get_position(w, str(p))
            except:
                print "Positon", (w, p), "is corrupt. Process again with CellCognition"
                cellh5file.close()
                continue
            
            obj_idx = xrange(len(cellh5pos.get_object_table('primary__primary')['obj_label_id']))
            orientation =  cellh5pos.get_orientation(obj_idx)[0]
            
            cnt = 0
            for cell_gal, orient in izip(cellh5pos.get_gallery_image_generator(obj_idx), orientation):
                rot_img = vigra.sampling.rotateImageRadiant( cell_gal.astype(numpy.float32), orient)[rr:-rr,rr:-rr].reshape(1, result_size**2)
                rot_images.append(rot_img)
                cnt+=1
#                 if cnt > 10:
#                     break
                
        rot_images = numpy.concatenate(rot_images)
        print rot_images.shape
        
        pca = PCA(n_components=300, whiten=True)
        data = pca.fit_transform(rot_images)
        
        #params = {'bandwidth': numpy.logspace(-1, 3, 10)}
        #grid = GridSearchCV(KernelDensity(), params)
        #grid.fit(data)
        
        #print "best bandwidth: {0}".format(grid.best_estimator_.bandwidth)
        
        # use the best estimator to compute the kernel density estimate
        #kde = grid.best_estimator_
        
        # sample 44 new points from the data
        #new_data = kde.sample(100, random_state=41)
        new_data = pca.inverse_transform(data).reshape((-1,result_size, result_size))
        
        orig_data = rot_images.reshape((-1,result_size, result_size))
        
        pylab.figure()
        for i in range(100):
            ax = pylab.subplot(10,10, i)
            ax.imshow(numpy.concatenate((new_data[i,:,:], orig_data[i,:,:])), interpolation='nearest', clim=(0, 196), cmap='gray' )
            pylab.axis('off')
        pylab.show()
        
    
    
    def normalize_cell_images(self):
        rot_images = []
        for _, (plate, w, p) in self.mapping[['Plate', 'Well','Site']].iterrows(): 
            try:
                cellh5file = cellh5.CH5File(self.cellh5_files[plate])
                cellh5pos = cellh5file.get_position(w, str(p))
            except:
                print "Positon", (w, p), "is corrupt. Process again with CellCognition"
                cellh5file.close()
                continue
            
            obj_idx = xrange(len(cellh5pos.get_object_table('primary__primary')['obj_label_id']))
            orientation =  cellh5pos.get_orientation(obj_idx)[0]
            
            cnt = 0
            for cell_gal, orient in izip(cellh5pos.get_gallery_image_generator(obj_idx), orientation):
                rot_img = vigra.sampling.rotateImageRadiant( cell_gal.astype(numpy.float32), orient)[30:-30,30:-30].reshape(1, 3600)
                rot_images.append(rot_img)
                cnt+=1
#                 if cnt > 10:
#                     break
                
        rot_images = numpy.concatenate(rot_images)
        print rot_images.shape 
            
        pca = PCA(n_components=1000, whiten=False)
        data = pca.fit_transform(rot_images)
        
        params = {'bandwidth': numpy.logspace(-1, 3, 10)}
        grid = GridSearchCV(KernelDensity(), params)
        grid.fit(data)
        
        print "best bandwidth: {0}".format(grid.best_estimator_.bandwidth)
        
        # use the best estimator to compute the kernel density estimate
        kde = grid.best_estimator_
        
        # sample 44 new points from the data
        new_data = kde.sample(100, random_state=41)
        new_data = pca.inverse_transform(new_data).reshape((-1,60,60))
        
        pylab.figure()
        for i in range(100):
            ax = pylab.subplot(10,10, i)
            ax.imshow(new_data[i,:,:], interpolation='nearest', clim=(0, 196), cmap='gray' )
            pylab.axis('off')
        pylab.show()

        
    
def test_features_pca(): 
    pm = CellH5Analysis('test_features_pca', 
                        {'SP_9': "F:/sara_adhesion_screen/sp9.txt"}, 
                        {'SP_9': "F:/sara_adhesion_screen/sp9__all_positions_with_data_combined.ch5"}, 
                        sites=(1,),
                        rows=("B","C" ), 
                        cols=(5,9),
                        )
    
    pm.set_read_feature_time_predicate(numpy.equal, 0)
    pm.read_feature()
    pm.train_pca()
    pm.predict_pca()
    
def test_eigen_cell():
    ec = CellH5EigenCell('test_eigen_cell', 
                        {'002324': 'M:/experiments/Experiments_002300/002324/meta/CellCog/mapping/MD9_Grape_over_Time.txt'}, 
                        {'002324': 'F:/matthias_screen/cecog/analysis/hdf5/_all_positions.ch5'},
                        )
    
    ec.set_read_feature_time_predicate(numpy.equal, 0)
    ec.compute_eigen_face_features()

    
    

if __name__ == '__main__':
    test_eigen_cell()
#     test_event_tracking()
    