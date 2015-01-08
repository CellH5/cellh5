import numpy
import faulthandler
faulthandler.enable()

import pylab as plt
import cellh5
import cPickle as pickle
from numpy import recfromcsv
import pandas
import time
import h5py
import matplotlib.pyplot as mpl
import os
import collections
import functools
import time
import cProfile
import re
import datetime
import os
import random
import pylab
import myhmm
from itertools import izip_longest
import csv 
from matplotlib.mlab import PCA
from scipy.stats import nanmean
from matplotlib.backends.backend_pdf import PdfPages

from itertools import cycle
#from cecog.util.color import rgb_to_hex
import matplotlib

from matplotlib.backends.backend_pdf import PdfPages

from sklearn import hmm
from estimator import HMMConstraint, HMMAgnosticEstimator, normalize
from collections import OrderedDict
hmm.normalize = lambda A, axis=None: normalize(A, axis, eps=10e-99)

from fate_utils import CMAP17, CMAP17SIMPLE, hex_to_rgb, split_str_into_len, CMAP17_MULTI, pairwise
                 
OUTPUT_FORMATS = ['png', 'pdf']
TRACK_IMG_CROP = 240 # min

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] *= 2
rcParams['font.sans-serif'] = ['Arial']
#matplotlib.rc('xtick', labelsize=14) 
#matplotlib.rc('ytick', labelsize=14) 
#matplotlib.rc('axes', labelsize=14) 
matplotlib.rc('legend',**{'fontsize':18})
rcParams['axes.linewidth'] = rcParams['axes.linewidth']*2
rcParams['legend.numpoints'] = 1
#rcParams['legend.markerscale'] = 0
rcParams['xtick.major.width'] = 2
rcParams['ytick.major.width'] = 2
rcParams['pdf.fonttype'] = 42
rcParams['xtick.major.pad']= 8
rcParams['ytick.major.pad']= 8

SECURIN_BACKGROUND = 16 


class CellFateAnalysis(object):
    def __init__(self, plate_id, ch5_file, mapping_file, 
                 
                 events_before_frame=9999, 
                 onset_frame=0, 
                 
                 output_dir=None,
                 
                 time_lapse=None,
                 
                 hmm_constraint_file=None,
                 hmm_n_classes=None,
                 hmm_n_obs=None,
                 
                 securin_region=None,
                 
                 sites=None, 
                 rows=None, 
                 cols=None, 
                 locations=None):
        
        self.events_before_frame = events_before_frame
        self.onset_frame = onset_frame
        
        self.hmm_constraint_file = hmm_constraint_file
        self.hmm_n_classes = hmm_n_classes
        self.hmm_n_obs = hmm_n_obs
        
        self.time_lapse = time_lapse
        self.plate_id = plate_id
        
        self.output_dir = output_dir
        if output_dir is None:
            self.output_dir = plate_id +"/" + datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
            try:
                os.makedirs(self.output_dir)
            except:
                pass
        print "Output Directory: ", self.output_dir
        
        assert os.path.exists(ch5_file)
        assert os.path.exists(mapping_file)
#         assert os.path.exists(hmm_constraint_file)
#         assert time_lapse is not None
        
        self.mcellh5 = cellh5.CH5MappedFile(ch5_file)
        
        self.mcellh5.read_mapping(mapping_file, sites=sites, rows=rows, cols=cols, locations=locations)
        
        self.class_colors = self.mcellh5.class_definition('primary__primary')['color']
        self.class_names = self.mcellh5.class_definition('primary__primary')['name']

        hex_col = list(self.class_colors) 
        
        rgb_col = map(lambda x: hex_to_rgb(x), ['#FFFFFF'] + hex_col)
        rgb_col_cycle3 = map(lambda x: hex_to_rgb(x), ['#FFFFFF'] + hex_col[:-1] + hex_col[:-1] + hex_col[:-1] +[hex_col[-1]])
        
        self.cmap = matplotlib.colors.ListedColormap(rgb_col, 'classification_cmap')
        self.cmap_cycle_3 = matplotlib.colors.ListedColormap(rgb_col_cycle3, 'classification_cycle_3_cmap')
        self.tracks = {}
        
        for _, (w, p) in self.mcellh5.mapping[['Well','Site']].iterrows(): 
            try:
                cellh5pos = self.mcellh5.get_position(w,str(int(p)))
            except:
                print "Positon", (w, p), "is corrupt. Process again with CellCognition"
                continue
            #preds = cellh5pos.get_class_prediction()
            self.tracks[(w, p)] = {}
            event_ids = cellh5pos.get_events()
            
            event_ids = [e for e in event_ids if cellh5pos.get_time_idx(e[onset_frame]) < events_before_frame]
            
            self.tracks[(w, p)]['ids'] = event_ids
            self.tracks[(w, p)]['labels'] = [cellh5pos.get_class_label(e) for e in event_ids]
            
            print "Read events from ch5:", w, p, "with", list(self.mcellh5.get_treatment_of_pos(w, p))
           
           
    def output(self, file):
        file = self.str_sanatize(file)
        return os.path.join(self.output_dir, file) 
        
    @staticmethod    
    def str_sanatize(input_str):
        input_str = input_str.replace("/","_")
        input_str = input_str.replace("#","_")
        input_str = input_str.replace(")","_")
        input_str = input_str.replace("(","_")
        
        return input_str
          
#     def smooth_and_simplify_tracks(self, in_selector, out_name):   
#         print 'Track and hmm predict',
#         for w, p in self.tracks:
#             print w, p
#             cell5pos = self.mcellh5.get_position(w, p)
#             class_labels = self.tracks[(w,p)][in_selector]
#             class_label_str = map(lambda x : "".join(map(str, x)), class_labels)
#             class_label_str = map(lambda x: x.replace('3','2'), class_label_str)
#             class_label_str = map(lambda x: x.replace('4','2'), class_label_str)
#             class_label_str = map(lambda x: x.replace('5','3'), class_label_str)
#             
#             #class_label_str = map(lambda x: x.replace('6','1'), class_label_str)
#             #class_label_str = map(lambda x: x.replace('7','1'), class_label_str)
#             
#             
#             
#             self.tracks[(w,p)][out_name] = class_label_str 
#             #print class_label_str
                
    def fate_tracking(self, out_name):
        print 'Tracking cells',
        for w, p in self.tracks:
            print w,
            cell5pos = self.mcellh5.get_position(w, p)
            
            class_labels_list = []
            id_list = []
            for k, e_idx in enumerate(self.tracks[(w,p)]['ids']):   
                start_idx = e_idx[-1]
                track = list(e_idx) + cell5pos.track_first(start_idx)
                class_labels = cell5pos.get_class_label(track)
                class_labels_list.append(class_labels)
                id_list.append(track)

            self.tracks[(w,p)][out_name] = class_labels_list
            self.tracks[(w,p)]['track_ids'] = id_list
        print ' ...done'
        
#     def extract_topro(self):
#         topro = []
#         for w, p in self.tracks:
#             feature_table = self.mcellh5.get_position(w,str(p)).get_object_features('secondary__expanded')
#             for t in self.tracks[(w,p)]['track_ids']:
#                 topro.extend(feature_table[t, 6])  
#         pylab.figure()
#         pylab.hist(topro, 256)
#         pylab.figure()
#         pylab.hist(numpy.log2(numpy.array(topro)), 256, log=True)
#         pylab.show()
#         
#     def predict_topro(self, thres):
#         for w, p in self.tracks:
#             topro = []
#             topro_2 = []
#             feature_table = self.mcellh5.get_position(w,str(p)).get_object_features('secondary__expanded')
#             feature_table = self.mcellh5.get_position(w,str(p)).get_object_features('secondary__expanded')
#             for t_ids, t in zip(self.tracks[(w,p)]['track_ids'], self.tracks[(w,p)]['class_label_str']):
#                 t_topro_pos = feature_table[t_ids, 6] > thres
#                 t_ = numpy.array(list(t))
#                 t_[t_topro_pos] = 3
#                 topro.append(t_)
#                 
#                 t__ = numpy.zeros((len(t),), dtype=numpy.uint8)
#                 t__[t_topro_pos] = 3
#                 
#                 topro_2.append(t__)
#             self.tracks[(w,p)]['topro_class_labels'] = topro
#             self.tracks[(w,p)]['topro_pos'] = topro_2
               
    def predict_hmm(self, class_selector, class_out_name):
        print 'Predict hmm',
        for w, p in self.tracks:
            print w, 
            cell5pos = self.mcellh5.get_position(w, p)
            
            hmm_labels_list = []
            for k, t in enumerate(self.tracks[(w,p)][class_selector]):   
                class_labels = t
                if not isinstance(t, (list,)):
                    class_labels = list(t)
                    class_labels = numpy.array(map(int, t))
                
                hmm_class_labels = self.hmm.predict(numpy.array(class_labels-1)) + 1
                hmm_labels_list.append(hmm_class_labels)

                
            #self.tracks[(w,p)]['class_labels'] = class_labels_list
            self.tracks[(w,p)][class_out_name] = hmm_labels_list
        print ' ... done'
                            
    def event_fate_curves(self, event_selector, 
                           title,
                           region_name,
                           feature_name,
                           with_fate,
                           cmap,
                           xlim,
                           ylim,
                           ):
        pp = PdfPages(self.output("%s.pdf") % title)
        
        time_unit = 'min'
        
        for w, p in sorted(self.tracks):
            cnt = 0
            f = pylab.figure(figsize=(8, 8))
            ax = pylab.gca()
            
            
            feature_table = self.mcellh5.get_position(w,str(p)).get_object_features(region_name)
            feature_idx = self.mcellh5.get_object_feature_idx_by_name(region_name, feature_name)

            id_selector = 'ids'
            fate_ = 'short'
            if with_fate:
                id_selector = 'track_ids'
                fate_ = 'long'
            
            feature_min = SECURIN_BACKGROUND 
            
            fate_classes = [(('mito_int', 'mito_int_mito_int_mito', 'mito_int_mito_int_apo', 'mito_int_mito_apo'), 'b', 'Mitosis - live interphase',),
                            (('mito_int_apo',),'g', 'Mitosis - death in interphase'),
                            (('mito_apo',), 'r', 'Mitosis - death in mitosis')]
            
            for fate_names, fate_color, fate_label in fate_classes:
                for fate_name in fate_names:
                    feature_values = []
                    for track_1, track_2, track_ids in self.tracks[(w,p)][fate_name]:
                        track_str = "".join(map(lambda x: "%02d" % x, track_2))
                        mito_re = re.search(r'01+(?P<mito>(02|03)+)', track_str)
                        if mito_re is not None:
                            end = mito_re.end('mito') / 2
                        else:
                            print 'oarg?'
                            print track_str
                            print ""
                            continue
                        
                        feature_values = numpy.array([feature_table[t, feature_idx] for t in track_ids[:end]])
                        values = feature_values - feature_min
                        values = values / numpy.mean(values[(self.onset_frame-1):(self.onset_frame+2)])
                        ax.plot(numpy.arange(-self.onset_frame, len(values)-self.onset_frame, 1)*self.time_lapse, values[:len(values)], color=fate_color, linewidth=1, label=fate_label)
  
#             handles, labels = ax.get_legend_handles_labels()
#             lg = pylab.legend(handles, labels, loc=3, ncol=1)
#             lg.draw_frame(False)
            pylab.vlines(0, ylim[0], ylim[1], 'k', '--', label='NEBD')
            title = '%s - %s (%s)'% tuple(list(self.mcellh5.get_treatment_of_pos(w, p)) + [w,])
            ax.set_title(title)
            ax.set_xlabel('Time (%s)' % time_unit)
            ax.set_ylabel('Fluorescence (AU)')
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            
            pylab.savefig(pp, format='pdf')
            for fmt in OUTPUT_FORMATS:
                pylab.savefig(self.output('securin_fate_all_%s_%s.%s' % (fate_, title, fmt)))
            pylab.clf()
            pylab.close(f)    
        pp.close()
        
    def event_mean_fate_curves(self, event_selector, 
                           title,
                           region_name,
                           feature_name,
                           with_fate,
                           cmap,
                           xlim,
                           ylim,
                           ):
        pp = PdfPages(self.output("%s.pdf") % title)
        
        time_unit = 'min'
        
        for w, p in sorted(self.tracks):
            cnt = 0
            f = pylab.figure(figsize=(8, 8))
            ax = pylab.gca()
            
            
            feature_table = self.mcellh5.get_position(w,str(p)).get_object_features(region_name)
            feature_idx = self.mcellh5.get_object_feature_idx_by_name(region_name, feature_name)

            id_selector = 'ids'
            fate_ = 'short'
            if with_fate:
                id_selector = 'track_ids'
                fate_ = 'long'
            
            feature_min = SECURIN_BACKGROUND 
            
            fate_classes = [(('mito_int', 'mito_int_mito_int_mito', 'mito_int_mito_int_apo', 'mito_int_mito_apo'), 'b', 'Mitosis - live interphase',),
                            (('mito_int_apo',),'g', 'Mitosis - death in interphase'),
                            (('mito_apo',), 'r', 'Mitosis - death in mitosis')]
            
            for fate_names, fate_color, fate_label in fate_classes:
                feature_values = []
                for fate_name in fate_names:
                    for track_1, track_2, track_ids in self.tracks[(w,p)][fate_name]:
                        track_str = "".join(map(lambda x: "%02d" % x, track_2))
                        mito_re = re.search(r'01+(?P<mito>(02|03)+)', track_str)
                        if mito_re is not None:
                            end = mito_re.end('mito') / 2
                        else:
                            print 'oarg?'
                            print track_str
                            print ""
                            continue
                        values = numpy.array([feature_table[t, feature_idx] for t in track_ids[:end]])
                        values -= feature_min
                        values /= numpy.mean(values[(self.onset_frame-1):(self.onset_frame+2)])
                        feature_values.append(values)
                y = []
                yerr = []
                for tmp in izip_longest(*feature_values):
                    y.append(numpy.array([t for t in tmp if t is not None]).mean())
                    yerr.append(numpy.array([t for t in tmp if t is not None]).std())
                    
                xa = numpy.arange(-self.onset_frame, len(y)-self.onset_frame, 1) * self.time_lapse
                
                step = 1
                fate_ = 'short'
                if with_fate:
                    step = 4
                    fate_ = 'long'
                pylab.errorbar(xa[::step], y[::step], yerr=yerr[::step], fmt='o-', color=fate_color, markeredgecolor=fate_color, label=fate_label+" (%d)"%len(feature_values))
  

            handles, labels = ax.get_legend_handles_labels()
            # remove the errorbars
            handles = [h[0] for h in handles]
            lg = pylab.legend(handles, labels, loc=3, ncol=1)
            lg.draw_frame(False)
            pylab.vlines(0, ylim[0], ylim[1], 'k', '--', label='NEBD')
            title = '%s - %s (%s)'% tuple(list(self.mcellh5.get_treatment_of_pos(w, p)) + [w,])
            ax.set_title(title)
            ax.set_xlabel('Time (%s)' % time_unit)
            ax.set_ylabel('Fluorescence (AU)')
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            
            pylab.savefig(pp, format='pdf')
            for fmt in OUTPUT_FORMATS:
                pylab.savefig(self.output('securin_fate_mean_%s_%s.%s' % (fate_, title, fmt)))
            pylab.clf()   
            pylab.close(f) 
        pp.close()
        
    def event_mean_curves(self, event_selector, 
                           title,
                           region_name,
                           feature_name,
                           with_fate,
                           cmap,
                           xlim,
                           ylim,
                           ):
        pp = PdfPages(self.output("%s.pdf") % title)
        
        
        time_unit = 'min'
        
        for w, p in sorted(self.tracks):
            cnt = 0
            f = pylab.figure(figsize=(12, 8))
            
            
            
            feature_table = self.mcellh5.get_position(w,str(p)).get_object_features(region_name)
            feature_idx = self.mcellh5.get_object_feature_idx_by_name(region_name, feature_name)

            id_selector = 'ids'
            if with_fate:
                id_selector = 'track_ids'
            
            all_feature_values = [feature_table[t, feature_idx] for t in self.tracks[(w,p)][id_selector]]
                
            feature_min = SECURIN_BACKGROUND
                
            lines = {}
            lines['Mitotic exit'] = []
            lines['Dying in mitosis'] = []
            lines['Dying in interphase'] = []
            
            for line, feature_values in zip(self.tracks[(w,p)][event_selector], all_feature_values):
                x_values = numpy.array(map(int,list(line)))
                if numpy.max(feature_values) < 5:
                    print 'excluding event due to low signal'
                    continue
                values = numpy.array(feature_values - feature_min) 
                values = values / numpy.mean(values[(self.onset_frame-1):(self.onset_frame+2)])
                if 17 in x_values:
                    apo_ind = list(x_values).index(17)
                    if apo_ind > 0:
                        if x_values[apo_ind-1] in (2,3,4,6,7,8,10,11,12,14,15,16):
                            lines['Dying in mitosis'].append(values)
                        elif x_values[apo_ind-1] in (1,5,9,13):
                            lines['Dying in interphase'].append(values)
                        else:
                            print 'Should not happen...'
                    else:
                        print 'Apoptotic from start on... ignoring as unclassified'
                else:
                    lines['Mitotic exit'].append(values)

            from itertools import izip_longest
            
            for n, c in [('Mitotic exit', 'b'), ('Dying in mitosis', 'r'), ('Dying in interphase', 'g')]:
                y = []
                yerr = []
                for tmp in izip_longest(*lines[n]):
                    y.append(numpy.array([t for t in tmp if t is not None]).mean())
                    yerr.append(numpy.array([t for t in tmp if t is not None]).std())
                    
                xa = numpy.arange(-self.onset_frame, len(y)-self.onset_frame, 1) * self.time_lapse
                
                step = 1
                
                fate_ = 'short'
                if with_fate:
                    step = 4
                    fate_ = 'long'
                pylab.errorbar(xa[::step], y[::step], yerr=yerr[::step], fmt='o-', color=c, markeredgecolor=c, label=n+" (%d)"%len(lines[n]))

            ax = pylab.gca()
            # add lines
            handles, labels = ax.get_legend_handles_labels()
            # remove the errorbars
            handles = [h[0] for h in handles]
            lg = pylab.legend(handles, labels, loc=3, ncol=3)
            lg.draw_frame(False)
            pylab.vlines(0, ylim[0], ylim[1], 'k', '--', label='NEBD')
            title = '%s - %s (%s)'% tuple(list(self.mcellh5.get_treatment_of_pos(w, p)) + [w,])
            ax.set_title(title) 
            ax.set_xlabel('Time (%s)' % time_unit)
            ax.set_ylabel('Fluorescence (AU)')
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            
            f.savefig(pp, format='pdf')
            for fmt in OUTPUT_FORMATS:   
                f.savefig(self.output('securin_2_mean_%s_%s_.%s'% (fate_, title, fmt)))
            pylab.close(f)  
        pp.close()
        
    def event_curves(self, event_selector, 
                           title,
                           region_name,
                           feature_name,
                           with_fate,
                           cmap,
                           xlim,
                           ylim,
                           ):
        pp = PdfPages(self.output("%s.pdf") % title)
        
        
        time_unit = 'min'
        
        for w, p in sorted(self.tracks):
            cnt = 0
            f = pylab.figure(figsize=(8, 8))
            ax = pylab.gca()
            
            
            feature_table = self.mcellh5.get_position(w,str(p)).get_object_features(region_name)
            feature_idx = self.mcellh5.get_object_feature_idx_by_name(region_name, feature_name)

            id_selector = 'track_ids'
            fate_ = 'short'
            if with_fate:
                id_selector = 'track_ids'
                fate_ = 'long'
            
            all_feature_values = [feature_table[t, feature_idx] for t in self.tracks[(w,p)][id_selector]]
                
            feature_min = SECURIN_BACKGROUND #numpy.min(map(numpy.min, all_feature_values))
                
            #print "FEATURE_MIN", feature_min
            for line, feature_values in zip(self.tracks[(w,p)][event_selector], all_feature_values):
                x_values = numpy.array(map(int,list(line)))
                if numpy.max(feature_values) < 5:
                    print 'excluding event due to low signal'
                    continue
                values = numpy.array(feature_values - feature_min) 
                values = values / numpy.mean(values[(self.onset_frame-1):(self.onset_frame+2)])
                self._plot_curve(x_values[:len(feature_values)], values, cmap, ax)
  

            pylab.vlines(0, ylim[0], ylim[1], 'k', '--', label='NEBD')
            title = '%s - %s (%s)'% tuple(list(self.mcellh5.get_treatment_of_pos(w, p)) + [w,])
            ax.set_title(title)
            ax.set_xlabel('Time (%s)' % time_unit)
            ax.set_ylabel('Fluorescence (AU)')
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            
            pylab.savefig(pp, format='pdf')
            for fmt in OUTPUT_FORMATS:
                pylab.savefig(self.output('securin_2_all_%s_%s.%s' % (fate_, title, fmt)))
            pylab.clf()  
            pylab.close(f)  
        pp.close()
    
    def setup_hmm(self, k_classes, constraint_xml):
        constraints = HMMConstraint(constraint_xml)
        
        transmat = numpy.array([
                                [10 ,  0.1,  0.0,  0.0,  0.1],
                                [0.1,  10 ,  0.1,  0.0,  0.1],
                                [0.1,  0.0,  10 ,  0.1,  0.1],
                                [0.1,  0.0,  0.0,  10 ,  0.1],
                                [0.0,  0.0,  0.0,  0.0,  10 ],
                                ])
        transmat = normalize(transmat, axis=1, eps=0.001)
        
        est = HMMAgnosticEstimator(k_classes, transmat, numpy.ones((k_classes, k_classes)), numpy.ones((k_classes, )) )
        est.constrain(constraints)
        self.hmm = hmm.MultinomialHMM(n_components=est.nstates, transmat=transmat, startprob=est.startprob, init_params="")
        self.hmm._set_emissionprob(est.emis) 
             
    def plot_tracks(self, track_selection, cmaps, names, title='plot_tracks'):
        n = len(track_selection)
        pp = PdfPages(self.output('%s.pdf') % title)
        
        old_w = 'ZZZ'
        j = 0
        for w, p in sorted(self.tracks):
            if w[0] != old_w[0]:
                if j != 0:
                    j=0
                    pylab.savefig(pp, format='pdf')
                m = map(lambda x: x[0][0], self.tracks.keys()).count(w[0])
                fig = pylab.figure(figsize=(n*8,m*4)) 
                old_w = w
            
            cond = self.mcellh5.get_treatment_of_pos(w,p)
            for k, class_selector in enumerate(track_selection):
                cmap = cmaps[k]
                name = names[k]
                ax = pylab.subplot(m, n , k + j*n + 1)
                
                tracks = self.tracks[w,p][class_selector]
                track_lens = map(len, tracks)
                
                if len(track_lens) > 0:
                    
                    max_track_length = max(track_lens)
                
                    max_track_length = max(map(lambda x: len(x), tracks))
                    n_tracks = len(tracks)
                    img = numpy.zeros((n_tracks, max_track_length), dtype=numpy.uint8)
                    
                    track_crop_frame = int(TRACK_IMG_CROP/self.time_lapse)+1
                    
                    def my_cmp(x,y):
                        counter = lambda x, items: reduce(lambda a,b:a+b, [list(x).count(xx) for xx in items])
                        tmp =  cmp(counter(x, [2,3,4]), counter(y, [2,3,4]))
                        return tmp if tmp!=0 else cmp(len(x),len(y)) 
                    
                    for i, t in enumerate([tmp[0] for tmp in sorted(zip(tracks, self.tracks[w,p][track_selection[-1]]), cmp=my_cmp, key=lambda x:x[1])]):
                        if not isinstance(t, (list,)):
                            b = list(t)
                        img[i,:len(t)] = b
                      
                    
                      
                    ax.matshow(img[:,:track_crop_frame], cmap=cmap, vmin=0, vmax=cmap.N-1)
                    title = '%s - %s (%s) %s' % tuple(list(self.mcellh5.get_treatment_of_pos(w, p)) + [w, name]) 
                    ax.set_title(title)
                    pylab.axis('off')
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    fig.savefig(self.output('tracks_short_%s_%d.png' % (title, TRACK_IMG_CROP)), bbox_inches=extent)  

                    ax.matshow(img, cmap=cmap, vmin=0, vmax=cmap.N-1) 
                    ax.set_title(title)
                    pylab.axis('off')
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    fig.savefig(self.output('tracks_full_%s_%d.png' % (title, int(max_track_length*self.time_lapse))), bbox_inches=extent)
                else:
                    print w, p, 'Nothing to plot'
            j+=1
                    
        pylab.savefig(pp, format='pdf')    
        pp.close()   
                    
    def classify_tracks(self, class_selector):
        for w, p in self.tracks:
            cond = self.mcellh5.get_treatment_of_pos(w,p)
            self.tracks[(w,p)]['second_mitosis_inter'] = []
            self.tracks[(w,p)]['second_mitosis_apo'] = []
            self.tracks[(w,p)]['death_in_mitosis'] = []
            self.tracks[(w,p)]['death_in_interphase'] = []
            self.tracks[(w,p)]['no_second_mitosis_no_death'] = []
            self.tracks[(w,p)]['unclassified'] = []

            for t_idx, track in enumerate(self.tracks[w,p][class_selector]):
                track_str = "".join(map(str,track))
            
                dim = self._has_death_in_mitosis(track_str)
                dii = self._has_death_in_interphase(track_str)
                
                if dim is not None:
                    self.tracks[(w, p)]['death_in_mitosis'].append(dim)
                    
                elif dii is not None:
                     self.tracks[(w, p)]['death_in_interphase'].append(dii)

                else:
                
                    smi = self._has_second_mitosis(track_str, 1)
                    sma = self._has_second_mitosis(track_str, 3)
                    if smi is not None and sma is not None:
                        if len(smi) >= len(sma):
                            self.tracks[(w, p)]['second_mitosis_inter'].append(smi)
                        else:
                            self.tracks[(w, p)]['second_mitosis_apo'].append(sma)
                    elif smi is None and sma is not None:
                        self.tracks[(w, p)]['second_mitosis_apo'].append(sma)
                    elif smi is not None and sma is None:
                        self.tracks[(w, p)]['second_mitosis_inter'].append(smi)
                    elif smi is None and sma is None:
                        nsmnd = self._has_no_second_mitosis_no_death(track_str)
                        
                        if nsmnd is not None:
                            self.tracks[(w,p)]['no_second_mitosis_no_death'].append(track_str)
                        else:
                            self.tracks[(w,p)]['unclassified'].append(track_str)
                    
    def _has_death_in_mitosis(self, track_str):
        MIN_MITOSIS_LEN = 1
        MIN_APO_AFTER_MITOSIS = 1
        #print track_str
        MITOSIS_PATTERN = r'^1+2{%d,}3{%d}' % (MIN_MITOSIS_LEN, MIN_APO_AFTER_MITOSIS)
        second_mitosis_re = re.search(MITOSIS_PATTERN, track_str)
        
        MITOSIS_PATTERN_2 = r'^3+'
        second_mitosis_re_2 = re.search(MITOSIS_PATTERN_2, track_str)
        if second_mitosis_re is not None:
            start = 0
            end = second_mitosis_re.end()
            return track_str[start:end]
        elif second_mitosis_re_2 is not None:
            start = 0
            end = second_mitosis_re.end()
            return track_str[start:end]
        return None      
    
    def _has_death_in_interphase(self, track_str):
        MIN_INTER_LEN = 1
        MIN_APO_AFTER_MITOSIS = 1
        MITOSIS_PATTERN = r'^1+2+1{%d,}3{%d}' % (MIN_INTER_LEN, MIN_APO_AFTER_MITOSIS)
        second_mitosis_re = re.search(MITOSIS_PATTERN, track_str)
        if second_mitosis_re is not None:
            start = 0
            end = second_mitosis_re.end()
            return track_str[start:end]
        return None       
                    
    def _has_no_second_mitosis_no_death(self, track_str):
        MITOSIS_PATTERN = r'^1+2+1+[1,2]1+[1,2]1+$' 
        no_second_mitosis_re = re.search(MITOSIS_PATTERN, track_str)
        if no_second_mitosis_re is not None:
            start = 0
            end = no_second_mitosis_re.end()
            return track_str[start:end]
        return None 
                    
              
    def _has_second_mitosis(self, track_str, phase_after):
        MIN_INTER_LEN=20
        MIN_MITOSIS_LEN = 3
        MIN_PHASE_AFTER_MITOSIS = 1
        MITOSIS_PATTERN = r'^1+2+1{%d,}2{%d,}%d{%d}' % (MIN_INTER_LEN, MIN_MITOSIS_LEN, phase_after, MIN_PHASE_AFTER_MITOSIS)
        second_mitosis_re = re.search(MITOSIS_PATTERN, track_str)
        
        MITOSIS_PATTERN_2 = r'^1+2+1{%d,}2{%d,}' % (MIN_INTER_LEN, MIN_MITOSIS_LEN)
        second_mitosis_re_2 = re.search(MITOSIS_PATTERN_2, track_str)
        
        MITOSIS_PATTERN_3 = '2+3+$'
        second_mitosis_re_3 = re.search(MITOSIS_PATTERN_3, track_str)
        
        if second_mitosis_re is not None:
            start = 0
            end = second_mitosis_re.end()
            return track_str[start:end]
        
        elif (second_mitosis_re_2 is not None) and (phase_after==1):
            start = 0
            end = second_mitosis_re_2.end()
            return track_str[start:end]
        
        elif (second_mitosis_re_3 is not None) and (phase_after==3):
            start = 0
            end = second_mitosis_re_3.end()
            return track_str[start:end]
        
        
        return None
    
    
    def _plot_line(self, line, line_height, cmap, ax):
        old_l = line[0]
        old_l_idx = 0
        #print map(int,line)
        for l_idx, l1 in enumerate(line):
            if l1 != old_l:
                ax.plot([old_l_idx*self.time_lapse, l_idx*self.time_lapse], [line_height, line_height], color=cmap(int(old_l)), linewidth=3, solid_capstyle="butt")
                #print [old_l_idx, l_idx],
                old_l = l1
                old_l_idx = l_idx
        #print [old_l_idx, len(line)]
        ax.plot([old_l_idx*self.time_lapse, len(line)*self.time_lapse], [line_height, line_height], color=cmap(int(l1)), linewidth=3, solid_capstyle="butt")
      
    def _plot_curve(self, line, values, cmap, ax):
        old_l = line[0]
        old_l_idx = 0
        for l_idx, l1 in enumerate(line):
            if l1 != old_l:
                x = (numpy.arange(old_l_idx, l_idx+1) - self.onset_frame ) * self.time_lapse
                y = values[old_l_idx:(l_idx+1)]
                #print 'x', x[0], 'to', x[-1]
                #print 'y', y[0], 'to', y[-1]
                ax.plot(x, y, color=self.cmap(int(old_l)), linewidth=1)
                old_l = l1
                old_l_idx = l_idx
                
        x = (numpy.arange(old_l_idx, len(line)) - self.onset_frame) * self.time_lapse
        y = values[old_l_idx:(len(line))]
        ax.plot(x, y, color=self.cmap(int(l1)), linewidth=1)
        
                   
    def plot_proliferaton_timing_histogram(self):
        pp = PdfPages(self.output('_proliferation_ctrl.pdf'))
        for w, p in self.tracks:
            bins = numpy.zeros((21,), dtype=numpy.int32)
            cnt = 0
            f = pylab.figure(figsize=(12,7))
            ax = pylab.gca()
            for t in self.tracks[(w,p)]['track_ids']:
                onset_id = t[4]
                onset_time = int(self.mcellh5.get_position(w,str(p)).get_time_idx(onset_id))
                bins[onset_time/20]+=1
                
            pylab.bar(range(0,420,20), bins, 16, color='w')
            treatment = list(self.mcellh5.get_treatment_of_pos(w, p))
            ax.set_title('%s - %s (%s) '% tuple(treatment + [w,] ))
            ax.set_xlabel('Time (frames)')
            ax.set_ylabel('Mitotic Onset (count)')
            ax.set_xlim(0,420)
            ax.set_ylim(0,50)
            treatment = treatment.replace('/','_')
            pylab.savefig(pp, format='pdf')
            pylab.clf()
        pp.close()              
         
    def plot(self, title, split_len=1):
        def _plot_separator(cnt, label, col='k'):
            cnt+=2
            ax.axhline(cnt, color=col, linewidth=1)
            ax.text(460*self.time_lapse, cnt-0.5, label)
            cnt+=2
            return cnt
        
        def _cmp_len_of_first_inter(x,y):
            x_ = "".join(map(lambda x: "%02d" % x, x[0]))
            y_ = "".join(map(lambda x: "%02d" % x, y[0]))
            try:
                x_len = re.search(r"(02|03|04)+", x_).end()
                y_len = re.search(r"(02|03|04)+", y_).end()
                return cmp(y_len, x_len)
            except:
                return cmp(len(y[0]), len(x[0]))
     
        pp = PdfPages(self.output("%s.pdf") % title)
        
        exp_header = []
        exp_content = OrderedDict()
        
        exp_content['mito_int'] = []
        exp_content['mito_int_mito_int_mito'] = []
        exp_content['mito_int_mito_int_apo'] = []
        exp_content['mito_int_apo'] = []
        exp_content['mito_int_mito_apo'] = []
        exp_content['mito_apo'] = []
        exp_content['mito_unclassified'] = []

        exp_total = []
        
        for w, p in sorted(self.tracks):
            cnt = 0
            f = pylab.figure(figsize=(8,12))
            ax = pylab.gca()
            
            exp_header.append(w)
            
            for c in exp_content.keys():
                exp_content[c].append(len(self.tracks[(w,p)][c]))
            
            total_count = 0
            
            for line in sorted( self.tracks[(w,p)]['mito_int'], cmp=_cmp_len_of_first_inter):
                self._plot_line(line[0], cnt, self.cmap, ax)
                cnt+=1
            total_count += len(self.tracks[(w,p)]['mito_int'])
            cnt = _plot_separator(cnt, str(len(self.tracks[(w,p)]['mito_int'])))
            
            for line in sorted( self.tracks[(w,p)]['mito_int_mito_int_mito'], cmp=_cmp_len_of_first_inter): 
                self._plot_line(line[0], cnt, self.cmap, ax)
                cnt+=1
            total_count += len(self.tracks[(w,p)]['mito_int_mito_int_mito'])
                
            cnt = _plot_separator(cnt, str(len(self.tracks[(w,p)]['mito_int_mito_int_mito'])))
            
            for line in sorted(self.tracks[(w,p)]['mito_int_mito_int_apo'], cmp=_cmp_len_of_first_inter):
                self._plot_line(line[0], cnt, self.cmap, ax)
                cnt+=1
            total_count += len(self.tracks[(w,p)]['mito_int_mito_int_apo'])
                
            cnt = _plot_separator(cnt, str(len(self.tracks[(w,p)]['mito_int_mito_int_apo'])))
            
            for line in sorted(self.tracks[(w,p)]['mito_int_apo'], cmp=_cmp_len_of_first_inter):
                self._plot_line(line[0], cnt, self.cmap, ax)
                cnt+=1
            total_count += len(self.tracks[(w,p)]['mito_int_apo'])
                
            cnt = _plot_separator(cnt, str(len(self.tracks[(w,p)]['mito_int_apo'])))
                
            for line in sorted(self.tracks[(w,p)]['mito_int_mito_apo'], cmp=_cmp_len_of_first_inter):
                self._plot_line(line[0], cnt, self.cmap, ax)
                cnt+=1
            total_count += len(self.tracks[(w,p)]['mito_int_mito_apo'])
     
            cnt = _plot_separator(cnt, str(len(self.tracks[(w,p)]['mito_int_mito_apo'])))
            
            for line in sorted(self.tracks[(w,p)]['mito_apo'], cmp=_cmp_len_of_first_inter):
                self._plot_line(line[0], cnt, self.cmap, ax)
                cnt+=1
            total_count += len(self.tracks[(w,p)]['mito_apo'])
     
            cnt = _plot_separator(cnt, str(len(self.tracks[(w,p)]['mito_apo'])))
            
            for line in sorted(self.tracks[(w,p)]['mito_unclassified'], cmp=_cmp_len_of_first_inter):
                self._plot_line(line[0], cnt, self.cmap, ax)
                cnt+=1
                
            exp_total.append(total_count)
            
            total_count += len(self.tracks[(w,p)]['mito_unclassified'])
                
            #cnt = _plot_separator(cnt, str(total_count))
            
            #cnt +=4
            #ax.text(460*self.time_lapse, cnt-0.5, str(total_count))
            
            
            ax.set_ylim(0, cnt)
                
            ax.invert_yaxis()
            ax.set_xlim(0, 3500)
            title = '%s - %s (%s)' % tuple(list(self.mcellh5.get_treatment_of_pos(w, p)) + [w,])
            ax.set_title(title) 
            ax.set_xlabel('Time (min)')
            ax.set_yticklabels([])
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().set_ticks([])
            
            pylab.tight_layout()
            pylab.savefig(pp, format='pdf')
            for fmt in OUTPUT_FORMATS:
                pylab.savefig(self.output('fate_%s.%s' % (title, fmt))) 
            pylab.clf()
        pp.close()
        
        import csv
        with open(self.output('__dyinging_in_mito_or_apo.txt'), 'wb') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(exp_header)
            for v in exp_content.values():
                writer.writerow(v)
            writer.writerow(exp_total)
            
    def plot_fate_just_2_groups(self, title, split_len=1):
        def _plot_separator(cnt, label, col='k'):
            cnt+=2
            ax.axhline(cnt, color=col, linewidth=1)
            #ax.text(460*self.time_lapse, cnt-0.5, label)
            cnt+=2
            return cnt
        
        def _cmp_len_of_first_inter(x,y):
            x_ = "".join(map(lambda x: "%02d" % x, x[1]))
            y_ = "".join(map(lambda x: "%02d" % x, y[1]))
            try:
                x_len = re.search(r"(02|03|04)+", x_).end()
                y_len = re.search(r"(02|03|04)+", y_).end()
                return cmp(y_len, x_len)
            except:
                return cmp(len(y[0]), len(x[0]))
     
        pp = PdfPages(self.output("%s.pdf") % title)
        
        exp_header = []
        exp_content = OrderedDict()
        
        exp_content['mito_int'] = []
        exp_content['mito_int_mito_int_mito'] = []
        exp_content['mito_int_mito_int_apo'] = []
        exp_content['mito_int_apo'] = []
        exp_content['mito_int_mito_apo'] = []
        exp_content['mito_apo'] = []
        exp_content['mito_unclassified'] = []

        exp_total = []
        
        for w, p in sorted(self.tracks):
            cnt = 0
            f = pylab.figure(figsize=(8,12))
            ax = pylab.gca()
            
            exp_header.append(w)
            
            for c in exp_content.keys():
                exp_content[c].append(len(self.tracks[(w,p)][c]))
            
            total_count = 0
            
            combined_groups =  (self.tracks[(w,p)]['mito_int'] + 
                                self.tracks[(w,p)]['mito_int_mito_int_mito'] +
                                self.tracks[(w,p)]['mito_int_mito_int_apo'] +
                                self.tracks[(w,p)]['mito_int_apo'] +
                                self.tracks[(w,p)]['mito_int_mito_apo'])
            
            for line in sorted( combined_groups
                                , cmp=_cmp_len_of_first_inter):
                self._plot_line(line[0], cnt, self.cmap, ax)
                cnt+=1
            total_count += len(combined_groups)
            cnt = _plot_separator(cnt, str(len(self.tracks[(w,p)]['mito_int'])))
            
            
            
            for line in sorted(self.tracks[(w,p)]['mito_apo'], cmp=_cmp_len_of_first_inter):
                self._plot_line(line[0], cnt, CMAP17, ax)
                cnt+=1
            total_count += len(self.tracks[(w,p)]['mito_apo'])

                
            exp_total.append(total_count)
            
            total_count += len(self.tracks[(w,p)]['mito_unclassified'])
                
            #cnt = _plot_separator(cnt, str(total_count))
            
            #cnt +=4
            #ax.text(460*self.time_lapse, cnt-0.5, str(total_count))
            
            
            ax.set_ylim(0, cnt)
                
            ax.invert_yaxis()
            ax.set_xlim(0, 3500)
            title = '%s - %s (%s)' % tuple(list(self.mcellh5.get_treatment_of_pos(w, p)) + [w,])
            ax.set_title(title) 
            ax.set_xlabel('Time (min)')
            ax.set_yticklabels([])
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().set_ticks([])
            
            pylab.tight_layout()
            pylab.savefig(pp, format='pdf')
            for fmt in OUTPUT_FORMATS:
                pylab.savefig(self.output('fate2_%s.%s' % (title, fmt))) 
            pylab.clf()
        pp.close()
        
        import csv
        with open(self.output('__dyinging2_in_mito_or_apo.txt'), 'wb') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(exp_header)
            for v in exp_content.values():
                writer.writerow(v)
            writer.writerow(exp_total)
            
    def read_manual_annotations(self, filename, plate_id):
        def decode(code_string):
            code = code_string.split(',')
            res = {}
            for a,b in zip(code[::2], code[1::2]):
                res[int(a)] = int(b)
            
            return res
        
        f = h5py.File(filename, 'r')
        for w, p in self.tracks:
            print 'Read manual annotations', w, p
            cell5pos = self.mcellh5.get_position(w, p)
            
            anno_dset = f['/sample/0/plate/%s/experiment/%s/position/%s/feature/event_annotation/' % (plate_id, w, str(p))]
            self.tracks[(w, p)]['manual_annotations'] = []
            print len(anno_dset)
            for k, (track_id, anno_string) in enumerate(sorted(anno_dset)):
                if len(anno_string) > 0:
                    self.tracks[(w, p)]['manual_annotations'].append(decode(anno_string))
            
            
            
        
class CellFateAnalysisMultiHMM(CellFateAnalysis):    
                
    def setup_hmm(self):
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
        
    def mitotic_entries_within(self, frame_idx, apo_class_label = 5):
        
        with open(self.output("__mitotic_entries_and_intitial_cell_count_interphase_only.txt"), 'wb') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["Well", "Site", "T1", "T2", "LiveCellCountAtTimeZero", "MitoticEntries"])
            
            for w, p in sorted(self.tracks):
                number_of_events = len(self.tracks[(w,p)]["ids"])
                cellh5pos = self.mcellh5.get_position(w,str(int(p)))
                initial_cell_object_idx = numpy.nonzero(cellh5pos.get_object_table('primary__primary')["time_idx"] == 0)[0]
                interphace_count = numpy.count_nonzero(cellh5pos.get_class_label(initial_cell_object_idx) == 1)
                writer.writerow([w, str(p)] + list(self.mcellh5.get_treatment_of_pos(w, p)) + [ str(interphace_count), str(number_of_events)])
                          
                
    def classify_tracks(self, class_selector):
        class named_list(list):
            def __init__(self, name):
                self.name = name
                list.__init__(self)
                
            def append(self, x):
                #print x, "added to", self.name
                list.append(self, x)
        
        for w, p in self.tracks:
            cond = self.mcellh5.get_treatment_of_pos(w,p)
            self.tracks[(w,p)]['mito_int'] = named_list('mito_int')
            self.tracks[(w,p)]['mito_apo'] = named_list('mito_apo')
            self.tracks[(w,p)]['mito_int_mito_int_mito'] =  named_list('mito_int_mito_int_mito')
            self.tracks[(w,p)]['mito_int_mito_int_apo'] =  named_list('mito_int_mito_int_apo')
            self.tracks[(w,p)]['mito_int_apo'] = named_list('mito_int_apo')
            self.tracks[(w,p)]['mito_int_mito_apo'] = named_list('mito_int_mito_apo')
            self.tracks[(w,p)]['mito_unclassified'] = named_list('mito_unclassified')

            for t_idx, (track, track_ids) in enumerate(zip(self.tracks[w,p][class_selector], self.tracks[w,p]['track_ids'])):
                track_str = "".join(map(lambda x: "%02d" % x,track))
            
                #print self.tracks[w,p][class_selector][t_idx]
                #print self.tracks[w,p]['Raw class labels'][t_idx]
                
                
                mito_int = self._has_mito_int(track_str)
                if mito_int:
                    self.tracks[(w,p)]['mito_int'].append((mito_int, track, track_ids))
                    continue   
                
                mito_apo = self._has_mito_apo(track_str)
                if mito_apo:
                    self.tracks[(w,p)]['mito_apo'].append((mito_apo, track, track_ids))
                    continue 
                
                mito_int_mito_apo = self._has_mito_int_mito_apo(track_str)
                if mito_int_mito_apo:
                    self.tracks[(w,p)]['mito_int_mito_apo'].append((mito_int_mito_apo, track, track_ids))
                    continue 
                
                mito_int_mito_int_apo = self._has_mito_int_mito_int_apo(track_str)
                if mito_int_mito_int_apo:
                    self.tracks[(w,p)]['mito_int_mito_int_apo'].append((mito_int_mito_int_apo, track, track_ids))
                    continue 
                
                mito_int_mito_int_mito = self._has_mito_int_mito_int_mito(track_str)
                if mito_int_mito_int_mito:
                    self.tracks[(w,p)]['mito_int_mito_int_mito'].append((mito_int_mito_int_mito, track, track_ids))
                    continue 
                
                mito_int_apo = self._has_mito_int_apo(track_str)
                if mito_int_apo:
                    self.tracks[(w,p)]['mito_int_apo'].append((mito_int_apo, track, track_ids))
                    continue 
                
                          
                
                track_ints = map(int, list(split_str_into_len(track_str, 2)))
                self.tracks[(w,p)]['mito_unclassified'].append((track_ints, track, track_ids))
             
    def _has_mito_apo(self, track_str):
        MIN_MITOSIS_LEN = 1
        MIN_APO_AFTER_MITOSIS = 1
        MITOSIS_PATTERN = r'^(01)+(02|03|04){%d,}(17)' % (MIN_MITOSIS_LEN)
        second_mitosis_re = re.search(MITOSIS_PATTERN, track_str)
        if second_mitosis_re is not None:
            end = second_mitosis_re.end() / 2
            track_ints = map(int, list(split_str_into_len(track_str, 2)))
            return [track_ints[i] if i < 4 else 17 for i in xrange(len(track_ints))][:end]
            # all red
        return None      
    
    def _has_mito_int_apo(self, track_str):
        MIN_INTER_LEN = 1
        MIN_APO_AFTER_MITOSIS = 1
        MIN_MITO_LEN = 3 
        MITOSIS_PATTERN = r'^(01)+(?P<blue>(02|03|04){%d,})(05){%d,}(17)' % (MIN_MITO_LEN, MIN_INTER_LEN)
        second_mitosis_re = re.search(MITOSIS_PATTERN, track_str)
        if second_mitosis_re is not None:
            end = second_mitosis_re.end() / 2
            end_blue = second_mitosis_re.end('blue') / 2
            track_ints = map(int, list(split_str_into_len(track_str, 2)))
            return [track_ints[i] if i < 4 else 2 if i < end_blue else 18 for i in xrange(len(track_ints))][:end]
            # blue, green
        return None       
                    
    def _has_mito_int(self, track_str):
        MITOSIS_PATTERN = r'^(01)+(?P<blue>(02|03|04)+)(05)*$' 
        no_second_mitosis_re = re.search(MITOSIS_PATTERN, track_str)
        if no_second_mitosis_re is not None:
            end = no_second_mitosis_re.end() / 2
            end_blue = no_second_mitosis_re.end('blue') / 2
            track_ints = map(int, list(split_str_into_len(track_str, 2)))
            return [track_ints[i] if i < 4 else 2 for i in xrange(len(track_ints))][:end_blue]
            # just blue
        return None 
                    
              
    def _has_mito_int_mito_int_mito(self, track_str):

        MITOSIS_PATTERN = r'^(01)+(02|03|04)+(05)+(06|07|08)+(09)*(10|11|12)*(13)*(14|15|16)*'
        second_mitosis_re = re.search(MITOSIS_PATTERN, track_str)
        
        if (second_mitosis_re is not None) and (track_str[-2:] != '17'):
            end = second_mitosis_re.end() / 2
            track_ints = map(int, list(split_str_into_len(track_str, 2)))
            return track_ints[:end]
            # as is
        return None
    
    def _has_mito_int_mito_int_apo(self, track_str):
        MIN_INTER_LEN=1
        MIN_MITOSIS_LEN = 1
        MIN_PHASE_AFTER_MITOSIS = 1
        MITOSIS_PATTERN = r'^(01)+(?P<blue>(02|03|04)+(05){%d,}(06|07|08){%d,})(09){%d,}((?P<blue_2>(10|11|12)+)(13)+)*(17)' % (MIN_INTER_LEN, MIN_MITOSIS_LEN, MIN_PHASE_AFTER_MITOSIS)
        second_mitosis_re = re.search(MITOSIS_PATTERN, track_str)
        
        if second_mitosis_re is not None:
            end = second_mitosis_re.end() / 2
            end_blue = second_mitosis_re.end('blue') / 2
            end_blue_2 = second_mitosis_re.end('blue_2') / 2
            if end_blue_2 > end_blue:
                end_blue = end_blue_2
            track_ints = map(int, list(split_str_into_len(track_str, 2)))
            return [track_ints[i] if i < end_blue else 18 for i in xrange(len(track_ints))][:end]
            # blue, as is, green
        return None
    
    def _has_mito_int_mito_apo(self, track_str):
        MIN_INTER_LEN=1
        MIN_MITOSIS_LEN = 1
        MIN_PHASE_AFTER_MITOSIS = 1
        MITOSIS_PATTERN = r'^(01)+(?P<blue>(02|03|04)+(05){%d,}(06|07|08){%d,})((09)+(10|11|12)+((13)+(14|15|16)+)*)*(17)' % (MIN_INTER_LEN, MIN_MITOSIS_LEN)
        second_mitosis_re = re.search(MITOSIS_PATTERN, track_str)
        
        if second_mitosis_re is not None:
            end = second_mitosis_re.end() / 2
            end_blue = second_mitosis_re.end('blue') / 2
            track_ints = map(int, list(split_str_into_len(track_str, 2)))
            return [track_ints[i] for i in xrange(len(track_ints))][:end]
            # blue, as is, red  
        return None
    
    def plot_mitotic_timing(self, track_selector):
        from plot_ext import spreadplot
        import re
        from collections import OrderedDict
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
                
            treatment = self.mcellh5.get_treatment_of_pos(w, p)[1]
            xticklabels.append("%s_%02d" % (w,p))
        
        from itertools import izip_longest
        import csv

        with open(self.output('__mito_timing_%s.txt' % self.plate_id), 'wb') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(xticklabels)
            writer.writerows(izip_longest(*mito_timing.values(), fillvalue=""))
               
        with open(self.output('__mito_classes_%s.txt' % self.plate_id), 'wb') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(xticklabels)
            writer.writerows(izip_longest(*mito_class.values(), fillvalue="")) 
        
        colors = ['g', 'g',] + ['r']*5 + ['b']*5 + ["#ffa500"]*3 + ['r']*5 + ['b']*5 + ["#ffa500"]*3 + ['k']*50
#         ax = p(mito_timing.values(), xticklabels=xticklabels, colors=colors, spread_type='g', spread=0.1)
        pylab.boxplot(mito_timing.values())
        ax = pylab.gca()
        ax.set_xticklabels(xticklabels, rotation=90)
        ax.set_title('Mitotic Timing')
        ax.set_ylabel('Mitotic timing (min)')
        pylab.tight_layout()
        pylab.savefig(pp, format='pdf')
        
        pp.close()
    
    
    
class CellFateMitoticTiming(CellFateAnalysis):
    def __init__(self, plate_id, ch5_file, mapping_file, events_before_frame=9999, onset_frame=4, sites=None, rows=None, cols=None, locations=None, time_lapse=None):
        CellFateAnalysis.__init__(self, plate_id, ch5_file, mapping_file, events_before_frame=events_before_frame, onset_frame=onset_frame, sites=sites, rows=rows, cols=cols, locations=locations,  time_lapse=time_lapse)
    
    def setup_hmm_2308(self, k_classes, constraint_xml):
        constraints = HMMConstraint(constraint_xml)
        
        transmat = numpy.array([
                                [0.1,  0.1,  0.0,  0.0,  0.0, 0.1, 0.1],
                                [0.0,  2  ,  0.1,  0.0,  0.0, 0.1, 0.1],
                                [0.0,  0.0,  2  ,  0.1,  0.0, 0.1, 0.1],
                                [0.1,  0.0,  0.0,  2  ,  0.1, 0.1, 0.0],
                                [0.1,  0.0,  0.0,  0.0,  2  , 0.1, 0.0],
                                [0.0,  0.0,  0.0,  0.0,  0.0, 1  , 0.0],
                                [0.0,  0.1,  0.1,  0.0,  0.0, 0.1,  2 ],
                                ])
        transmat = normalize(transmat, axis=1, eps=0)
        
        est = HMMAgnosticEstimator(k_classes, transmat, numpy.ones((k_classes, k_classes)), numpy.ones((k_classes, )) )
        est.constrain(constraints)
        self.hmm = hmm.MultinomialHMM(n_components=est.nstates, transmat=transmat, startprob=est.startprob, init_params="")
        self.hmm._set_emissionprob(est.emis) 
        
    def setup_hmm_2408(self, k_classes, constraint_xml):
        constraints = HMMConstraint(constraint_xml)
        
        transmat = numpy.array([
                                [0.1,  0.1,  0.0,  0.0,  0.1, 0.1, 0.1], # inter
                                [0.0,  2  ,  0.1,  0.0,  0.1, 0.1, 0.1], # prometa
                                [0.0,  0.0,  2  ,  0.1,  0.1, 0.1, 0.1], # meta
                                [0.1,  0.0,  0.0,  2  ,  0.1, 0.0, 0.1], # ana
                                [0.0,  0.0,  0.0,  0.0,  0.0, 0.0,  1 ], # apo 2
                                [0.0,  0.1,  0.1,  0.0,  0.1, 2  , 0.1], # unaligned
                                [0.1,  0.0,  0.0,  0.0,  0.0, 0.0,  1 ], # apo 2
                                ])
        transmat = normalize(transmat, axis=1, eps=0)
        
        est = HMMAgnosticEstimator(k_classes, transmat, numpy.ones((k_classes, k_classes)), numpy.ones((k_classes, )) )
        est.constrain(constraints)
        self.hmm = hmm.MultinomialHMM(n_components=est.nstates, transmat=transmat, startprob=est.startprob, init_params="")
        self.hmm._set_emissionprob(est.emis)
        
    def setup_hmm(self, k_classes, constraint_xml):
        constraints = HMMConstraint(constraint_xml)
        
        transmat = numpy.array([
                                [0.2,  0.1,  0.0,  0.0,  0.1,], # inter
                                [0.0,  10 ,  0.1,  0.0,  0.1,], # prometa
                                [0.0,  0.0,  10 ,  0.1,  0.1,], # meta
                                [0.2,  0.0,  0.0,  2  ,  0.1,], # ana
                                [0.0,  0.0,  0.0,  0.0,  1  ,], # apo 2
                                ])
        transmat = normalize(transmat, axis=1, eps=0)
        
        est = HMMAgnosticEstimator(k_classes, transmat, numpy.ones((k_classes, k_classes)), numpy.ones((k_classes, )) )
        est.constrain(constraints)
        self.hmm = hmm.MultinomialHMM(n_components=est.nstates, transmat=transmat, startprob=est.startprob, init_params="")
        self.hmm._set_emissionprob(est.emis)
    
    
    def plot_mitotic_timing(self):
        from plot_ext import spreadplot
        import re
        from collections import OrderedDict
        pp = PdfPages(self.output('mitotic_timing.pdf'))
        f = pylab.figure(figsize=(20,8))
        mito_timing = OrderedDict()
        
        xticklabels=[]
        
        for w, p in sorted(self.tracks.keys()):
            
            mito_timing[(w,p)] = []
            if (w,p) not in self.tracks:
                continue
            for t in self.tracks[(w,p)]['HMM']:
                track_str = "".join(map(str, t))
                mito_re = re.search(r'1+(?P<mito>(2|3)+)(1|4|5)+', track_str)
                if mito_re is not None:
                    span = mito_re.span('mito')
                    mito_timing[(w,p)].append((span[1] - span[0])* self.time_lapse)
                
            treatment = self.mcellh5.get_treatment_of_pos(w, p)[1]
            xticklabels.append("%s_%02d" % (w,p))
        
        from itertools import izip_longest
        import csv

        with open(self.output('mito_timing_%s.txt' % self.plate_id), 'wb') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(xticklabels)
            writer.writerows(izip_longest(*mito_timing.values(), fillvalue=""))   
        
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
        

            
            
EXP_LOOKUP = {
         '002200':
            {
             'ch5_file': "M:/experiments/Experiments_002200/002200/_meta/Cecog/Aalysis_with_split/hdf5/_all_positions.ch5",
             'mapping_file': "M:/experiments/Experiments_002200/002200/_meta/Cecog/Mapping/002200_2.txt",
             'time_lapse': 6.7, 
             'events_before_frame': 108, # in frames
             'onset_frame': 4, # in frames
             'hmm_constraint_file':'hmm_constraints/graph_5_to_17_ms_special.xml',
             'hmm_n_classes': 17,
             'hmm_n_obs': 5,
             'output_dir' : 'M:/experiments/Experiments_002200/002200/_meta/fate',
             'securin_region' : "tertiary__expanded"
             },
              
        '002308':
            {
             'ch5_file': r"M:\experiments\Experiments_002300\002308\_meta\Analysis\hdf5\_all_positions.ch5",
             'mapping_file': r"M:\experiments\Experiments_002300\002308\_meta\002308.txt",
             'time_lapse': 4.7, 
             'events_before_frame': 150, # in frames
             'onset_frame': 4, # in frames
             },
              
              
        '002338':
            {
             'ch5_file': "M:/experiments/Experiments_002300/002338/002338/_meta/Analysis/hdf5/_all_positions.ch5",
             'mapping_file': "M:/experiments/Experiments_002300/002338/002338/_meta/Cecog/Mapping/002338.txt",
             'time_lapse': 8, 
             'events_before_frame': 90, # in frames
             'onset_frame': 5, # in frames
             'hmm_constraint_file':'hmm_constraints/graph_5_to_17_ms_special.xml',
             'hmm_n_classes': 17,
             'hmm_n_obs': 5,
              'output_dir' : 'M:/experiments/Experiments_002300/002338/002338/_meta/fate',
             'securin_region' : "tertiary__expanded"
             },
         '002377':
            {
             'ch5_file':     "M:/experiments/Experiments_002300/002377/_meta/Analysis/hdf5/_all_positions.ch5",
             'mapping_file': "M:/experiments/Experiments_002300/002377/_meta/Mapping/002377.txt",
             'time_lapse': 5.9, 
             'events_before_frame': 125, # in frames
             'onset_frame': 5, # in frames
             'hmm_constraint_file':'hmm_constraints/graph_5_to_17_ms_special.xml',
             'hmm_n_classes': 17,
             'hmm_n_obs': 5,
             'output_dir' : 'M:/experiments/Experiments_002300/002377/_meta/fate/test',
             'securin_region' : "secondary__expanded"
             },
              
              '002325':
            {
             'ch5_file':     "M:/experiments/Experiments_002300/002325/_meta/Analysis/hdf5/_all_positions.ch5",
             'mapping_file': "M:/experiments/Experiments_002300/002325/_meta/Mapping/002325.txt",
             'time_lapse': 4.6, 
             'events_before_frame': 160, # in frames
             'onset_frame': 5, # in frames
             'hmm_constraint_file':'hmm_constraints/graph_5_to_17_ms_special.xml',
             'hmm_n_classes': 17,
             'hmm_n_obs': 5,
#              'output_dir' : 'M:/experiments/Experiments_002300/002325/_meta/fate',
             'securin_region' : "secondary__expanded"
             },
              
              '002288':
            {
             'ch5_file':     "M:/experiments/Experiments_002200/002288/_meta/Analysis/hdf5/_all_positions.ch5",
             'mapping_file': "M:/experiments/Experiments_002200/002288/_meta/Mapping/002288_1.txt",
             'time_lapse': 4.5, 
             'events_before_frame': 160, # in frames
             'onset_frame': 4, # in frames
             'hmm_constraint_file':'hmm_constraints/graph_5_to_17_ms_special.xml',
             'hmm_n_classes': 17,
             'hmm_n_obs': 5,
             #'output_dir' : 'M:/experiments/Experiments_002200/002288/_meta/fate',
             'securin_region' : "tertiary__expanded"
             },
              
              '002301':
            {
             'ch5_file':     "M:/experiments/Experiments_002300/002301/_meta/Analysis/hdf5/_all_positions.ch5",
             'mapping_file': "M:/experiments/Experiments_002300/002301/_meta/Mapping/002301_01.txt",
             'time_lapse': 4.6, 
             'events_before_frame': 160, # in frames
             'onset_frame': 4, # in frames
             'hmm_constraint_file':'hmm_constraints/graph_5_to_17_ms_special.xml',
             'hmm_n_classes': 17,
             'hmm_n_obs': 5,
             #'output_dir' : 'M:/experiments/Experiments_002300/002301/_meta/fate',
             'securin_region' : "tertiary__expanded"
             },  
        '002404':
            {
             'ch5_file': "M:/experiments/Experiments_002400/002404/_meta/Analysis/Analysis_half_plate_2_100/hdf5/_all_positions.ch5",
             'mapping_file': "M:/experiments/Experiments_002400/002404/_meta/Mapping/002404.txt",
             'time_lapse': 4.9, 
             'events_before_frame': 9999, # in frames
             'onset_frame': 4, # in frames
             },  
        '002405':
            {
             'ch5_file': "M:/experiments/Experiments_002400/002405/_meta/Analysis/half_plate_2_100/hdf5/_all_positions.ch5",
             'mapping_file': "M:/experiments/Experiments_002400/002404/_meta/Mapping/002404.txt",
             'time_lapse': 4.8, 
             'events_before_frame': 9999, # in frames
             'onset_frame': 4, # in frames
             },   
        '002408':
            {
             'ch5_file': "M:/experiments/Experiments_002400/002408/_meta/Analysis/half_plate_2_100/hdf5/_all_positions.ch5",
             'mapping_file': "M:/experiments/Experiments_002400/002404/_meta/Mapping/002404.txt",
             'time_lapse': 4.8, 
             'events_before_frame': 9999, # in frames
             'onset_frame': 4, # in frames
             },  
        '002410':
            {
             'ch5_file': "M:/experiments/Experiments_002400/002410/_meta/Analysis/hdf5/_all_positions.ch5",
             'mapping_file': "M:/experiments/Experiments_002400/002411/_meta/Mapping/002411.txt",
             'time_lapse': 5.4, 
             'hmm_constraint_file':'hmm_constraints/graph_5_to_17_ms_special.xml',
             'hmm_n_classes': 17,
             'hmm_n_obs': 5,
             'events_before_frame': 9999, # in frames
             'onset_frame': 4, # in frames
             },        
        
        '002411':
            {
             'ch5_file': "M:/experiments/Experiments_002400/002411/_meta/Analysis/hdf5/_all_positions.ch5",
             'mapping_file': "M:/experiments/Experiments_002400/002411/_meta/Mapping/002411.txt",
             'time_lapse': 5.4, 
             'hmm_constraint_file':'hmm_constraints/graph_5_to_17_ms_special.xml',
             'hmm_n_classes': 17,
             'hmm_n_obs': 5,
             'events_before_frame': 9999, # in frames
             'onset_frame': 4, # in frames
             },  
        '002415':
            {
             'ch5_file': "M:/experiments/Experiments_002400/002415/_meta/Analysis/hdf5/_all_positions.ch5",
             'mapping_file': "M:/experiments/Experiments_002400/002411/_meta/Mapping/002411.txt",
             'time_lapse': 5.4, 
             'hmm_constraint_file':'hmm_constraints/graph_5_to_17_ms_special.xml',
             'hmm_n_classes': 17,
             'hmm_n_obs': 5,
             'events_before_frame': 9999, # in frames
             'onset_frame': 4, # in frames
             },
        '002382':
            {
             'ch5_file': "M:/experiments/Experiments_002300/002382/_meta/Analysis/hdf5/_all_positions.ch5",
             'mapping_file': "M:/experiments/Experiments_002300/002382/_meta/Mapping/002382.txt",
             'time_lapse': 8.0, 
             'hmm_constraint_file':'hmm_constraints/graph_5_to_17_ms_special.xml',
             'hmm_n_classes': 17,
             'hmm_n_obs': 5,
             'output_dir' : 'M:/experiments/Experiments_002300/002382/_meta/fate',
             'events_before_frame': 90, # in frames
             'onset_frame': 4, # in frames
             'securin_region' : "tertiary__expanded"
             },
              
        '002383':
           {
             'ch5_file': "M:/experiments/Experiments_002300/002383/_meta/Analysis/hdf5/_all_positions.ch5",
             'mapping_file': "M:/experiments/Experiments_002300/002383/_meta/Mapping/002383.txt",
             'time_lapse': 5.1, 
             'hmm_constraint_file':'hmm_constraints/graph_5_to_17_ms_special.xml',
             'hmm_n_classes': 17,
             'hmm_n_obs': 5,
             'output_dir' : 'M:/experiments/Experiments_002300/002383/_meta/fate',
             'events_before_frame': 145, # in frames
             'onset_frame': 4, # in frames
#              'securin_region' : "tertiary__expanded"
             },
        
         '002587':
            {
             'ch5_file': "M:/experiments/Experiments_002500/002587/_meta/Analysis/hdf5/_all_positions.ch5",
             'mapping_file': "M:/experiments/Experiments_002500/002587/_meta/Mapping/002587.txt",
             'time_lapse': 4.5, 
             'hmm_constraint_file':'hmm_constraints/graph_5_to_17_ms_special.xml',
             'hmm_n_classes': 17,
             'hmm_n_obs': 5,
             'output_dir' : 'M:/experiments/Experiments_002500/002587/_meta/fate',
             'events_before_frame': 160, # in frames
             'onset_frame': 4, # in frames
             },
              
          '002760':
            {
             'ch5_file': "M:/experiments/Experiments_002700/002760/_meta/Analysis/time lapse half plate/hdf5/_all_positions.ch5",
             'mapping_file': "M:/experiments/Experiments_002700/002760/_meta/Mapping/002760.txt",
             'time_lapse': 4.6, 
             'hmm_constraint_file':'hmm_constraints/graph_5_to_17_ms_special.xml',
             'hmm_n_classes': 17,
             'hmm_n_obs': 5,
             'output_dir' : 'M:/experiments/Experiments_002700/002760/_meta/fate',
             'events_before_frame': 156, # in frames
             'onset_frame': 4, # in frames
             },
              
          '002773':
            {
             'ch5_file': "M:/experiments/Experiments_002700/002773/_meta/Analysis/time lapse half plate/hdf5/_all_positions.ch5",
             'mapping_file': "M:/experiments/Experiments_002700/002773/_meta/Mapping/002773.txt",
             'time_lapse': 4.8, 
             'hmm_constraint_file':'hmm_constraints/graph_5_to_17_ms_special.xml',
             'hmm_n_classes': 17,
             'hmm_n_obs': 5,
             'output_dir' : 'M:/experiments/Experiments_002700/002773/_meta/fate',
             'events_before_frame': 150, # in frames
             'onset_frame': 4, # in frames
             },
              
          '002787':
            {
             'ch5_file': "M:/experiments/Experiments_002700/002787/_meta/Analysis/time lapse half plate/hdf5/_all_positions.ch5",
             'mapping_file': "M:/experiments/Experiments_002700/002787/_meta/Mapping/002787.txt",
             'time_lapse': 4.5, 
             'hmm_constraint_file':'hmm_constraints/graph_5_to_17_ms_special.xml',
             'hmm_n_classes': 17,
             'hmm_n_obs': 5,
             'output_dir' : 'M:/experiments/Experiments_002700/002787/_meta/fate',
             'events_before_frame': 160, # in frames
             'onset_frame': 4, # in frames
             },
              
          '002614_1' :
            {
             'ch5_file': "M:/experiments/Experiments_002600/002614/_meta/Analysis_classifie_anaphase_corrected/hdf5/_all_positions.ch5",
             'mapping_file': "M:/experiments/Experiments_002600/002614/_meta/Mapping/2614.txt",
             'time_lapse': 5.7, 
             'hmm_constraint_file':'hmm_constraints/graph_5_to_17_ms_special.xml',
             'hmm_n_classes': 17,
             'hmm_n_obs': 5,
             'output_dir' : 'M:/experiments/Experiments_002600/002614/_meta/Analysis_classifie_anaphase_corrected/fate',
             'events_before_frame': 130, # in frames
             'onset_frame': 4, # in frames
             },   
              
            '002614_2' :
            {
             'ch5_file': "M:/experiments/Experiments_002600/002614/_meta/Analysis_classifier_140901_shorter_track_length/hdf5/_all_positions.ch5",
             'mapping_file': "M:/experiments/Experiments_002600/002614/_meta/Mapping/2614.txt",
             'time_lapse': 5.7, 
             'hmm_constraint_file':'hmm_constraints/graph_5_to_17_ms_special.xml',
             'hmm_n_classes': 17,
             'hmm_n_obs': 5,
             'output_dir' : 'M:/experiments/Experiments_002600/002614/_meta/Analysis_classifier_140901_shorter_track_length/fate',
             'events_before_frame': 130, # in frames
             'onset_frame': 2, # in frames
             },  
              
             '002666_1' :
            {
             'ch5_file': "M:/experiments/Experiments_002600/002666/002666_1/2666_1_1/_meta/Analysis_shorter_track_lenght/hdf5/_all_positions.ch5",
             'mapping_file': "M:/experiments/Experiments_002600/002666/002666_1/2666_1_2/_meta/Mapping/002666_1.txt",
             'time_lapse': 6.5, 
             'hmm_constraint_file':'hmm_constraints/graph_5_to_17_ms_special.xml',
             'hmm_n_classes': 17,
             'hmm_n_obs': 5,
             'output_dir' : 'M:/experiments/Experiments_002600/002666/002666_1/2666_1_1/_meta/fate',
             'events_before_frame': 120, # in frames
             'onset_frame': 2, # in frames
             },  
              
              '002666_2' :
            {
             'ch5_file': "M:/experiments/Experiments_002600/002666/002666_1/2666_1_2/_meta/Analysis_shorter_track_length/hdf5/_all_positions.ch5",
             'mapping_file': "M:/experiments/Experiments_002600/002666/002666_1/2666_1_2/_meta/Mapping/002666_1.txt",
             'time_lapse': 6.5, 
             'hmm_constraint_file':'hmm_constraints/graph_5_to_17_ms_special.xml',
             'hmm_n_classes': 17,
             'hmm_n_obs': 5,
             'output_dir' : 'M:/experiments/Experiments_002600/002666/002666_1/2666_1_2/_meta/fate',
             'events_before_frame': 120, # in frames
             'onset_frame': 2, # in frames
             },  
              
      }
       
       
def fate_mitotic_time(plate_id):
    locs = [('G', 4), ('G', 5),
            #('B', 5), ('C', 5), ('D', 5), ('E', 5), ('F', 5),
            #('B', 11), ('C', 11), ('D', 11), ('E', 11), ('F', 11),
            ('B', 3), ('C', 3), ('D', 3),
            ]
    locs = None
    
    pm = CellFateAnalysisMultiHMM(plate_id, locations=locs, **EXP_LOOKUP[plate_id])

    pm.fate_tracking('Raw class labels')
#     pm.setup_hmm(5, 'hmm_constraints/graph_6_left2right.xml')
    pm.setup_hmm()
    pm.predict_hmm('Raw class labels', 'Multi State HMM')   
    pm.cmap = matplotlib.colors.ListedColormap(map(lambda x: hex_to_rgb(x), 
                                                   ['#FFFFFF', 
                                                    '#00ff00', 
                                                    '#ff8000',
                                                    '#d28dce',
                                                    '#0055ff',
                                                    '#ff0000']), 'classification_cmap')
    pm.plot_tracks(['Raw class labels', 'Multi State HMM', 'Multi State HMM', 'Multi State HMM',], 
                   [pm.cmap, pm.cmap_cycle_3, CMAP17, CMAP17_MULTI],
                   ['Raw class labels', 'HMM class labels', 'HMM Multi simple', 'HMM Multi full'],
                    'trajectories_all')
    pm.classify_tracks('Multi State HMM')
    pm.plot_mitotic_timing('Multi State HMM')   
    
      
def fate_mutli_bi(plate_id):
    pm = CellFateAnalysisMultiHMM(plate_id, 
#                                 rows=("A", ), 
#                                 cols=(3,), 
                                **EXP_LOOKUP[plate_id])
    
#     pm.mitotic_entries_within(pm.events_before_frame)
    pm.fate_tracking(out_name='Raw class labels')
    pm.setup_hmm()
    pm.predict_hmm('Raw class labels', 'Multi State HMM')   
     
    pm.plot_tracks(['Raw class labels', 'Multi State HMM', 'Multi State HMM', 'Multi State HMM',], 
                   [pm.cmap, pm.cmap_cycle_3, CMAP17, CMAP17_MULTI],
                   ['Raw class labels', 'HMM class labels', 'HMM Multi simple', 'HMM Multi full'],
                    'trajectories_all')
      
    pm.classify_tracks('Multi State HMM')
    #pm.plot_mitotic_timing('Multi State HMM')
    pm.cmap = CMAP17
    pm.plot('__fate_all', 2)
    pm.cmap = CMAP17SIMPLE
    pm.plot_fate_just_2_groups('__fate_simple_all', 2)
    
    if False: # Securin ?
    
        securin_region = EXP_LOOKUP[plate_id]['securin_region']
         
        pm.event_curves('Multi State HMM',
                        '_securin_degradation_short',
                        'tertiary__expanded',
                        'n2_avg',
                        False,
                        pm.cmap,
                        (-20,240),
                        (0,1.5),
                               )
           
        pm.event_curves('Multi State HMM',
                        '_securin_degradation_long',
                        'tertiary__expanded',
                        'n2_avg',
                        True,
                        pm.cmap,
                        (-20, 1200),
                        (0,1.5),
                               )
          
        pm.event_mean_curves('Multi State HMM',
                                '_securin_degradation_short_mean',
                                'tertiary__expanded',
                                'n2_avg',
                                False,
                                pm.cmap,
                                (-20,240),
                                (0,1.5),
                                       )
          
        pm.event_mean_curves('Multi State HMM', 
                        '_securin_degradation_long_mean',
                        'tertiary__expanded',
                        'n2_avg',
                        True,
                        pm.cmap,
                        (-20,1200),
                        (0,1.5),
                               )
        
        pm.event_mean_fate_curves('Multi State HMM', 
                        '_securin_degradation_per_fate_mean_long',
                        securin_region,
                        'n2_avg',
                        True,
                        pm.cmap,
                        (-20,1200),
                        (0,1.5),
                               )
          
        pm.event_mean_fate_curves('Multi State HMM', 
                        '_securin_degradation_per_fate_mean_short',
                        securin_region,
                        'n2_avg',
                        False,
                        pm.cmap,
                        (-20,120),
                        (0,1.5),
                               )
          
        pm.event_fate_curves('Multi State HMM', 
                        '_securin_degradation_per_fate_all_long',
                        securin_region,
                        'n2_avg',
                        True,
                        pm.cmap,
                        (-20,1200),
                        (0,1.5),
                               )
          
        pm.event_fate_curves('Multi State HMM', 
                        '_securin_degradation_per_fate_all_short',
                        securin_region,
                        'n2_avg',
                        False,
                        pm.cmap,
                        (-20,120),
                        (0,1.5),
                               )
    
    print 'CellFateAnalysisMultiHMM done'
    
if __name__ == "__main__":
    #fate_mutli_bi('002200')
#     fate_mutli_bi('002338')
    #fate_mutli_bi('002377')
#     fate_mutli_bi('002325')
#     fate_mutli_bi('002288')
#     fate_mutli_bi('002301')
#     fate_mutli_bi('002382')
#     fate_mutli_bi('002383')
#     fate_mutli_bi('002614_1')
#     fate_mutli_bi('002614_2')
    
#     fate_mutli_bi('002666_1')
#     fate_mutli_bi('002666_2')

#     fate_mutli_bi('002587')
    #fate_mitotic_time()
#     fate_mitotic_time('002404')
#     fate_mitotic_time('002405')
#     fate_mitotic_time('002408')
#     fate_mitotic_time('002410')
#     fate_mitotic_time('002415')
#     fate_mitotic_time('002411')
#     fate_mitotic_time('002587')
    fate_mitotic_time('002760')
    
    print 'FINISH'

