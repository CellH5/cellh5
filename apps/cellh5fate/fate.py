import numpy
import pylab as plt
import cellh5
import cPickle as pickle
from numpy import recfromcsv
import pandas
import time
import vigra
import h5py
import matplotlib.pyplot as mpl
import os
import collections
import functools
import time
import cProfile
import re
import random
import pylab
import myhmm

from matplotlib.mlab import PCA
from scipy.stats import nanmean
from matplotlib.backends.backend_pdf import PdfPages

from itertools import cycle
from cecog.util.color import rgb_to_hex
import matplotlib

from matplotlib.backends.backend_pdf import PdfPages


from sklearn import hmm
from estimator import HMMConstraint, HMMAgnosticEstimator, normalize
hmm.normalize = lambda A, axis=None: normalize(A, axis, eps=10e-99)
#hmm.normalize = lambda A, axis: normalize(A, axis=len(A.shape)-1, eps=eps)

# def matrix_to_dict(matrix):
#     d = {}
#     for row in range(matrix.shape[0]):
#         d[row] = {}
#         if len(matrix.shape) == 2:
#             for col in range(matrix.shape[1]):
#                 d[row][col] = numpy.log2(matrix[row, col])
#         else:
#             d[row] = numpy.log2(matrix[row])
#     return d

def split_str_into_len(s, l=2):
    """ Split a string into chunks of length l """
    return [s[i:i+l] for i in range(0, len(s), l)]

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

def class_lookup(class_label):
    color_dct = {
                 1 : '#00AA00',
                 2 : '#ff8000',
                 3 : '#ff8000',
                 4 : '#ff8000',
                 5 : '#ff8000',
                 6 : '#00AA00',
                 7 : '#00AA00',
                 8 : '#FF0000',
                 9 : '#00FFFF',
                10 : '#00FFFF' }
    return color_dct[class_label]
                
cmap3 = matplotlib.colors.ListedColormap(map(lambda x: hex_to_rgb(x), 
                                                   ['#FFFFFF', 
                                                    '#AAAAAA', 
                                                    '#0000FF',
                                                    '#FF0000']), 'classification_cmap')
cmap53 = matplotlib.colors.ListedColormap(map(lambda x: hex_to_rgb(x), 
                                                   ['#FFFFFF', 
                                                    '#AAAAAA', 
                                                    '#0000FF',
                                                    '#FF0000']), 'classification_cmap')     

cmap17 = matplotlib.colors.ListedColormap(map(lambda x: hex_to_rgb(x), 
                                                   ['#FFFFFF',
                                                     
                                                    '#E31A1C', 
                                                    '#FD8D3C',
                                                    '#FECC5C',
                                                    '#FFFFB2',
                                                    
                                                    '#238443', 
                                                    '#78C679',
                                                    '#C2E699',
                                                    '#FFFFCC',
                                                    
                                                    '#2171B5', 
                                                    '#6BAED6',
                                                    '#BDD7E7',
                                                    '#EFF3FF',
                                                    
                                                    '#238443', 
                                                    '#78C679',
                                                    '#C2E699',
                                                    '#FFFFCC',
                                                    
                                                    '#000000',
                                                    
                                                    ]), 'classification_cmap')                        

class ColorPicker(object):
    def __init__(self, color_list=None):
        if color_list is None:
            import colorbrewer
            color_list = []
            color_list.extend(colorbrewer.Set3[12])
            color_list.extend(colorbrewer.Oranges[5])
            color_list.extend(colorbrewer.Blues[5])
            color_list.extend(colorbrewer.Reds[5])
            color_list.extend(colorbrewer.Greens[5])
            import random
            random.shuffle(color_list)
            
            
        self.color_list = color_list
        self.mapped_colors = {}
        self.colors = cycle(self.color_list)
        
    def get_color(self, key):
        if key in self.mapped_colors:
            return self.mapped_colors[key]
        else:
            c =  self.colors.next()
            self.mapped_colors[key] = c
            return c

class CellFateAnalysis(object):
    def __init__(self, ch5_file, mapping_file, events_before_frame=108, onset_frame=0, sites=None, rows=None, cols=None, locations=None):
        # 108 frames = 12h
        self.mcellh5 = cellh5.CH5MappedFile(ch5_file)
        
        #self.mcellh5.read_mapping(mapping_file, rows=('B'), cols=(3,5,8,11))
        self.mcellh5.read_mapping(mapping_file, sites=sites, rows=rows, cols=cols, locations=locations)
        
        self.class_colors = self.mcellh5.class_definition('primary__primary')['color']
        self.class_names = self.mcellh5.class_definition('primary__primary')['name']

        hex_col = list(self.class_colors) 
        
        rgb_col = map(lambda x: hex_to_rgb(x), ['#FFFFFF'] + hex_col)
        
        self.cmap = matplotlib.colors.ListedColormap(rgb_col, 'classification_cmap')
        self.tracks = {}
        
        for _, (w, p) in self.mcellh5.mapping[['Well','Site']].iterrows(): 
            cellh5pos = self.mcellh5.get_position(w,str(p))
            preds = cellh5pos.get_class_prediction()
            self.tracks[(w, p)] = {}
            event_ids = cellh5pos.get_events()
            
            event_ids = [e[onset_frame:] for e in event_ids if cellh5pos.get_time_idx(e[onset_frame]) < events_before_frame]
            
            self.tracks[(w, p)]['ids'] = event_ids
            self.tracks[(w, p)]['labels'] = [cellh5pos.get_class_label(e) for e in event_ids]
            
            print w, p, self.mcellh5.get_treatment_of_pos(w, p)[0]
            
            
    def smooth_and_simplify_tracks(self, in_selector, out_name):   
        print 'Track and hmm predict',
        for w, p in self.tracks:
            print w, p
            cell5pos = self.mcellh5.get_position(w, p)
            class_labels = self.tracks[(w,p)][in_selector]
            class_label_str = map(lambda x : "".join(map(str, x)), class_labels)
            class_label_str = map(lambda x: x.replace('3','2'), class_label_str)
            class_label_str = map(lambda x: x.replace('4','2'), class_label_str)
            class_label_str = map(lambda x: x.replace('5','3'), class_label_str)
            
            #class_label_str = map(lambda x: x.replace('6','1'), class_label_str)
            #class_label_str = map(lambda x: x.replace('7','1'), class_label_str)
            
            
            
            self.tracks[(w,p)][out_name] = class_label_str 
            #print class_label_str
                
       
    def fate_tracking(self, out_name):
        print 'Track',
        for w, p in self.tracks:
            print w, p
            cell5pos = self.mcellh5.get_position(w, p)
            
            class_labels_list = []
            id_list = []
            #hmm_labels_list = []
            for k, e_idx in enumerate(self.tracks[(w,p)]['ids']):   
                start_idx = e_idx[-1]
                track = e_idx + cell5pos.track_first(start_idx)
                class_labels = cell5pos.get_class_label(track)
                #hmm_class_labels = numpy.array(self.hmm.predict(class_labels-1)) + 1
                #print hmm_class_labels
                class_labels_list.append(class_labels)
                id_list.append(track)
                #hmm_labels_list.append(hmm_class_labels)

                
            self.tracks[(w,p)][out_name] = class_labels_list
            self.tracks[(w,p)]['track_ids'] = id_list
            #self.tracks[(w,p)]['hmm_class_labels'] = hmm_labels_list,
        print 'done'
        
    def extract_topro(self):
        topro = []
        for w, p in self.tracks:
            feature_table = self.mcellh5.get_position(w,str(p)).get_object_features('secondary__expanded')
            for t in self.tracks[(w,p)]['track_ids']:
                topro.extend(feature_table[t, 6])  
        pylab.figure()
        pylab.hist(topro, 256)
        pylab.figure()
        pylab.hist(numpy.log2(numpy.array(topro)), 256, log=True)
        pylab.show()
        
    def predict_topro(self, thres):
        for w, p in self.tracks:
            topro = []
            topro_2 = []
            feature_table = self.mcellh5.get_position(w,str(p)).get_object_features('secondary__expanded')
            feature_table = self.mcellh5.get_position(w,str(p)).get_object_features('secondary__expanded')
            for t_ids, t in zip(self.tracks[(w,p)]['track_ids'], self.tracks[(w,p)]['class_label_str']):
                t_topro_pos = feature_table[t_ids, 6] > thres
                t_ = numpy.array(list(t))
                t_[t_topro_pos] = 3
                topro.append(t_)
                
                t__ = numpy.zeros((len(t),), dtype=numpy.uint8)
                t__[t_topro_pos] = 3
                
                topro_2.append(t__)
            self.tracks[(w,p)]['topro_class_labels'] = topro
            self.tracks[(w,p)]['topro_pos'] = topro_2
               
    def predict_hmm(self, class_selector, class_out_name):
        print 'Predict hmm',
        for w, p in self.tracks:
            print w, p
            cell5pos = self.mcellh5.get_position(w, p)
            
            hmm_labels_list = []
            for k, t in enumerate(self.tracks[(w,p)][class_selector]):   
                class_labels = t
                if not isinstance(t, (list,)):
                    class_labels = list(t)
                    class_labels = numpy.array(map(int, t))
                
                hmm_class_labels = self.hmm.predict(numpy.array(class_labels-1)) + 1
                if not 2 in  hmm_class_labels:
                    print hmm_class_labels
                hmm_labels_list.append(hmm_class_labels)

                
            #self.tracks[(w,p)]['class_labels'] = class_labels_list
            self.tracks[(w,p)][class_out_name] = hmm_labels_list
                            
    def event_curves(self, event_selector, 
                           title,
                           region_name,
                           feature_name,
                           with_fate,
                           cmap,
                           time_lapse,
                           event_onset_indicator,
                           xlim,
                           ylim,
                           ):
        pp = PdfPages("%s.pdf" % title)
        
        
        time_unit = 'min'
        if time_lapse is None:
            time_unit = 'frame'
        
        for w, p in sorted(self.tracks):
            cnt = 0
            f = pylab.figure(figsize=(8, 8))
            ax = pylab.gca()
            
            
            feature_table = self.mcellh5.get_position(w,str(p)).get_object_features(region_name)
            feature_idx = self.mcellh5.get_object_feature_idx_by_name(region_name, feature_name)

            id_selector = 'ids'
            if with_fate:
                id_selector = 'track_ids'
            
            all_feature_values = [feature_table[t, feature_idx] for t in self.tracks[(w,p)][id_selector]]
                
            feature_min = numpy.min(map(numpy.min, all_feature_values))
                
            for line, feature_values in zip(self.tracks[(w,p)][event_selector], all_feature_values):
                x_values = numpy.array(map(int,list(line)))
                if numpy.max(feature_values) < 15:
                    print 'excluding event due to low signal'
                    continue
                values = numpy.array(feature_values - feature_min) 
                values = values / numpy.mean(values[(event_onset_indicator-1):(event_onset_indicator+2)])
                #print values[(event_onset_indicator-1):(event_onset_indicator+1)]
                self._plot_curve(x_values[:len(feature_values)], values, cmap, ax, event_onset_indicator, time_lapse)
  

            ax.set_title('%s_%s -- %s'% (w, p, self.mcellh5.get_treatment_of_pos(w, p)[0]) )
            ax.set_xlabel('Time [%s]' % time_unit)
            ax.set_ylabel('Fluorescence (AU)')
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            pylab.savefig(pp, format='pdf')
            pylab.clf()    
        pp.close()
        
    def event_mean_curves(self, event_selector, 
                           title,
                           region_name,
                           feature_name,
                           with_fate,
                           cmap,
                           time_lapse,
                           event_onset_indicator,
                           xlim,
                           ylim,
                           ):
        pp = PdfPages("%s.pdf" % title)
        
        
        time_unit = 'min'
        if time_lapse is None:
            time_unit = 'frame'
        
        for w, p in sorted(self.tracks):
            cnt = 0
            f = pylab.figure(figsize=(12, 8))
            
            
            
            feature_table = self.mcellh5.get_position(w,str(p)).get_object_features(region_name)
            feature_idx = self.mcellh5.get_object_feature_idx_by_name(region_name, feature_name)

            id_selector = 'ids'
            if with_fate:
                id_selector = 'track_ids'
            
            all_feature_values = [feature_table[t, feature_idx] for t in self.tracks[(w,p)][id_selector]]
                
            feature_min = numpy.min(map(numpy.min, all_feature_values))
                
            lines = {}
            lines['Mitotic Exit'] = []
            lines['Apoptosis after Mitosis'] = []
            for line, feature_values in zip(self.tracks[(w,p)][event_selector], all_feature_values):
                x_values = numpy.array(map(int,list(line)))
                if numpy.max(feature_values) < 15:
                    print 'excluding event due to low signal'
                    continue
                values = numpy.array(feature_values - feature_min) 
                values = values / numpy.mean(values[(event_onset_indicator-1):(event_onset_indicator+2)])
                if 17 in x_values:
                    lines['Apoptosis after Mitosis'].append(values)
                else:
                    lines['Mitotic Exit'].append(values)

            from itertools import izip_longest
            
            for n, c in [('Mitotic Exit', 'b'), ('Apoptosis after Mitosis', 'r')]:
                y = []
                yerr = []
                for tmp in izip_longest(*lines[n]):
                    y.append(numpy.array([t for t in tmp if t is not None]).mean())
                    yerr.append(numpy.array([t for t in tmp if t is not None]).std())
                    
                xa = numpy.arange(-4, len(y)-4, 1) * time_lapse
                
                step = 1
                if with_fate:
                    step = 5
                pylab.errorbar(xa[::step], y[::step], yerr=yerr[::step], fmt='o-', color=c, markeredgecolor=c, label=n+" (%d)"%len(lines[n]))

            ax = pylab.gca()
            pylab.legend(loc=1)
            pylab.vlines(0, 0, 1.5, 'k', '--', label='NEBD')
            ax.set_title('%s_%s -- %s'% (w, p, self.mcellh5.get_treatment_of_pos(w, p)[0]) )
            ax.set_xlabel('Time [%s]' % time_unit)
            ax.set_ylabel('Fluorescence (AU)')
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            f.savefig(pp, format='pdf')             
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
             
    def plot_tracks(self, track_selection, cmaps, title='plot_tracks'):
        n = len(track_selection)
        
        pp = PdfPages('%s.pdf' % title)
        
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
                #print (m, n , k + j*n+1)
                ax = pylab.subplot(m, n , k + j*n + 1)
                
                tracks = self.tracks[w,p][class_selector]
                track_lens = map(len, tracks)
                
                if len(track_lens) > 0:
                    
                    max_track_length = max(track_lens)
                
                    max_track_length = max(map(lambda x: len(x), tracks))
                    n_tracks = len(tracks)
                    img = numpy.zeros((n_tracks, max_track_length), dtype=numpy.uint8)
                    
                    for i, t in enumerate(sorted(tracks, cmp=lambda x,y: cmp(len(x), len(y)))):
                        if not isinstance(t, (list,)):
                            b = list(t)
                        img[i,:len(t)] = b
                        

                    ax.matshow(img, cmap=cmap, vmin=0, vmax=cmap.N-1)
                    ax.set_title("%s_%s_%s" % (w,p,class_selector))
                    pylab.axis('off')
                    
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
                ax.plot([old_l_idx, l_idx], [line_height, line_height], color=cmap(int(old_l)), linewidth=3)
                #print [old_l_idx, l_idx],
                old_l = l1
                old_l_idx = l_idx
        #print [old_l_idx, len(line)]
        ax.plot([old_l_idx, len(line)], [line_height, line_height], color=cmap(int(l1)), linewidth=3)
      
    def _plot_curve(self, line, values, cmap, ax, event_onset_indicator=0, time_lapse=None):
        if time_lapse is None:
            time_lapse = 1
            
        print values

        old_l = line[0]
        old_l_idx = 0
        for l_idx, l1 in enumerate(line):
            if l1 != old_l:
                x = (numpy.arange(old_l_idx, l_idx+1) - event_onset_indicator ) * time_lapse
                y = values[old_l_idx:(l_idx+1)]
                print 'x', x[0], 'to', x[-1]
                print 'y', y[0], 'to', y[-1]
                ax.plot(x, y, color=self.cmap(int(old_l)), linewidth=1)
                old_l = l1
                old_l_idx = l_idx
                
        x = (numpy.arange(old_l_idx, len(line)) - event_onset_indicator) * time_lapse
        y = values[old_l_idx:(len(line))]
        ax.plot(x, y, color=self.cmap(int(l1)), linewidth=1)
        
                   
    def plot_proliferaton_timing_histogram(self):
        pp = PdfPages('proliferation_ctrl.pdf')
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
            treatment = self.mcellh5.get_treatment_of_pos(w, p)[0]
            ax.set_title('%s_%s - %s'% (w, p, treatment ))
            ax.set_xlabel('Time [frames]')
            ax.set_ylabel('Mitotic Onset [count]')
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
            ax.text(420, cnt-0.5, label)
            cnt+=2
            return cnt
        
        def _cmp_len_of_first_inter(x,y):
            x_ = "".join(map(lambda x: "%02d" % x, x))
            y_ = "".join(map(lambda x: "%02d" % x, y))
            try:
                x_len = re.search(r"(02|03|04)+", x_).end()
                y_len = re.search(r"(02|03|04)+", y_).end()
                return cmp(y_len, x_len)
            except:
                return cmp(len(y), len(x))

        
        pp = PdfPages("%s.pdf" % title)
        
        for w, p in sorted(self.tracks):
            cnt = 0
            f = pylab.figure(figsize=(8,12))
            ax = pylab.gca()
            
            total_count = 0
            
            for line in sorted( self.tracks[(w,p)]['mito_int'], cmp=_cmp_len_of_first_inter):
                self._plot_line(line, cnt, self.cmap, ax)
                cnt+=1
            total_count += len(self.tracks[(w,p)]['mito_int'])
            cnt = _plot_separator(cnt, str(len(self.tracks[(w,p)]['mito_int'])))
            
            for line in sorted( self.tracks[(w,p)]['mito_int_mito_int_mito'], cmp=_cmp_len_of_first_inter): 
                self._plot_line(line, cnt, self.cmap, ax)
                cnt+=1
            total_count += len(self.tracks[(w,p)]['mito_int_mito_int_mito'])
                
            cnt = _plot_separator(cnt, str(len(self.tracks[(w,p)]['mito_int_mito_int_mito'])))
            
            for line in sorted(self.tracks[(w,p)]['mito_int_mito_int_apo'], cmp=_cmp_len_of_first_inter):
                self._plot_line(line, cnt, self.cmap, ax)
                cnt+=1
            total_count += len(self.tracks[(w,p)]['mito_int_mito_int_apo'])
                
            cnt = _plot_separator(cnt, str(len(self.tracks[(w,p)]['mito_int_mito_int_apo'])))
            
            for line in sorted(self.tracks[(w,p)]['mito_int_apo'], cmp=_cmp_len_of_first_inter):
                self._plot_line(line, cnt, self.cmap, ax)
                cnt+=1
            total_count += len(self.tracks[(w,p)]['mito_int_apo'])
                
            cnt = _plot_separator(cnt, str(len(self.tracks[(w,p)]['mito_int_apo'])))
                
            for line in sorted(self.tracks[(w,p)]['mito_int_mito_apo'], cmp=_cmp_len_of_first_inter):
                self._plot_line(line, cnt, self.cmap, ax)
                cnt+=1
            total_count += len(self.tracks[(w,p)]['mito_int_mito_apo'])
     
            cnt = _plot_separator(cnt, str(len(self.tracks[(w,p)]['mito_int_mito_apo'])))
            
            for line in sorted(self.tracks[(w,p)]['mito_apo'], cmp=_cmp_len_of_first_inter):
                self._plot_line(line, cnt, self.cmap, ax)
                cnt+=1
            total_count += len(self.tracks[(w,p)]['mito_apo'])
     
            cnt = _plot_separator(cnt, str(len(self.tracks[(w,p)]['mito_apo'])))
            
            for line in sorted(self.tracks[(w,p)]['mito_unclassified'], cmp=_cmp_len_of_first_inter):
                self._plot_line(line, cnt, self.cmap, ax)
                cnt+=1
            total_count += len(self.tracks[(w,p)]['mito_unclassified'])
                
            cnt = _plot_separator(cnt, str(total_count))
            
            
                
            ax.invert_yaxis()
            ax.set_xlim(0, 450)
            ax.set_title('%s_%s -- %s'% (w, p, self.mcellh5.get_treatment_of_pos(w, p)[0]) )
            ax.set_xlabel('Time [frame]')
            ax.set_yticklabels([])
            pylab.savefig(pp, format='pdf')
            pylab.clf()
        pp.close()
            
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
                
    def setup_hmm(self, k_classes, constraint_xml):
        constraints = HMMConstraint(constraint_xml)
        
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
        transmat = normalize(transmat, axis=1, eps=0)
        
        est = HMMAgnosticEstimator(k_classes, transmat, numpy.ones((k_classes, 5)), numpy.ones((k_classes, )) )
        est.constrain(constraints)
        self.hmm = hmm.MultinomialHMM(n_components=est.nstates, transmat=transmat, startprob=est.startprob, init_params="")
        self.hmm._set_emissionprob(est.emis)               
                
    def classify_tracks(self, class_selector):
        class named_list(list):
            def __init__(self, name):
                self.name = name
                list.__init__(self)
                
            def append(self, x):
                print x, "added to", self.name
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

            for t_idx, track in enumerate(self.tracks[w,p][class_selector]):
                track_str = "".join(map(lambda x: "%02d" % x,track))
            
                print self.tracks[w,p][class_selector][t_idx]
                print self.tracks[w,p]['Raw class labels'][t_idx]
                
                
                mito_int = self._has_mito_int(track_str)
                if mito_int:
                    self.tracks[(w,p)]['mito_int'].append(mito_int)
                    continue   
                
                mito_apo = self._has_mito_apo(track_str)
                if mito_apo:
                    self.tracks[(w,p)]['mito_apo'].append(mito_apo)
                    continue 
                
                mito_int_mito_apo = self._has_mito_int_mito_apo(track_str)
                if mito_int_mito_apo:
                    self.tracks[(w,p)]['mito_int_mito_apo'].append(mito_int_mito_apo)
                    continue 
                
                mito_int_mito_int_apo = self._has_mito_int_mito_int_apo(track_str)
                if mito_int_mito_int_apo:
                    self.tracks[(w,p)]['mito_int_mito_int_apo'].append(mito_int_mito_int_apo)
                    continue 
                
                mito_int_mito_int_mito = self._has_mito_int_mito_int_mito(track_str)
                if mito_int_mito_int_mito:
                    self.tracks[(w,p)]['mito_int_mito_int_mito'].append(mito_int_mito_int_mito)
                    continue 
                
                mito_int_apo = self._has_mito_int_apo(track_str)
                if mito_int_apo:
                    self.tracks[(w,p)]['mito_int_apo'].append(mito_int_apo)
                    continue 
                
                          
                
                track_ints = map(int, list(split_str_into_len(track_str, 2)))
                self.tracks[(w,p)]['mito_unclassified'].append(track_ints)
             
    def _has_mito_apo(self, track_str):
        MIN_MITOSIS_LEN = 1
        MIN_APO_AFTER_MITOSIS = 1
        MITOSIS_PATTERN = r'^(01)+(02|03|04){%d,}(17){%d}' % (MIN_MITOSIS_LEN, MIN_APO_AFTER_MITOSIS)
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
        MITOSIS_PATTERN = r'^(01)+(?P<blue>(02|03|04){%d,})(05){%d,}(17){%d,}' % (MIN_MITO_LEN, MIN_INTER_LEN, MIN_APO_AFTER_MITOSIS)
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
        MITOSIS_PATTERN = r'^(01)+(?P<blue>(02|03|04)+(05){%d,}(06|07|08){%d,}(09){%d,})((10|11|12)+(13)+)*(17)+' % (MIN_INTER_LEN, MIN_MITOSIS_LEN, MIN_PHASE_AFTER_MITOSIS)
        second_mitosis_re = re.search(MITOSIS_PATTERN, track_str)
        
        if second_mitosis_re is not None:
            end = second_mitosis_re.end() / 2
            end_blue = second_mitosis_re.end('blue') / 2
            track_ints = map(int, list(split_str_into_len(track_str, 2)))
            return [track_ints[i] if i < end_blue else 18 for i in xrange(len(track_ints))][:end]
            # blue, as is, green
        return None
    
    def _has_mito_int_mito_apo(self, track_str):
        MIN_INTER_LEN=1
        MIN_MITOSIS_LEN = 1
        MIN_PHASE_AFTER_MITOSIS = 1
        MITOSIS_PATTERN = r'^(01)+(?P<blue>(02|03|04)+(05){%d,}(06|07|08){%d,})((09)+(10|11|12)+((13)+(14|15|16)+)*)*(17)+' % (MIN_INTER_LEN, MIN_MITOSIS_LEN)
        second_mitosis_re = re.search(MITOSIS_PATTERN, track_str)
        
        if second_mitosis_re is not None:
            end = second_mitosis_re.end() / 2
            end_blue = second_mitosis_re.end('blue') / 2
            track_ints = map(int, list(split_str_into_len(track_str, 2)))
            return [track_ints[i] if i < end_blue else 17 for i in xrange(len(track_ints))][:end]
            # blue, as is, red  
        return None
    
    
    
class CellFateMitoticTiming(CellFateAnalysis):
    def __init__(self, ch5_file, mapping_file, events_before_frame=9999, onset_frame=4, sites=None, rows=None, cols=None, locations=None):
        CellFateAnalysis.__init__(self, ch5_file, mapping_file, events_before_frame=events_before_frame, onset_frame=onset_frame, sites=sites, rows=rows, cols=cols, locations=locations)
    
    def setup_hmm(self, k_classes, constraint_xml):
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
    
    
    def plot_mitotic_timing(self):
        from plot_ext import spreadplot
        import re
        from collections import OrderedDict
        pp = PdfPages('mitotic_timing.pdf')
        f = pylab.figure(figsize=(20,8))
        mito_timing = OrderedDict()
        
        xticklabels=[]
        
        for w, p in sorted(self.tracks.keys()):
            
            mito_timing[(w,p)] = []
            if (w,p) not in self.tracks:
                continue
            for t in self.tracks[(w,p)]['HMM']:
                track_str = "".join(map(str, t))
                mito_re = re.search(r'1+(?P<mito>(2|3|7)+)(1|4|5|6)+', track_str)
                if mito_re is not None:
                    span = mito_re.span('mito')
                    mito_timing[(w,p)].append((span[1] - span[0])*4.7)
                
            treatment = self.mcellh5.get_treatment_of_pos(w, p)[1]
            xticklabels.append("%s_%02d" % (w,p))
        
        from itertools import izip_longest
        import csv

        with open('mito_timing.txt', 'wb') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(xticklabels)
            writer.writerows(izip_longest(*mito_timing.values(), fillvalue=""))   
        
        colors = ['g', 'g',] + ['r']*5 + ['b']*5 + ["#ffa500"]*3 + ['r']*5 + ['b']*5 + ["#ffa500"]*3 + ['k']*50
#         ax = p(mito_timing.values(), xticklabels=xticklabels, colors=colors, spread_type='g', spread=0.1)
        ax = pylab.boxplot(mito_timing.values())
        ax = pylab.gca()
        ax.set_xticklabels(xticklabels, rotation=90)
        ax.set_title('Mitotic Timing')
        ax.set_ylabel('Mitotic timing [min]')
        pylab.tight_layout()
        pylab.savefig(pp, format='pdf')
        
        pp.close()   
        
def fate_mitotic_time():
    locs = [('G', 4), ('G', 5),
            #('B', 5), ('C', 5), ('D', 5), ('E', 5), ('F', 5),
            #('B', 11), ('C', 11), ('D', 11), ('E', 11), ('F', 11),
            ('B', 3), ('C', 3), ('D', 3),
            ]
    
    pm = CellFateMitoticTiming(
                          r"M:\experiments\Experiments_002300\002308\_meta\Analysis\hdf5\_all_positions.ch5",
                          r"M:\experiments\Experiments_002300\002308\_meta\002308.txt",
                          events_before_frame=150,
                          #rows=("B", "C", "D", "E", "F", "G",), 
                          #cols=(3, 4, 5, 11),
                          #rows= ("G",),
                          #cols= (4,5)
                          sites=(1,),
                          #locations=locs
                          )
    pm.fate_tracking('Raw class labels')
    pm.setup_hmm(7, 'graph_7_left2right.xml')
    pm.predict_hmm('Raw class labels', 'HMM')   
    pm.cmap = matplotlib.colors.ListedColormap(map(lambda x: hex_to_rgb(x), 
                                                   ['#FFFFFF', 
                                                    '#00ff00', 
                                                    '#ff8000',
                                                    '#d28dce',
                                                    '#0055ff',
                                                    '#aa5500', 
                                                    '#ff0000',
                                                    '#ffff00']), 'classification_cmap')
    pm.plot_tracks(['Raw class labels', 'HMM'], [pm.cmap, pm.cmap], 'test2')
    pm.plot_mitotic_timing()
            
def fate_mutli():
    pm = CellFateAnalysisMultiHMM(
                          r"M:\experiments\Experiments_002200\002200\_meta\Cecog\Aalysis_with_split\hdf5\_all_positions.ch5",
                          r"M:\experiments\Experiments_002200\002200\_meta\Cecog\Mapping\130710_Mitotic_slippage_and_cell_death.txt",
                        #rows=("D",), 
                        #cols=(2,3,11),
                            rows=None,
                            cols=None,
                          )
    pm.fate_tracking('Raw class labels')
    pm.setup_hmm(17, 'graph_5_multi_states_left2right_17.xml')
    pm.predict_hmm('Raw class labels', 'Multi State HMM')   
    #pm.plot_tracks(['Raw class labels', 'Multi State HMM'], [pm.cmap, cmap17], 'test2')
    
    pm.classify_tracks('Multi State HMM')
    pm.cmap = matplotlib.colors.ListedColormap(map(lambda x: hex_to_rgb(x), 
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
                                                    '#00FF00']), 'classification_cmap')
    
    #pm.plot('Cell Fate Classification Multi HMM', 2)
    pm.event_mean_curves('Multi State HMM',
                            'securin_degradation_short_mean',
                            'tertiary__expanded',
                            'n2_avg',
                            False,
                            pm.cmap,
                            6.5,
                            4,
                            (-20,120),
                            (0,1.5),
                                   )
    
    pm.event_mean_curves('Multi State HMM', 
                    'securin_degradation_with_fate_mean_long',
                    'tertiary__expanded',
                    'n2_avg',
                    True,
                    pm.cmap,
                    6.5,
                    4,
                    (-20,2000),
                    (0,1.5),
                           )
    
    print 'CellFateAnalysisMultiHMM done'
    
def fate():
    pm = CellFateAnalysis(
                          r"M:\experiments\Experiments_002200\002200\_meta\Cecog\Aalysis_with_split\hdf5\_all_positions.ch5",
                          r"M:\experiments\Experiments_002200\002200\_meta\Cecog\Mapping\130710_Mitotic_slippage_and_cell_death.txt",
                         #rows=('B'), 
                         #cols=(3,4,5),
                            rows=None,
                             cols=None,
                          )
    pm.fate_tracking('Raw class labels')
    pm.setup_hmm(5, 'graph_5states_left2right.xml')
    #pm.setup_hmm_multi(13, 'graph_5_multi_states_left2right.xml')
    pm.predict_hmm('Raw class labels', 'Mapped State HMM')
    pm.smooth_and_simplify_tracks('Mapped State HMM', 'Mapped State HMM 3')
    
    pm.plot_tracks(['Raw class labels', 'Mapped State HMM', 'Mapped State HMM 3'], [pm.cmap, pm.cmap, cmap3], 'Mapped_HMM_tracks' )
       
    pm.cmap = matplotlib.colors.ListedColormap(map(lambda x: hex_to_rgb(x), 
                                                   ['#FFFFFF', 
                                                    '#AAAAAA', 
                                                    '#0000FF',
                                                    '#FF0000']), 'classification_cmap')
    pm.classify_tracks('Mapped State HMM 3')
    pm.plot('Cell Fate Classification')
    pm.event_curves('Mapped State HMM 3', 
                    'securin_degradation_short',
                    'tertiary__expanded',
                    'n2_avg',
                    False,
                    pm.cmap,
                    6.66,
                    4,
                    (-20,120),
                    (0,2),
                           )

    pm.event_curves('Mapped State HMM 3', 
                    'securin_degradation_with_fate',
                    'tertiary__expanded',
                    'n2_avg',
                    True,
                    pm.cmap,
                    6.66,
                    3,
                    (-20,2000),
                    (0,2),
                           )
    pm.event_mean_curves('Mapped State HMM 3', 
                    'securin_degradation_short',
                    'tertiary__expanded',
                    'n2_avg',
                    False,
                    pm.cmap,
                    6.66,
                    4,
                    (-20,120),
                    (0,2),
                           )
    
    print 'done'
     
def human_annotation_fate():
    pm = CellFateAnalysis(
                          r"M:\members\Claudia Blaukopf\Experiments\130710_Mitotic_slippage_and_cell_death\_meta\Cecog\Aalysis_with_split\hdf5\_all_positions.ch5",
                          r"M:\members\Claudia Blaukopf\Experiments\130710_Mitotic_slippage_and_cell_death\_meta\Cecog\Mapping\130710_Mitotic_slippage_and_cell_death.txt",
                          rows=None, 
                          cols=None,
                          )
    pm.read_manual_annotations(r"M:\members\Claudia Blaukopf\Experiments\130710_Mitotic_slippage_and_cell_death\_meta\Cecog\Aalysis_with_split\hdf5\annotations_all_positions.ch5",
                               '130710_Mitotic_slippage_and_cell_death')
    
    pm.smooth_and_simplify_tracks('manual_annotations', 'manual_annotations_3')
    
    pm.cmap = matplotlib.colors.ListedColormap(map(lambda x: hex_to_rgb(x), 
                                                   ['#FFFFFF', 
                                                    '#AAAAAA', 
                                                    '#0000FF',
#                                                     '#0000FF',
#                                                     '#0000FF', 
                                                    '#FF0000']), 'classification_cmap')
    pm.classify_tracks('manual_annotations_3')
    pm.plot('manual')

if __name__ == "__main__":
    #fate()
    #fate_mutli()
    fate_mitotic_time()
    print 'done'

