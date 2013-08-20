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
from sklearn import hmm
from itertools import cycle
from cecog.util.color import rgb_to_hex
import matplotlib
from hmmpytk import hmm_faster
from matplotlib.numerix import Matrix

from matplotlib.backends.backend_pdf import PdfPages

def matrix_to_dict(matrix):
    d = {}
    for row in range(matrix.shape[0]):
        d[row] = {}
        if len(matrix.shape) == 2:
            for col in range(matrix.shape[1]):
                d[row][col] = numpy.log2(matrix[row, col])
        else:
            d[row] = numpy.log2(matrix[row])
    return d


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
    def __init__(self, ch5_file, mapping_file, events_before_frame=108, onset_frame=0, rows=None, cols=None):
        # 108 frames = 12h
        self.mcellh5 = cellh5.CH5MappedFile(ch5_file)
        
        #self.mcellh5.read_mapping(mapping_file, rows=('B'), cols=(3,5,8,11))
        self.mcellh5.read_mapping(mapping_file, rows=rows, cols=cols)
        
        self.class_colors = self.mcellh5.class_definition('primary__primary')['color']
        self.class_names = self.mcellh5.class_definition('primary__primary')['name']

        hex_col = list(self.class_colors) 
        
        rgb_col = map(lambda x: hex_to_rgb(x), ['#FFFFFF'] + hex_col)
        rgb_col = map(lambda x: hex_to_rgb(x), ['#FFFFFF', '#AAAAAA', '#0000FF','#0000FF','#0000FF', '#FF0000'])
        
        self.cmap = matplotlib.colors.ListedColormap(rgb_col, 'classification_cmap')
        #self.cmap = matplotlib.colors.ListedColormap(rgb_col, 'classification_cmap')
        self.tracks = {}
        self.events = {}
        
        for _, (w, p) in self.mcellh5.mapping[['Well','Site']].iterrows(): 
            cellh5pos = self.mcellh5.get_position(w,str(p))
            preds = cellh5pos.get_class_prediction()
            self.events[(w, p)] = {}
            self.tracks[(w, p)] = {}
            event_ids = cellh5pos.get_events()
            
            event_ids = [e[onset_frame:] for e in event_ids if cellh5pos.get_time_idx(e[0]) < events_before_frame]
            
            self.events[(w, p)]['ids'] = event_ids
            self.events[(w, p)]['labels'] = [cellh5pos.get_class_label(e) for e in event_ids]
            
            #self.events[(w, p)]['labels'] = [preds[e] for e in self.events[(w, p)]['ids']]
            print w, p, self.mcellh5.get_treatment_of_pos(w, p)[0]
            
            
    def smooth_and_simplify_tracks(self, in_selector, out_name):   
        print 'Track and hmm predict',
        for w, p in self.events:
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
        for w, p in self.events:
            print w, p
            cell5pos = self.mcellh5.get_position(w, p)
            
            class_labels_list = []
            id_list = []
            #hmm_labels_list = []
            for k, e_idx in enumerate(self.events[(w,p)]['ids']):   
                start_idx = e_idx[-1]
                track = e_idx[:-1] + cell5pos.track_first(start_idx)
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
        for w, p in self.events:
            print w, p
            cell5pos = self.mcellh5.get_position(w, p)
            
            hmm_labels_list = []
            for k, t in enumerate(self.tracks[(w,p)][class_selector]):   
                class_labels = t
                if not isinstance(t, (list,)):
                    class_labels = list(t)
                    class_labels = numpy.array(map(int, t))
                
                # myhmm
                #hmm_class_labels = numpy.array(myhmm.viterbi(self.hmm, class_labels-1)[0]) + 1
                hmm_class_labels = self.hmm.predict(numpy.array(class_labels-1)) + 1
                #print hmm_class_labels
                hmm_labels_list.append(hmm_class_labels)

                
            #self.tracks[(w,p)]['class_labels'] = class_labels_list
            self.tracks[(w,p)][class_out_name] = hmm_labels_list
                            
    def securin_degradation(self, in_selector, title):
        pp = PdfPages("%s.pdf" % title)
        
        for w, p in self.tracks:
            cnt = 0
            f = pylab.figure(figsize=(8, 8))
            ax = pylab.gca()
            
            securin = []
            feature_table = self.mcellh5.get_position(w,str(p)).get_object_features('tertiary__expanded')
            for t in self.events[(w,p)]['ids']:
                securin.append(feature_table[t, 6])  
                
            securin_min = numpy.min(map(numpy.min, securin))
                
                
            for line, securin_value in zip(self.tracks[(w,p)][in_selector], securin) :
                values = numpy.array(securin_value - securin_min) 
                
                values = values / numpy.mean(values[2:5])
                self._plot_securinline(list(line)[:len(securin_value)], values, self.cmap, ax)
  

            ax.set_title('%s_%s -- %s'% (w, p, self.mcellh5.get_treatment_of_pos(w, p)[0]) )
            ax.set_xlabel('Time [frame]')
            #ax.set_yticklabels([])
            pylab.savefig(pp, format='pdf')
            pylab.clf()

        
        pp.close()
    

    def setup_hmm(self, k_classes, constraint_xml):
        from estimator import HMMConstraint, HMMAgnosticEstimator, normalize
        constraints = HMMConstraint(constraint_xml)
        transmat = numpy.array([
                                [0.8,  0.1,  0.0,  0.0,  0.1],
                                [0.0,  0.8,  0.1,  0.0,  0.1],
                                [0.0,  0.0,  0.8,  0.1,  0.1],
                                [0.1,  0.0,  0.0,  0.8,  0.1],
                                [0.0,  0.0,  0.0,  0.0,  1.0],
                                ])
        transmat = normalize(transmat, eps=0.001)
        
        est = HMMAgnosticEstimator(k_classes, transmat)
        est.constrain(constraints)
        
        self.hmm = hmm.MultinomialHMM(n_components=est.nstates)
        self.hmm._set_startprob(est.startprob)
        self.hmm._set_transmat(est.trans)
        self.hmm._set_emissionprob(est.emis)  
        
    def setup_hmm2(self, k_classes):
        h = hmm_faster.HMM()
        h.set_states(range(k_classes))
        h.set_observations(range(k_classes))
        
        emissionprob = numpy.array(
                              [ 
                                [ 1,  0,  0,],
                                [ 0,  1,  0,],
                                [ 0,  0,  1,],

                               ], dtype=numpy.float64      
                              )
        emissionprob = hmm.normalize(emissionprob, axis=1, eps=10e-60)
        h.set_emission_matrix(matrix_to_dict(emissionprob))
        print emissionprob

#         transmat_ = numpy.array(
#                               [ 
#                                 [ 1,  1,  0,  0,  0,  0,  0,],
#                                 [ 0,  1,  1,  0,  0,  0,  0,],
#                                 [ 0,  0,  1,  1,  0,  0,  0,],
#                                 [ 1,  0,  0,  1,  0,  0,  0,],
#                                 [ 0,  0,  0,  0,  1,  0,  1,],
#                                 [ 0,  0,  0,  0,  1,  1,  0,],
#                                 [ 0,  0,  0,  0,  1,  0,  1,],
#                                ], dtype=numpy.float64      
#                               )
        transmat_ = numpy.array(
                              [ 
                                [ 1,  0,  0,],
                                [ 0,  1,  0,],
                                [ 0,  0,  1,],

                               ], dtype=numpy.float64      
                              )
        transmat_ = hmm.normalize(transmat_, axis=1, eps=10e-60)
        #transmat_[transmat_ < 0.1] = 0
        
        #print 't', transmat_
        h.set_transition_matrix(matrix_to_dict(transmat_))
        print matrix_to_dict(transmat_)
        
        start_prob = numpy.ones((k_classes,), dtype=numpy.float64)
        start_prob[:] = 0
        start_prob[1] = 1
        
        
        start_prob /= start_prob.sum()
        h.set_initial_matrix(matrix_to_dict(start_prob))
        
        self.hmm = h
        print matrix_to_dict(start_prob)
        print 'setup_hmm(): done'
        
    def setup_hmm3(self, k_classes):
        import myhmm
        
        emissionprob = numpy.array(
                              [ 
                                [ 1,  0,  0,],
                                [ 0,  1,  0,],
                                [ 0,  0,  1,],

                               ], dtype=numpy.float64      
                              )
        emissionprob = hmm.normalize(emissionprob, axis=1, eps=10e-4)

        transmat = numpy.array(
                              [ 
                                [ 100,  1,  1,],
                                [ 1,  100,  1,],
                                [ 0,  1,  100,],

                               ], dtype=numpy.float64      
                              ).T
        transmat = hmm.normalize(transmat, axis=1, eps=0)

        
        start_prob = numpy.ones((k_classes,), dtype=numpy.float64)
        start_prob[:] = 0
        start_prob[0] = 1        
        
        start_prob /= start_prob.sum()
        
        h = myhmm.HMM(k_classes, V=range(k_classes), A=transmat, B=emissionprob, Pi=start_prob)
        
        self.hmm = h
        
        print h
        print 'setup_hmm(): done'
    
            
    def plot_tracks(self, class_selector="class_label_str"):
        for w, p in self.tracks:
            cond = self.mcellh5.get_treatment_of_pos(w,p)

            tracks = self.tracks[w,p][class_selector]
            track_lens = map(len, tracks)
            if len(track_lens) > 0:
                pylab.figure()
                max_track_length = max(track_lens)
            
                max_track_length = max(map(lambda x: len(x), tracks))
                n_tracks = len(tracks)
                img = numpy.zeros((n_tracks, max_track_length), dtype=numpy.uint8)
                
                for i, t in enumerate(sorted(tracks, cmp=lambda x,y: cmp(len(x), len(y)))):
                    if not isinstance(t, (list,)):
                        b = list(t)
                    img[i,:len(t)] = b
                    
                 
                pylab.clf()    
                pylab.imshow(img, interpolation='nearest', cmap=self.cmap, clim=(img.min(),img.max()+1))
                pylab.title("%s_%s_%s" % (w,p,class_selector))
                
                #filename = os.path.join(path, '%s_%s_%s.png' % (pos, cond, class_selector))
                #pylab.gcf().savefig(filename)
                #pylab.clf()
            else:
                print w, p, 'Nothing to plot'
          
            
    def classify_tracks(self, class_selector='hmm_class_labels'):
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
                print track_str
                
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
        for l_idx, l1 in enumerate(line[1:]):
            if l1 != old_l:
                ax.plot([old_l_idx, l_idx], [line_height, line_height], color=self.cmap(int(old_l)), linewidth=3)
                old_l = l1
                old_l_idx = l_idx
        ax.plot([old_l_idx+1, len(line)+1], [line_height, line_height], color=self.cmap(int(l1)), linewidth=3)
      
    def _plot_securinline(self, line, values, cmap, ax):
        old_l = line[0]
        old_l_idx = 0
        for l_idx, l1 in enumerate(line[1:]):
            if l1 != old_l:
                ax.plot(range(old_l_idx, l_idx+1), values[old_l_idx: l_idx+1], color=self.cmap(int(old_l)), linewidth=1)
                old_l = l1
                old_l_idx = l_idx
        ax.plot(range(old_l_idx, (len(line))), values[(old_l_idx):(len(line))], color=self.cmap(int(l1)), linewidth=1)
        
                   
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
         
    def plot(self, title):
        def _plot_separator(cnt, label, col='k'):
            cnt+=2
            ax.axhline(cnt, color=col, linewidth=3, linestyle=':')
            ax.text(350, cnt-0.5, label)
            cnt+=2
            return cnt
        
        def _cmp_len_of_first_inter(x,y):
            try:
                x_len =  re.search(r"2+", x).end()
                y_len = re.search(r"2+", y).end()
                return cmp(x_len, y_len)
            except:
                return 0
        
        pp = PdfPages("%s.pdf" % title)
        
        for w, p in self.tracks:
            cnt = 0
            f = pylab.figure(figsize=(8,12))
            ax = pylab.gca()
            
            for line in sorted( self.tracks[(w,p)]['second_mitosis_inter'], cmp=_cmp_len_of_first_inter):
                self._plot_line(list(line), cnt, self.cmap, ax)
                cnt+=1
            cnt = _plot_separator(cnt, 'Second Mitosis -> Interphase')
            
            for line in sorted( self.tracks[(w,p)]['second_mitosis_apo'], cmp=_cmp_len_of_first_inter): 
                self._plot_line(list(line), cnt, self.cmap, ax)
                cnt+=1
                
            cnt = _plot_separator(cnt, 'Second Mitosis -> Apoptotic')
            
            for line in sorted(self.tracks[(w,p)]['death_in_mitosis'], cmp=_cmp_len_of_first_inter):
                self._plot_line(list(line), cnt, self.cmap, ax)
                cnt+=1
                
            cnt = _plot_separator(cnt, 'Death in first Mitosis')
            
            for line in sorted(self.tracks[(w,p)]['death_in_interphase'], cmp=_cmp_len_of_first_inter):
                self._plot_line(list(line), cnt, self.cmap, ax)
                cnt+=1
                
            cnt = _plot_separator(cnt, 'Death in Interphase (after first Mitosis)')
                
            for line in sorted(self.tracks[(w,p)]['no_second_mitosis_no_death'], cmp=_cmp_len_of_first_inter):
                self._plot_line(list(line), cnt, self.cmap, ax)
                cnt+=1
     
            cnt = _plot_separator(cnt, 'No second Mitosis', 'r')
            
            for line in sorted(self.tracks[(w,p)]['unclassified'], cmp=_cmp_len_of_first_inter):
                self._plot_line(list(line), cnt, self.cmap, ax)
                cnt+=1
                #print 'unclassified', line
            cnt = _plot_separator(cnt, '...yet unclassified', )
            
            
                
            ax.invert_yaxis()
            ax.set_xlim(0,700)
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
        for w, p in self.events:
            print 'Read manual annotations', w, p
            cell5pos = self.mcellh5.get_position(w, p)
            
            anno_dset = f['/sample/0/plate/%s/experiment/%s/position/%s/feature/event_annotation/' % (plate_id, w, str(p))]
            self.tracks[(w, p)]['manual_annotations'] = []
            print len(anno_dset)
            for k, (track_id, anno_string) in enumerate(sorted(anno_dset)):
                if len(anno_string) > 0:
                    self.tracks[(w, p)]['manual_annotations'].append(decode(anno_string))
            
            
            
        
            
            
            
            
 
def fate():
  
    pm = CellFateAnalysis(
                          r"M:\members\Claudia Blaukopf\Experiments\130710_Mitotic_slippage_and_cell_death\_meta\Cecog\Aalysis_with_split\hdf5\_all_positions.ch5",
                          r"M:\members\Claudia Blaukopf\Experiments\130710_Mitotic_slippage_and_cell_death\_meta\Cecog\Mapping\130710_Mitotic_slippage_and_cell_death.txt",
                          rows=('B'), 
                          cols=(3,5,8,11),
                          )
    #pm.setup_hmm(7)
    pm.fate_tracking('class_labels')
    #pm.smooth_and_simplify_tracks()
    pm.setup_hmm(5, 'graph_5states_left2right.xml')
    #pm.predict_topro(12)
    pm.predict_hmm('class_labels', 'hmm_class_labels_5')
    pm.smooth_and_simplify_tracks('hmm_class_labels_5', 'hmm_class_labels_3')
    #pm.predict_hmm('topro_class_labels')
    
    #pm.extract_topro()
    pm.classify_tracks('hmm_class_labels_3')
    #pm.plot_tracks('topro_pos')
    #pm.plot_tracks('class_labels')
    #pm.plot_tracks('hmm_class_labels')
    #pm.plot_tracks('topro_class_labels')
#     pm.plot('fate_with_topro_all')
   
    pm.cmap = matplotlib.colors.ListedColormap(map(lambda x: hex_to_rgb(x), 
                                                   ['#FFFFFF', 
                                                    '#AAAAAA', 
                                                    '#0000FF',
#                                                     '#0000FF',
#                                                     '#0000FF', 
                                                    '#FF0000']), 'classification_cmap')
    #pm.plot('HMM class based fate')
    #pm.plot_proliferaton_timing_histogram()
    pm.securin_degradation('hmm_class_labels_3', 'securin_degradation')
    print 'done'
    pylab.show()  
 
#    plot_post_mito("_all_positions.h5")
#    get_single_event_images_of_tree('0022.hdf5', 42)        

def human_annotation_fate():
    pm = CellFateAnalysis(
                          r"M:\members\Claudia Blaukopf\Experiments\130710_Mitotic_slippage_and_cell_death\_meta\Cecog\Aalysis_with_split\hdf5\_all_positions.ch5",
                          r"M:\members\Claudia Blaukopf\Experiments\130710_Mitotic_slippage_and_cell_death\_meta\Cecog\Mapping\130710_Mitotic_slippage_and_cell_death.txt",
                          rows=('B'), 
                          cols=(3,5,11),
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
    human_annotation_fate()

    print 'done'

