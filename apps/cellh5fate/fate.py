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
    def __init__(self, ch5_file, mapping_file, events_before_frame=500, onset_frame=0):
        # 108 frames = 12h
        self.mcellh5 = cellh5.CH5MappedFile(ch5_file)
        self.mcellh5.read_mapping(mapping_file, rows=("E"), cols=(3,5,8,11))
        
        self.class_colors = self.mcellh5.class_definition('primary__primary')['color']
        self.class_names = self.mcellh5.class_definition('primary__primary')['name']

        hex_col = list(self.class_colors) 
        #rgb_col = map(lambda x: hex_to_rgb(x), ['#FFFFFF'] + hex_col)
        rgb_col = map(lambda x: hex_to_rgb(x), ['#FFFFFF', '#AAAAAA', '#0000FF', '#FF0000'])
        
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
            
            
    def smooth_and_simplify_tracks(self):   
        print 'Track and hmm predict',
        for w, p in self.events:
            print w, p
            cell5pos = self.mcellh5.get_position(w, p)
            class_labels = self.tracks[(w,p)]['class_labels']
            class_label_str = map(lambda x : "".join(map(str, x)), class_labels)
            class_label_str = map(lambda x: x.replace('3','2'), class_label_str)
            class_label_str = map(lambda x: x.replace('4','2'), class_label_str)
            class_label_str = map(lambda x: x.replace('5','3'), class_label_str)
            
            #class_label_str = map(lambda x: x.replace('6','1'), class_label_str)
            #class_label_str = map(lambda x: x.replace('7','1'), class_label_str)
            
            
            
            self.tracks[(w,p)]["class_label_str"] = class_label_str 
            #print class_label_str
                
               
            
    def fate_tracking(self):
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

                
            self.tracks[(w,p)]['class_labels'] = class_labels_list
            self.tracks[(w,p)]['track_ids'] = id_list
            #self.tracks[(w,p)]['hmm_class_labels'] = hmm_labels_list,

        print 'done'
        
    def predict_hmm(self, class_selector):
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
                
                hmm_class_labels = numpy.array(myhmm.viterbi(self.hmm, class_labels-1)[0]) + 1
                #print hmm_class_labels
                hmm_labels_list.append(hmm_class_labels)

                
            #self.tracks[(w,p)]['class_labels'] = class_labels_list
            self.tracks[(w,p)]['hmm_class_labels'] = hmm_labels_list
                            
    

    def setup_hmm(self, k_classes):
        h = hmm.MultinomialHMM(k_classes, random_state=12)
        emissionprob = numpy.eye(k_classes, k_classes)
        emissionprob = hmm.normalize(emissionprob, axis=1)
        emissionprob[emissionprob<0.1] = 0
        #print 'e', emissionprob
        h.emissionprob_ = emissionprob

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
        transmat_ = hmm.normalize(transmat_, axis=1)
        transmat_[transmat_ < 0.1] = 0
        
        #print 't', transmat_
        h.transmat_ = transmat_
        
        start_prob = numpy.ones((k_classes,))
        start_prob /= start_prob.sum()
        h.startprob_ = start_prob
        
        self.hmm = h
        print 'setup_hmm(): done'
        
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
        if second_mitosis_re is not None:
            start = 0
            end = second_mitosis_re.end()
            return track_str[start:end]
        return None      
    
    def _has_death_in_interphase(self, track_str):
        MIN_INTER_LEN = 1
        MIN_APO_AFTER_MITOSIS = 1
        MITOSIS_PATTERN = r'^1+[1,2]+1{%d,}3{%d}' % (MIN_INTER_LEN, MIN_APO_AFTER_MITOSIS)
        second_mitosis_re = re.search(MITOSIS_PATTERN, track_str)
        if second_mitosis_re is not None:
            start = 0
            end = second_mitosis_re.end()
            return track_str[start:end]
        return None       
                    
    def _has_no_second_mitosis_no_death(self, track_str):
        MIN_MITOSIS_LEN = 3
        MITOSIS_PATTERN = r'^1+2{%d,}1*$' % (MIN_MITOSIS_LEN, )
        no_second_mitosis_re = re.search(MITOSIS_PATTERN, track_str)
        if no_second_mitosis_re is not None:
            start = 0
            end = no_second_mitosis_re.end()
            return track_str[start:end]
        return None 
                    
                
    def _has_second_mitosis(self, track_str, phase_after):
        MIN_PHASE_BEFORE_MITOSIS = 3
        MIN_MITOSIS_LEN = 4
        MIN_PHASE_AFTER_MITOSIS = 1
        #print track_str
        MITOSIS_PATTERN = r'^1+2{%d,}1{%d,}2{%d,}%d{%d}' % (MIN_MITOSIS_LEN, MIN_PHASE_BEFORE_MITOSIS, MIN_MITOSIS_LEN, phase_after, MIN_PHASE_AFTER_MITOSIS)
        second_mitosis_re = re.search(MITOSIS_PATTERN, track_str)
        
        MITOSIS_PATTERN_2 = r'^1+2{%d,}1{%d,}2{%d,}' % (MIN_MITOSIS_LEN, MIN_PHASE_BEFORE_MITOSIS, MIN_MITOSIS_LEN)
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
                ax.plot([old_l_idx, l_idx], [line_height, line_height], color=self.cmap(int(old_l)), linewidth=2)
                old_l = l1
                old_l_idx = l_idx
        ax.plot([old_l_idx, l_idx+1], [line_height, line_height], color=self.cmap(int(l1)), linewidth=2)
      
    
              
                
    def plot_proliferaton_timing_histogram(self):
        for w, p in self.tracks:
            bins = numpy.zeros((21,), dtype=numpy.int32)
            cnt = 0
            f = pylab.figure(figsize=(8,12))
            ax = pylab.gca()
            for t in self.tracks[(w,p)]['track_ids']:
                onset_id = t[4]
                onset_time = int(self.mcellh5.get_position(w,str(p)).get_time_idx(onset_id))
                bins[onset_time/20]+=1
                
            pylab.bar(range(0,420,20), bins, 0.7, color='w')
                
            ax.set_title('%s_%s -- %s'% (w, p, self.mcellh5.get_treatment_of_pos(w, p)[0]))
            ax.set_xlabel('Time [frames]')
            ax.set_ylabel('Mitotic Onset [count]')
            
            
            
            
           
    
            
    def plot(self, title):
        def _plot_separator(cnt, label, col='k'):
            cnt+=2
            ax.axhline(cnt, color=col, linewidth=2, linestyle=':')
            ax.text(350, cnt-0.5, label)
            cnt+=2
            return cnt
        
        def _cmp_len_of_first_inter(x,y):
            x_len =  re.search(r"2+", x).end()
            y_len = re.search(r"2+", y).end()
            return cmp(x_len, y_len)
        
        
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
                print 'unclassified', line
            cnt = _plot_separator(cnt, '...yet unclassified', )
            
            
                
            ax.invert_yaxis()
            ax.set_xlim(0,700)
            ax.set_title('%s_%s -- %s'% (w, p, self.mcellh5.get_treatment_of_pos(w, p)[0]) )
            ax.set_xlabel('Time [frame]')
            ax.set_yticklabels([])
            f.savefig('%s_%s.pdf' % (w,p))
            
    def _plot_motility_1(self, path, pos):
        cond = self.position_dict[pos]
        tracks = self.tracks[pos]
        print pos, cond
        all_dists = []
        all_class = []
        counts = [[], [], []]
        phase_lengths = [[] for _ in range(len(self.class_names))]
        phase_lengths_2 = [[] for _ in range(len(self.class_names))]
        
        post_mitosis_found = []
        post_apoptosis_found = []

        class_selector ='hmm_class_labels'         
        for t in tracks:
            dist_list = []
            class_list = []
            # print "".join(map(str,t['class_labels']))
            counts[0].append(1 if re.search(self.MITOSIS_PATTERN, "".join(map(str,t[class_selector]))) is not None else 0)
            counts[1].append(1 if re.search(self.ENDS_APOPTOSIS, "".join(map(str,t[class_selector]))) is not None else 0)
            
            counts[2].append(0)
            if re.search(self.MITOSIS_PATTERN, "".join(map(str,t[class_selector][30:]))): 
                counts[2][-1] = 1
                #print re.search(self.MITOSIS_PATTERN, "".join(map(str,t[0]))), "".join(map(str,t[0]))
            if re.search(self.ENDS_APOPTOSIS, "".join(map(str,t[class_selector]))): counts[2][-1] = 2  
            
            
            for k in range(len(self.class_names)):
                phase_lengths[k].append(len(t[class_selector][(t[class_selector] == k+1)]))
                if k in [1,2,3,4,5,6]:
                    phase_lengths_2[k].append(len(t[class_selector][(t[class_selector][5:30] == k+1)]))
                else:
                    phase_lengths_2[k].append(len(t[class_selector][(t[class_selector][30:] == k+1)]))
                
            
            if len(t['centers']) > 150:
                for t0, t1 in zip(t['centers'][:-1], t['centers'][1:]):
                    x, yy = t0
                    r, s = t1
                    dist = (x-r)**2 + (yy-s)**2
                    dist_list.append(dist)
                for t0, t1 in zip(t[class_selector][:-1], t[class_selector][1:]):
                    class_list.append(t1)
            
                all_dists.append(dist_list[:150])
                all_class.append(class_list[:150])
                
                
            res = re.search(self.MITOSIS_PATTERN, "".join(map(str,t[class_selector][20:])))
            if res is not None:
                post_mitosis_found.append(res.start()+20)
                
            # ending apoptosis
            res = re.search(self.ENDS_APOPTOSIS, "".join(map(str,t[class_selector][20:])))
            if res is not None:
                post_apoptosis_found.append(res.start()+20)
                    


        post_mitosis_found.sort()
        post_apoptosis_found.sort()
        
        pylab.cla()
        pylab.clf()
        
        total_len = len(post_mitosis_found) + \
                    len(post_apoptosis_found)
        y = 0
        range_min = y / float(total_len)
        range_max = range_min + len(post_mitosis_found)/ float(total_len)
        poly = [[0, range_min], [0, range_max]]
        poly_g = [[0, range_min], [0, range_max]]
        #print range_min, range_max
        for val in post_mitosis_found:
            poly.append([val, range_max - y/float(total_len) + range_min])
            poly.append([val, range_max - (y+1)/float(total_len) + range_min])
            poly_g.append([20, range_max - y/float(total_len) + range_min])
            poly_g.append([20, range_max - (y+1)/float(total_len) + range_min])
            y += 1
        poly.append([0, range_min])
        poly_g.append([0, range_min])
        a = pylab.gca().add_patch(pylab.matplotlib.patches.Polygon(poly, closed=True, fill=True, color='g'))
        pylab.gca().add_patch(pylab.matplotlib.patches.Polygon(poly_g, closed=True, fill=True, color='gray'))
        a.set_label('Mitotic')
        
        range_min = y / float(total_len)
        range_max = range_min + len(post_apoptosis_found)/float(total_len)
        poly = [[0, range_min], [0, range_max]]
        poly_g = [[0, range_min], [0, range_max]]
        #print range_min, range_max
        for val in post_apoptosis_found:
            poly.append([val, range_max - y/float(total_len) + range_min])
            poly.append([val, range_max - (y+1)/float(total_len) + range_min])
            poly_g.append([20, range_max - y/float(total_len) + range_min])
            poly_g.append([20, range_max - (y+1)/float(total_len) + range_min])
            y += 1
        poly.append([0, range_min])
        poly_g.append([0, range_min])
        a = pylab.gca().add_patch(pylab.matplotlib.patches.Polygon(poly, closed=True, fill=True, color='r'))
        pylab.gca().add_patch(pylab.matplotlib.patches.Polygon(poly_g, closed=True, fill=True, color='gray'))
        a.set_label('Apoptotic')
        

        pylab.xlim(0, max(map(len,[t[class_selector] for t in tracks])))
        pylab.ylim(0, 1)
        pylab.ylabel('post mitotic events [proportion]')
        pylab.xlabel('time [frames]')
        #pylab.show()
        pylab.title('%s (%s)' % (cond, pos))
        pylab.legend()
        pylab.gcf().savefig('post_mito_hist_hmm/%s_%s_post_fate.png' % (pos, cond))
        pylab.cla()
        pylab.clf()
              
        img_dist = numpy.sqrt(numpy.array(all_dists))  
        img_class = numpy.array(all_class)
        
        # cell mitility track plots
        pylab.imshow(img_dist, interpolation='nearest')
        pylab.title('%s cell motility' % cond)
        pylab.xlabel('time')
        pylab.gcf().savefig(os.path.join(path, '__%s_%s_dist.png' % (pos, cond)))
        pylab.clf()
        
        pylab.imshow(img_class, interpolation='nearest', cmap=self.cmap)
        pylab.title('%s mitotic phase' % cond)
        pylab.xlabel('time')
        pylab.gcf().savefig(os.path.join(path, '__%s_%s_class.png' % (pos, cond)))
        pylab.clf()
        
        # cell motiltiy per class
        pylab.clf()
        cnt = 0
        sel_classes = ['pro', 'prometa', 'meta', 'earlyana', 'lateana', 'telo']
        for k in range(len(self.class_names)):
            cl_name = self.class_names[k]
            if cl_name not in sel_classes:
                continue
            motil_x = img_dist[img_class == k+1] + numpy.random.rand(len(img_dist[img_class == k+1])) 
            mitil_y = cnt + numpy.random.randn(len(motil_x)) / 8.0
            pylab.plot(motil_x, mitil_y, marker='.', linestyle='.', color=self.class_colors[k], label=self.class_names[k]) 
            cnt += 1
        #pylab.legend(loc=1)
        pylab.gca().set_yticks(range(cnt))
        pylab.gca().set_yticklabels(sel_classes)
        pylab.title('%s cell motility per phase' % cond)
        pylab.xlabel('Distance [px]')
        pylab.gcf().savefig(os.path.join(path, '__%s_%s_dist_class.png' % (pos, cond)))
        pylab.clf()
        
        # phase length vs. fate plot
        
        for k in range(len(self.class_names)):
            cl_name = self.class_names[k]
            cl_color = self.class_colors[k]
            x_fate = []
            
            ph_len = numpy.array(phase_lengths[k])
            for j in range(len(ph_len)):
                if counts[0][j]:
                    x_fate.append(1)
                elif counts[1][j]:
                    x_fate.append(2)
                else:
                    x_fate.append(3)
                
            x_fate = numpy.array(x_fate)    
            col = ['k', 'g', 'r', 'b' ]
            for z in range(1,4):
                ind = (x_fate==z).nonzero()[0]
                pylab.plot(numpy.array(x_fate[ind]) + numpy.random.randn(len(ph_len[ind])) / 10.0, ph_len[ind] *5.5, marker='.', linestyle='.', color=col[z])
                
            pylab.title('Post mitotic fate per class - %s - %s' % (cond, cl_name))
            pylab.gca().set_xticks([1,2,3])
            pylab.gca().set_xticklabels(['mitosis', 'apoptotic', 'interphase'])
            pylab.ylabel('phase duration %s [min]' %cl_name)
            pylab.xlim(0,4) 
            pylab.gcf().savefig(os.path.join('post-mito-hmm', '_%s_%s__%02d_%s.png' % (pos, cond, k+1, cl_name)))
            pylab.clf()
            
        pylab.clf()
        y_motitic_lens = []
        x_inter_lens = []
        xy_colors = []
        for k in range(len(phase_lengths_2[0])):
            y_motitic_lens.append(phase_lengths_2[2][k] +
                             phase_lengths_2[3][k] + phase_lengths_2[4][k]) 
            x_inter_lens.append(phase_lengths_2[0][k])
            xy_colors.append(counts[2][k])
           
        pylab.scatter(numpy.array(x_inter_lens) *5.5, numpy.array(y_motitic_lens) * 5.5, c=xy_colors, cmap=pylab.matplotlib.colors.ListedColormap([(0.5, 0.5, 0.5),(0,1,0), (1,0,0)]), s=50, edgecolors = 'none')     
        pylab.title('Post mitotic fate and timing - %s' % cond)
#            pylab.gca().set_xticks([1,2,3])
#            pylab.gca().set_xticklabels(['mitosis', 'apoptotic', 'interphase'])
        pylab.ylabel('mitotic duration [min]')
        pylab.xlim(-50, 3500) 
        pylab.ylim(0, 120) 
        pylab.xlabel('post-mitotic interphase duration [min]')
        p2 = pylab.Rectangle((0, 0), 1, 1, fc="#00FF00")
        p3 = pylab.Rectangle((0, 0), 1, 1, fc="#FF0000")
        pylab.legend((p2, p3), ('becomes mitotic','becomes apoptotic'))
        
        if os.path.exists('post-mito-timing/_%s.png' % cond):
            pylab.gcf().savefig(os.path.join('post-mito-timing-hmm','_%s_%s_2.png' % (pos, cond)))
        else:
            pylab.gcf().savefig(os.path.join('post-mito-timing-hmm','_%s_%s.png' % (pos, cond)))
        pylab.clf() 
            
            
            
            
 
if __name__ == "__main__":
  
    pm = CellFateAnalysis(
                          r"M:\members\Claudia Blaukopf\Experiments\130710_Mitotic_slippage_and_cell_death\_meta\Cecog\Aalysis_with_split\hdf5\_all_positions.ch5",
                          r"M:\members\Claudia Blaukopf\Experiments\130710_Mitotic_slippage_and_cell_death\_meta\Cecog\Mapping\130710_Mitotic_slippage_and_cell_death.txt",
                          )
    #pm.setup_hmm(7)
    pm.fate_tracking()
    pm.smooth_and_simplify_tracks()
    pm.setup_hmm3(3)
    pm.predict_hmm('class_label_str')
    pm.classify_tracks()
    #pm.plot_tracks('hmm_class_labels')
    #pm.plot_tracks('class_label_str')
    #pm.plot('test')
    pm.plot_proliferaton_timing_histogram()
    print 'done'
    pylab.show()  
 
#    plot_post_mito("_all_positions.h5")
#    get_single_event_images_of_tree('0022.hdf5', 42)        




