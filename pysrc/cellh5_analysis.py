import numpy
import cellh5
import pandas
import os
import collections
import functools
import time
import cProfile
import re
import datetime
import pylab
from collections import OrderedDict
from itertools import izip_longest
from matplotlib.mlab import PCA
from matplotlib.backends.backend_pdf import PdfPages
from cecog.util.color import rgb_to_hex
from sklearn import hmm
from estimator import HMMConstraint, HMMAgnosticEstimator, normalize
hmm.normalize = lambda A, axis=None: normalize(A, axis, eps=10e-99)


                         

class CellH5Analysis(object):
    def __init__(self, name, mapping_files, cellh5_files, output_dir=None, 
                       sites=None, rows=None, cols=None, locations=None):
        self.name = name
        self.mapping_files = mapping_files
        self.cellh5_files = cellh5_files
        self.output_dir = output_dir
        
        # read mappings for all plates
        mappings = []
        for plate_name, mapping_file in mapping_files.items():
            assert plate_name in cellh5_files.keys()
            cellh5_file = cellh5_files[plate_name]
        
            plate_cellh5  = cellh5.CH5MappedFile(cellh5_file)
        
            plate_cellh5.read_mapping(mapping_file, sites=sites, rows=rows, cols=cols, locations=locations, plate_name=plate_name)
            plate_mappings = plate_cellh5.mapping
            
            mappings.append(plate_mappings)
        self.mapping = pandas.concat(mappings, ignore_index=True)
        del mappings

        self.set_output_dir(output_dir)
        
    def set_output_dir(self, output_dir):
        if self.output_dir is None:
            self.output_dir = self.name +"/" + datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
            self.output_dir += "-p%d-k%s-n%f-g%f" % (self.pca_dims, self.kernel, self.nu, self.gamma)
            try:
                os.makedirs(self.output_dir)
            except:
                pass
        print "Output Directory: ", self.output_dir
        

    def output(self, file_):
        file_ = self.str_sanatize(file_)
        return os.path.join(self.output_dir, file_) 
        
    @staticmethod    
    def str_sanatize(input_str):
        input_str = input_str.replace("/","_")
        input_str = input_str.replace("#","_")
        input_str = input_str.replace(")","_")
        input_str = input_str.replace("(","_")
        return input_str
                
    def fate_tracking(self, out_name):
        print 'Tracking cells',
        for w, p in self.tracks:
            print w,
            cell5pos = self.mcellh5.get_position(w, p)
            
            class_labels_list = []
            id_list = []
            for k, e_idx in enumerate(self.tracks[(w,p)]['ids']):   
                start_idx = e_idx[-1]
                track = e_idx + cell5pos.track_first(start_idx)
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
                if numpy.max(feature_values) < 15:
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
                
            treatment = self.mcellh5.get_treatment_of_pos(w, p)[1]
            xticklabels.append("%s_%02d" % (w,p))
        
        from itertools import izip_longest
        import csv

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
    
    
