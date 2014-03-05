import matplotlib
import numpy
import pylab
import vigra

from sklearn.svm import OneClassSVM
import sklearn.cluster
import sklearn.mixture

import cellh5
import cellh5_analysis
import cPickle as pickle
from numpy import recfromcsv
import pandas
import time
from matplotlib.mlab import PCA
from scipy.stats import nanmean
import datetime
import os
from itertools import chain

import iscatter
from cellh5 import CH5File
from collections import defaultdict

def treatmentStackedBar(ax, treatment_dict, color_dict, label_list):
    width=0.33
        
    labels = []
    rects = []
    x = 0 
    for treatment, cluster_vec in treatment_dict.items():
        labels.append(treatment)
        hs = []
        for cluster in range(cluster_vec.max()+1):
            h = len((cluster_vec==cluster).nonzero()[0])
            hs.append(float(h) / len(cluster_vec))
            
        bottom=0
        for c, h in enumerate(hs):
            rect = ax.bar(x, h, width, bottom=bottom, color=color_dict[c])
            bottom+=h
            rects.append(rect)
        x +=1  
          
    rects.append(rect)
    lg = ax.legend(rects, label_list, 
                   loc=3, 
                   ncol=2,
                   #bbox_to_anchor=(-0.1, -0.4)
                   )
    lg.draw_frame(False)
            
    ax.set_xticks(numpy.arange(len(treatment_dict))+width/2.0)
    ax.set_xticklabels(labels, rotation=90)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    #ax.set_xlim(-0.2,len(fates)-0.35)
    
    pylab.xlabel('Treatment')
    pylab.ylabel('Cluster (relative frequency)')
    pylab.tight_layout()


class OutlierDetection(cellh5_analysis.CellH5Analysis):
    classifier_class = OneClassSVM
    def __init__(self, name, mapping_files, cellh5_files, training_sites=(1,2,3,4,), rows=None, cols=None, locations=None, gamma=None, nu=None, pca_dims=None, kernel=None):
        assert gamma != None
        assert pca_dims != None
        assert kernel != None
        assert nu != None
        cellh5_analysis.CellH5Analysis.__init__(self, name, mapping_files, cellh5_files, sites=(1,2,3,4,), rows=rows, cols=cols, locations=locations)
        self.gamma = gamma
        self.nu = nu
        self.pca_dims = pca_dims
        self.kernel = kernel

    def train(self, train_on=('neg',)):
        self.feature_set = 'Object features'
        print 'Training OneClass Classifier for', train_on, 'on', self.feature_set
        training_matrix = self.get_data(train_on, self.feature_set)
        
        if self.feature_set == 'Object features':
            training_matrix = self.normalize_training_data(training_matrix)
        
        self.train_classifier(training_matrix)
        
    def predict(self, test_on=('target', 'pos', 'neg')):
        print 'Predicting OneClass Classifier for', self.feature_set
        testing_matrix_list = self.mapping[self.mapping['Group'].isin(test_on)][['Well', 'Site', self.feature_set, "Gene Symbol", "siRNA ID"]].iterrows()

        predictions = {}
        distances = {}
        
        log_file_handle = open(self.output('_outlier_detection_log.txt'), 'w')
        
        for idx, (well, site, tm, t1, t2) in  testing_matrix_list:
            print well, site, t1, t2, "->",
            log_file_handle.write("%s\t%d\t%s\t%s" % (well, site, t1, t2))
            if isinstance(tm, (float,)) or (tm.shape[0] == 0):
                predictions[idx] = numpy.zeros((0, 0))
                distances[idx] = numpy.zeros((0, 0))
            else:
                if self.feature_set == 'Object features':
                    tm = self._remove_nan_rows(tm)
                pred, dist = self.predict_with_classifier(tm, log_file_handle)
                predictions[idx] = pred
                distances[idx] = dist
        log_file_handle.close()
        self.mapping['Predictions'] = pandas.Series(predictions)
        self.mapping['Hyperplane distance'] = pandas.Series(distances)
            
    def train_classifier(self, training_matrix):
        self.classifier = self.classifier_class(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
        
        if self.kernel == 'linear':
            max_training_samples = 1000
            idx = range(training_matrix.shape[0])
            numpy.random.shuffle(idx)
            self.classifier.fit(training_matrix[idx[:min(max_training_samples, training_matrix.shape[0])],:])
            
            
            f =  self.cellh5_handles.values()[0]
            features = f.object_feature_def()
            if self.feature_set == "PCA":
                coef = numpy.zeros((training_matrix.shape[1],))
                coef[:len(self.classifier.coef_[0])] = self.classifier.coef_[0]
                
                b = numpy.dot(numpy.abs(self.pca.Wt.T), coef) * self.pca.sigma
                
                blub = dict(zip(self._non_nan_feature_idx, b))
                s_blub = sorted(blub, key=blub.get, reverse=True)
                
                
                for k, b in enumerate(s_blub):
                    print k, b, features[b]
            else:
                blub = dict(enumerate(numpy.abs(self.classifier.coef_[0]))) 
                s_blub = sorted(blub, key=blub.get, reverse=True)
                for k, b in enumerate(s_blub):
                    print k, b, features[b]
        else:
            max_training_samples = 10000
            idx = range(training_matrix.shape[0])
            numpy.random.shuffle(idx)
            self.classifier.fit(training_matrix[idx[:min(max_training_samples, training_matrix.shape[0])],:])
            
    def compute_outlyingness(self):
        def _outlier_count(x):
            res = numpy.float32((x == -1).sum()) / len(x)
            return res
            
        res = pandas.Series(self.mapping[self.mapping['Group'].isin(('target', 'pos', 'neg'))]['Predictions'].map(_outlier_count))
        self.mapping['Outlyingness'] = res
        
    def predict_with_classifier(self, test_matrix, log_file_handle=None):
        prediction = self.classifier.predict(test_matrix)
        distance = self.classifier.decision_function(test_matrix)[:,0]
        log = "\t%d / %d outliers\t%3.2f" % ((prediction == -1).sum(),
                                             len(prediction),
                                             (prediction == -1).sum() / float(len(prediction)))
        print log
        if log_file_handle is not None:
            log_file_handle.write(log+"\n")
        return prediction, distance
        

    def cluster_get_k(self, training_data):
        max_k = 7
        bics = numpy.zeros(max_k)
        bics[0] = 0
        for k in range(1, max_k):
            gmm = sklearn.mixture.GMM(k)
            gmm.fit(training_data)
            b = gmm.bic(training_data)
            bics[k] = b
        K = numpy.argmin(bics[1:])
        K+=1
        pylab.plot(range(1, max_k), bics[1:])
        pylab.xlim(0, len(bics)+0.5)
        mi, ma = pylab.gca().get_ylim()
        pylab.vlines(K, mi, ma, color='r', lw=3)
        pylab.xlabel("Number of clusters (k)")
        pylab.ylabel("Baysian Information Criterion (BIC)")
        pylab.savefig(self.output("outlier_clustering_bic.pdf"))
        return K
    
    def cluster_outliers(self):
        # setup clustering
        print 'Setup clustering'
        training_data = []
        for _ , (data, prediction, g, s) in self.mapping[['PCA', 'Predictions', 'Gene Symbol', 'siRNA ID']].iterrows():
            print "  ", g, s
            data_i = data[prediction == -1, :]
            training_data.append(data_i)
            
        training_data = numpy.concatenate(training_data)
        
        k = self.cluster_get_k(training_data)
        
        k = 2
        
        print 'Run clustering for training data shape', training_data.shape
        km = sklearn.cluster.KMeans(k)
        km.fit(training_data)
        
        cluster_vectors = {}
        cluster_teatment_vectors = {}

        print 'Apply Clustering'
        for idx , (data, prediction, g, s)  in self.mapping[['PCA', 'Predictions', 'Gene Symbol', 'siRNA ID']].iterrows():        
            cluster_predict = km.predict(data) + 1
            cluster_predict[prediction==1, :]= 0
            cluster_vectors[idx] = cluster_predict
            cluster_teatment_vectors['%s\n%s' % (g, s)] = cluster_predict
            
        self.mapping['Outlier clustering'] = pandas.Series(cluster_vectors)
        
        # make plot
        
        fig = pylab.figure()
        ax = pylab.gca()
        treatmentStackedBar(ax, cluster_teatment_vectors, {0:'w', 1:'r', 2:'g', 3:'b', 4:'y', 5:'m'}, ['Inlier',] + ["Cluster %d" % d for d in range(1,k+1)])
        pylab.savefig(self.output("outlier_clustering.pdf"))

    def plot(self):
        f_x = 1
        f_y = 0
        
        x_min, y_min = 1000000, 100000
        x_max, y_max = -100000, -100000
        ch5_file = cellh5.CH5File(self.cellh5_file)
        
        print len(self.mapping['PCA'])
        for i in range(len(self.mapping['PCA'])):
            data = self.mapping['PCA'][i]
            prediction = self.mapping['Predictions'][i]
            # print self.mapping['siRNA ID'][i], data.shape
            
            if self.mapping['Group'][i] in ['pos', 'target']:
                pylab.scatter(data[prediction == -1, f_x], data[prediction == -1, f_y], c='red', marker='d', s=42)
                pylab.scatter(data[prediction == 1, f_x], data[prediction == 1, f_y], c='white', marker='d', s=42)
            else:
                pylab.scatter(data[prediction == -1, f_x], data[prediction == -1, f_y], c='white', s=42)
                pylab.scatter(data[prediction == 1, f_x], data[prediction == 1, f_y], c='white', s=42)
            
            x_min_cur, x_max_cur = data[:, f_x].min(), data[:, f_x].max()
            y_min_cur, y_max_cur = data[:, f_y].min(), data[:, f_y].max()
        
            x_min = min(x_min, x_min_cur)
            y_min = min(y_min, y_min_cur)
            x_max = max(x_max, x_max_cur)
            y_max = max(y_max, y_max_cur)
            
            
            
            import vigra
            well = str(self.mapping['Well'][i])
            site = str(self.mapping['Site'][i])
            ch5_pos = ch5_file.get_position(well, site)
            
            img = ch5_pos.get_gallery_image_matrix(numpy.where(prediction == -1)[0], (10, 10))
            vigra.impex.writeImage(img.swapaxes(1,0), '%s_%s_outlier.png' % (well, site))
            
            img = ch5_pos.get_gallery_image_matrix(numpy.where(prediction == 1)[0], (10, 10))
            vigra.impex.writeImage(img.swapaxes(1,0), '%s_%s_inlier.png' % (well, site))

            
        x_min = -12
        y_min = -25
        
        x_max = 42
        y_max = 42    
            
        xx, yy = numpy.meshgrid(numpy.linspace(x_min, x_max, 100), numpy.linspace(y_min, y_max, 100))
        # Z = self.classifier.decision_function(numpy.c_[xx.ravel(), yy.ravel()])
        matrix = numpy.zeros((100 * 100, self.pca_dims))
        matrix[:, f_x] = xx.ravel()
        matrix[:, f_y] = yy.ravel()
        
        
        Z = self.classifier.decision_function(matrix)
        Z = Z.reshape(xx.shape)
        # print Z
        # Z = (Z - Z.min())
        # Z = Z / Z.max()
        # print Z.min(), Z.max()
        # Z = numpy.log(Z+0.001)
        
        pylab.contourf(xx, yy, Z, levels=numpy.linspace(Z.min(), Z.max(), 8), cmap=pylab.matplotlib.cm.Greens, hold='on', alpha=0.5)
        # a = pylab.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
        # pylab.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')
        
        
        pylab.axis('tight')
        
        pylab.xlim((x_min, x_max))
        pylab.ylim((y_min, y_max))
        pylab.axis('off')
        pylab.show(block=True)
        
    def make_heat_map(self):
        print 'Make heat map plot'
        for plate_name in self.cellh5_files.keys():
            rows = sorted(numpy.unique(self.mapping['Row']))
            cols = sorted(numpy.unique(self.mapping['Column']))
            
            target_col = 'Outlyingness'
            fig = pylab.figure(figsize=(len(cols)*0.8 +4, len(rows)*0.6+2))
            
            heatmap = numpy.zeros((len(rows), len(cols)), dtype=numpy.float32)
            
            for r_idx, r in enumerate(rows):
                for c_idx, c in enumerate(cols):
                    target_value = self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c) & (self.mapping['Plate'] == plate_name)][target_col]
                    target_count = self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c) & (self.mapping['Plate'] == plate_name)]['Object count']
                    target_grp =   self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c) & (self.mapping['Plate'] == plate_name)]['Group']
                    
                    if target_count.sum() == 0:
                        value = -1
                    else:
                        value = (target_value * target_count).sum() / float(target_count.sum())
                        # value = nanmean(target_value) 
                    
                    
                    if numpy.isnan(value):
                        print 'Warning: there are nans...'
                    if target_count.sum() > 0:
                        heatmap[r_idx, c_idx] = value
    #                 else:
    #                     heatmap[r_idx, c_idx] = -1
    #                     
    #                 if target_grp.iloc[0] in ('neg', 'pos'):
    #                     heatmap[r_idx, c_idx] = -1
                    
            cmap = pylab.matplotlib.cm.Greens
            cmap.set_under(pylab.matplotlib.cm.Oranges(0))
            # cmap.set_under('w')
                        
            print 'Heatmap', heatmap.max(), heatmap.min()    
            #fig = pylab.figure(figsize=(40,25))
            
            ax = pylab.subplot(111)
            pylab.pcolor(heatmap, cmap=cmap, vmin=0, vmax=1)
            pylab.colorbar()
            ax.set_xlim(0,len(cols))
    
            for r_idx, r in enumerate(rows):
                for c_idx, c in enumerate(cols):
                    try:
                        count = self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c) & (self.mapping['Plate'] == plate_name)]['Object count'].sum()
                        text_grp = str(self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c) & (self.mapping['Plate'] == plate_name)]['Group'].iloc[0]) + " " + ("%0.2f" % heatmap[r_idx, c_idx])[1:]
                        text_gene = str(self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c) & (self.mapping['Plate'] == plate_name)]['siRNA ID'].iloc[0])
                        text_gene2 = str(self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c) & (self.mapping['Plate'] == plate_name)]['Gene Symbol'].iloc[0])
                        
                    except IndexError:
                        text_grp = "empty"
                        text_gene = "empty"
                        text_gene2 = "empty"
                        count = -1
                        
                    t = pylab.text(c_idx + 0.5, r_idx + 0.5, '%s\n%s\n%s\n%d' % (text_grp, text_gene, text_gene2, count), horizontalalignment='center', verticalalignment='center', fontsize=8)
                    if heatmap[r_idx, c_idx] > 0.3:
                        t.set_color('w')
                        
            # put the major ticks at the middle of each cell
            ax.set_xticks(numpy.arange(heatmap.shape[1]) + 0.5, minor=False)
            ax.set_yticks(numpy.arange(heatmap.shape[0]) + 0.5, minor=False)
            
            # want a more natural, table-like display
            # ax.invert_yaxis()
            ax.xaxis.tick_top()
            
            ax.set_xticklabels(list(cols), minor=False)
            ax.set_yticklabels(list(rows), minor=False)
            
            for label in ax.get_xticklabels() + ax.get_yticklabels(): 
                 label.set_fontsize(16) 
            
            #pylab.title("%s %s" % (self.name, plate_name))
            
            pylab.tight_layout()
            pylab.savefig(self.output('outlier_heatmap_%s.pdf' % plate_name))
        
    def make_hit_list_single_feature(self, feature_name):
        print 'Make hit single list plot for', feature_name
        group_on = ['Plate', 'siRNA ID', 'Gene Symbol']
        
        f =  self.cellh5_handles.values()[0]
        feature_idx = f.get_object_feature_idx_by_name('primary__primary', feature_name)
        
        feature_agg_mean = lambda x: numpy.sum([numpy.sum(y[:, feature_idx]) for y in x]) / numpy.sum([len(y[:, feature_idx]) for y in x])
        feature_agg_median = lambda x: numpy.median(list(chain.from_iterable([y[:, feature_idx] for y in x])))
        feature_agg = feature_agg_median
        
        min_object_count_site = 0
        min_object_coutn_group = 15
        # get global values of all plates        
        group = self.mapping[(self.mapping['Object count'] > min_object_count_site) ].groupby(group_on)
        
        overall_min = group['Object features'].apply(feature_agg).min()
        overall_max = group['Object features'].apply(feature_agg).max()
        
        neg_group = self.mapping[(self.mapping['Object count'] > min_object_count_site) & (self.mapping['Group'] == 'neg')].groupby(group_on)
        neg_mean = neg_group['Object features'].apply(feature_agg).mean()
        neg_std = neg_group['Object features'].apply(feature_agg).std()
        
        pos_group = self.mapping[(self.mapping['Object count'] > min_object_count_site) & (self.mapping['Group'] == 'pos')].groupby(group_on)
        pos_mean = pos_group['Object features'].apply(feature_agg).mean()       
        
        #iterate over plates and make hit list figure
        for plate_name in self.mapping_files.keys():
            group = self.mapping[(self.mapping['Object count'] > min_object_count_site) & (self.mapping['Plate'] == plate_name)].groupby(['Well', 'siRNA ID', 'Gene Symbol'])
            
            means = group['Object features'].apply(feature_agg)
            means.sort()
            
            genes = []
            stds = []
            values = []
            for g, m in means.iteritems():
                count = self.mapping['Object count'][group.groups[g]].sum()
                if count > min_object_coutn_group:
                    stds.append(0)
                    values.append(m)
                    genes.append("%s %s %s (%d)" % (g + (count,)))
            n = len(values)
            fig = pylab.figure(figsize=(len(genes)/6 + 3, 10))
            ax = pylab.subplot(111)
            ax.errorbar(range(n), values, yerr=stds, fmt='o', markeredgecolor='none')
            ax.set_xticks(range(n), minor=False)
            ax.set_xticklabels(genes, rotation=90)
            ax.axhline(numpy.mean(values), label='Target mean')
            ax.axhline(numpy.mean(values) + numpy.std(values) * 2, color='k', linestyle='--', label='Target cutoff at +2 sigma')
            ax.axhline(numpy.mean(values) - numpy.std(values) * 2, color='k', linestyle='--', label='Target cutoff at -2 sigma')
            ax.axhline(neg_mean, color='g', label='Negative control mean')
            ax.axhline(neg_mean + neg_std * 2, color='g', linestyle='--', label='Negative control +2 sigma')
            ax.axhline(neg_mean - neg_std * 2, color='g', linestyle='--', label='Negative control -2 sigma')
            #ax.axhline(pos_mean, color='r', label='Positive control mean')
            
            pylab.legend(loc=2)
            pylab.ylabel('Outlyingness (%s)' % feature_name)
            pylab.xlabel('Target genes')
            pylab.title('%s' % plate_name)
            pylab.ylim(overall_min-0.1, overall_max+0.1)
            pylab.xlim(-1, n+1)
            pylab.tight_layout()
            
            pylab.savefig(self.output('%s_hit_list_%s.pdf' % (plate_name, feature_name)))
        #pylab.show()
    
    
    def make_top_hit_list(self, top=100):
        print 'Make hit list plot'
        min_object_coutn_group = 15
        group_on = ['Plate', 'siRNA ID', 'Gene Symbol']

        # get global values of all plates        
        group = self.mapping[(self.mapping['Object count'] > 0) ].groupby(group_on)
        overall_max = group['Outlyingness'].max().max()
        
        neg_group = self.mapping[(self.mapping['Object count'] > 0) & (self.mapping['Group'] == 'neg')].groupby(group_on)
        neg_mean = neg_group.mean()['Outlyingness'].mean()
        
        pos_group = self.mapping[(self.mapping['Object count'] > 0) & (self.mapping['Group'] == 'pos')].groupby(group_on)
        pos_mean = pos_group.mean()['Outlyingness'].mean()        
        
        #iterate over plates and make hit list figure
        label = []
        values = []
        

        group = self.mapping[(self.mapping['Object count'] > 0)].groupby(['Plate', 'Well', 'siRNA ID', 'Gene Symbol'])            
        means = group.apply(lambda x: (x['Outlyingness'] * x['Object count']).sum() / x['Object count'].sum())

        for g, m in means.iteritems():
            print g
            count = self.mapping['Object count'][group.groups[g]].sum()
            if count > min_object_coutn_group:
                values.append(m)
                label.append(g + (count,))
                
                    
        svalues, slabel = zip(*sorted(zip(values, label)))
#         svalues = svalues[-top:]
#         slabel = slabel[-top:]
        
        svalues = svalues[:top]
        slabel = slabel[:top]
        
                
        fig = pylab.figure(figsize=(top/6 + 3, 10))
        ax = pylab.subplot(111)
        ax.errorbar(range(len(svalues)), svalues, fmt='o', markeredgecolor='none')
        ax.set_xticks(range(len(svalues)), minor=False)
        ax.set_xticklabels(["%s %s %s %s (%d)" % g for g in slabel], rotation=90)
        ax.axhline(numpy.mean(values), label='Target mean')
        ax.axhline(numpy.mean(values) + numpy.std(values) * 2, color='k', linestyle='--', label='Target cutoff at +2 sigma')
        ax.axhline(numpy.mean(values) - numpy.std(values) * 2, color='k', linestyle='--', label='Target cutoff at -2 sigma')
        ax.axhline(neg_mean, color='g', label='Negative control mean')
        #ax.axhline(pos_mean, color='r', label='Positive control mean')
        
        pylab.legend(loc=2)
        pylab.ylabel('Outlyingness (OC-SVM)')
        pylab.xlabel('Target genes')
        pylab.ylim(0, overall_max+0.1)
        pylab.xlim(-1, len(svalues)+1)
        pylab.tight_layout()
        pylab.savefig(self.output('top_%d_hit_list.pdf' % top))
        pylab.show()
        
        prefix_lut = {}
        for ii, g in enumerate(reversed(slabel)):
            prefix_lut[(g[0], g[1])] = "%04d" % (ii+1)
        self.make_outlier_galleries(prefix_lut)
            

    def make_hit_list(self):
        print 'Make hit list plot'
        min_object_coutn_group = 15
        group_on = ['Plate', 'siRNA ID', 'Gene Symbol']

        # get global values of all plates        
        group = self.mapping[(self.mapping['Object count'] > 0) ].groupby(group_on)
        overall_max = group['Outlyingness'].max().max()
        
        neg_group = self.mapping[(self.mapping['Object count'] > 0) & (self.mapping['Group'] == 'neg')].groupby(group_on)
        neg_mean = neg_group.mean()['Outlyingness'].mean()
        
        pos_group = self.mapping[(self.mapping['Object count'] > 0) & (self.mapping['Group'] == 'pos')].groupby(group_on)
        pos_mean = pos_group.mean()['Outlyingness'].mean()        
        
        #iterate over plates and make hit list figure
        
        for plate_name in self.mapping_files.keys():
            group = self.mapping[(self.mapping['Object count'] > 0) & (self.mapping['Plate'] == plate_name)].groupby(['Well', 'siRNA ID', 'Gene Symbol'])
            
            means = group.apply(lambda x: (x['Outlyingness'] * x['Object count']).sum() / x['Object count'].sum())
            means.sort()
            
            genes = []
            stds = []
            values = []
            for g, m in means.iteritems():
                count = self.mapping['Object count'][group.groups[g]].sum()
                if count > min_object_coutn_group:
                    stds.append(0)
                    values.append(m)
                    genes.append("%s %s %s (%d)" % (g + (count,)))
                
            fig = pylab.figure(figsize=(len(genes)/6 + 3, 10))
            ax = pylab.subplot(111)
            ax.errorbar(range(len(values)), values, yerr=stds, fmt='o', markeredgecolor='none')
            ax.set_xticks(range(len(values)), minor=False)
            ax.set_xticklabels(genes, rotation=90)
            ax.axhline(numpy.mean(values), label='Target mean')
            ax.axhline(numpy.mean(values) + numpy.std(values) * 2, color='k', linestyle='--', label='Target cutoff at +2 sigma')
            ax.axhline(numpy.mean(values) - numpy.std(values) * 2, color='k', linestyle='--', label='Target cutoff at -2 sigma')
            ax.axhline(neg_mean, color='g', label='Negative control mean')
            #ax.axhline(pos_mean, color='r', label='Positive control mean')
            
            pylab.legend(loc=2)
            pylab.ylabel('Outlyingness (OC-SVM)')
            pylab.xlabel('Target genes')
            pylab.title('%s' % plate_name)
            pylab.ylim(0, overall_max+0.1)
            pylab.xlim(-1, len(values)+1)
            pylab.tight_layout()
            
            pylab.savefig(self.output('%s_hit_list.pdf' % plate_name))
        #pylab.show()
        
        
    def interactive_plot(self, shape=(7, 25)):
        sample_names = []
        data_features = []
        data_pca = []
        cellh5_list = []
        wells = []
        sites = []
        for row_index, row in self.mapping[self.mapping['Object count'] > 0].iterrows():
            features = self.mapping['Object features'].iloc[row_index]
#             pca = row['PCA']
            pca = self.mapping['PCA'].iloc[row_index]
            cellh5idx = numpy.array(row['CellH5 object index'])
            plate = row['Plate']
            well = row['Well']
            site = row['Site']
            sirna = row['siRNA ID']
            gene = row['Gene Symbol']
            
            data_features.append(features)
            data_pca.append(pca)
            for _ in range(features.shape[0]):
                sample_names.append((plate, well, site, sirna, gene))
            cellh5_list.append(cellh5idx)
            
        data_pca = numpy.concatenate(data_pca)
        data_features = numpy.concatenate(data_features)
        data_cellh5 = numpy.concatenate(cellh5_list)
  
        cf = self.cellh5_handles.values()[0]
        feature_names = cf.object_feature_def()



        pca_names = ['PCA %d' % d for d in range(data_pca.shape[1])]
        
        app = iscatter.start_qt_event_loop()
        
        def img_gen(treat, ch5_ids):
            gen = []
            max_count = shape[0] * shape[1]
            for i, (treat, c) in enumerate(zip(treat, ch5_ids)):
                plate = treat[0]
                well = treat[1]
                site = str(treat[2])
                cf = self.cellh5_handles[plate]
                img_gen = cf.gallery_image_matrix_gen(((well, site, (c,)),))
                gen.append(img_gen)
                if i > max_count:
                    break
                
            img_gens = chain.from_iterable(gen)
            
            img = CH5File.gallery_image_matrix_layouter(img_gens, shape)
            return img
  
        iscatter_widget = iscatter.IScatterWidget()
        iscatter_widget.set_data(data_pca, pca_names, sample_names, 0, 1, data_cellh5, img_gen)
    
        
        mw = iscatter.IScatter(iscatter_widget)
        mw.show()
        app.exec_()
        
        
    def make_pca_scatter(self):    
        KK = min(5, self.pca_dims)
            
        pcs = [(x, y) for x in range(KK) for y in range(KK)]
        legend_there = False
        for ii, (f_x, f_y) in enumerate(pcs):
            if f_x >= f_y:
                continue
            
            fig = pylab.figure(figsize=(20, 15))
            legend_points = []
            legend_labels = []

            treatment_group = self.mapping.groupby(['Gene Symbol','siRNA ID'])
            
            x_min, y_min = 1000000, 100000
            x_max, y_max = -100000, -100000

            for tg in treatment_group:
                treatment = "%s %s" % tg[0]
                wells = list(tg[1]['Well'])
                pca_components = numpy.concatenate(list(tg[1]['PCA']))  
                prediction = numpy.concatenate(list(tg[1]['Predictions']))  
        
                x_min_cur, x_max_cur = pca_components[:, f_x].min(), pca_components[:, f_x].max()
                y_min_cur, y_max_cur = pca_components[:, f_y].min(), pca_components[:, f_y].max()
                
                x_min = min(x_min, x_min_cur)
                y_min = min(y_min, y_min_cur)
                x_max = max(x_max, x_max_cur)
                y_max = max(y_max, y_max_cur)

                ax = pylab.subplot(1, 2, 1)
                ax.set_title("Outlier detection PCA=%d vs %d, nu=%f, g=%f (%s)" %(f_x+1, f_y+1, self.nu, self.gamma, self.classifier.kernel))
                    
                if "Taxol" in treatment:
                    color = 'blue'
                    if "No Reversine" in treatment:
                        color = "cyan"
                elif "Noco" in treatment:
                    color = "red"
                    if "No Reversine" in treatment:
                        color = "orange"
                else:
                    assert 'wt control' in treatment
                    color = "green"
                        
                #if color=='green':
                points = ax.scatter(pca_components[prediction == 1, f_x], pca_components[prediction == 1, f_y], c=color, marker="o", facecolors=color, zorder=999, edgecolor="none", s=20)
                legend_points.append(points)
                legend_labels.append("Inlier " + treatment)
                
                points = ax.scatter(pca_components[prediction == -1, f_x], pca_components[prediction == -1, f_y], c=color, marker="o", facecolors='none', zorder=999, edgecolor=color, s=20)
                legend_points.append(points)
                legend_labels.append("Outlier " + treatment)    
                
                ax = pylab.subplot(1, 2, 2)
                cluster_vectors = numpy.concatenate(list(tg[1]['Outlier clustering']))
                ax.set_title("Outlier clustering (%d)" % cluster_vectors.max())
                cluster_colors = {0:'k', 1:'r', 2:'g', 3:'b', 4:'y', 5:'m'}
                for k in range(1, cluster_vectors.max()+1):
                    points = ax.scatter(pca_components[cluster_vectors == k, f_x], pca_components[cluster_vectors == k, f_y], c=cluster_colors[k], marker="o", facecolors=cluster_colors[k], zorder=999, edgecolor="none", s=20)
                        

            
            ax = pylab.subplot(1, 2, 1)    
            xx, yy = numpy.meshgrid(numpy.linspace(x_min, x_max, 100), numpy.linspace(y_min, y_max, 100))
            # Z = self.classifier.decision_function(numpy.c_[xx.ravel(), yy.ravel()])
            matrix = numpy.zeros((100 * 100, self.pca_dims))
            matrix[:, f_x] = xx.ravel()
            matrix[:, f_y] = yy.ravel()
            
            Z = self.classifier.decision_function(matrix)
            Z = Z.reshape(xx.shape)
            
            ax.contourf(xx, yy, Z, levels=numpy.linspace(Z.min(), 0, 17), cmap=pylab.matplotlib.cm.Reds_r, alpha=0.2)
            ax.contour(xx, yy, Z, levels=[0], linewidths=1, colors='k')
            ax.contourf(xx, yy, Z, levels=numpy.linspace(0, Z.max(), 17), cmap=pylab.matplotlib.cm.Greens, alpha=0.3)
            
                    
            if not legend_there:
                pylab.figlegend(legend_points, legend_labels, loc = 'lower center', ncol=4, labelspacing=0.1 )
                lengend_there = True
            
            ax = pylab.subplot(1, 2, 1)
            ax.set_xlim((x_min, x_max))
            ax.set_ylim((y_min, y_max))    
            pylab.xticks([])
            pylab.yticks([])
            
            ax = pylab.subplot(1, 2, 2)
            ax.set_xlim((x_min, x_max))
            ax.set_ylim((y_min, y_max))    
            pylab.xticks([])
            pylab.yticks([])
            
            
            pylab.subplots_adjust(wspace=0.05, hspace=0.05)
            pylab.tight_layout()
            pylab.savefig(self.output("outlier_detection_pca_%d_vs_%d.pdf" %(f_x+1, f_y+1)))   
        
    def make_outlier_galleries(self, prefix_lut=None):  
        group = self.mapping[(self.mapping['Object count'] > 0)].groupby(('Plate', 'Well'))
        plates_inlier = defaultdict(list)
        plates_outlier = defaultdict(list)
        plates_info  = defaultdict(list)
         
        for grp, grp_values in group:
            index_tpl_in = []
            index_tpl_out = []
            
            if prefix_lut is None:
                pass
            elif not  (grp_values['Plate'].unique()[0],  grp_values['Well'].unique()[0]) in prefix_lut:
                continue

            for _, (plate_name, well, site, ge, si, prediction, hyper, ch5_ind) in grp_values[['Plate', 'Well', 'Site', 'Gene Symbol', 'siRNA ID', 'Predictions', "Hyperplane distance", "CellH5 object index"]].iterrows():
                in_ch5_index = ch5_ind[prediction == -1]
                in_dist =  hyper[prediction == -1]
                in_sorted_ch5_index = zip(*sorted(zip(in_dist, in_ch5_index), reverse=True))
                if len(in_sorted_ch5_index) > 1:
                    in_sorted_ch5_index = in_sorted_ch5_index[1]
                else:
                    in_sorted_ch5_index = []
                    
                out_ch5_index = ch5_ind[prediction == 1]
                out_dist =  hyper[prediction == 1]
                out_sorted_ch5_index = zip(*sorted(zip(out_dist, out_ch5_index), reverse=True))
                if len(out_sorted_ch5_index) > 1:
                    out_sorted_ch5_index = out_sorted_ch5_index[1]
                else:
                    out_sorted_ch5_index = []
                    
                index_tpl_in.append((well, site, in_sorted_ch5_index))
                index_tpl_out.append((well, site, out_sorted_ch5_index))
                    
            plates_inlier[plate_name].append(index_tpl_in)
            plates_outlier[plate_name].append(index_tpl_out)
            plates_info[plate_name].append((plate_name, well, ge, si))
           
        for plate_name in plates_inlier.keys():
            pl_in_index_tpl = plates_inlier[plate_name]
            pl_on_index_tpl = plates_outlier[plate_name]
            pl_info = plates_info[plate_name]
            
            for index_tpl_in, index_tpl_out, info in zip(pl_in_index_tpl, pl_on_index_tpl, pl_info): 
                cf = self.cellh5_handles[plate_name]
                outlier_img = cf.get_gallery_image_matrix(index_tpl_in, (15, 10)).swapaxes(1,0)
                inlier_img = cf.get_gallery_image_matrix(index_tpl_out, (15, 10)).swapaxes(1,0)
                
                img = numpy.concatenate((inlier_img, numpy.ones((5, inlier_img.shape[1]))*255, outlier_img))
                assert plate_name == info[0]
                if prefix_lut is None:
                    img_name = 'xgal_%s_%s_%s_%s.png' % info
                else:
                    img_name = '%s_%s_%s_%s_%s.png' % ((prefix_lut[info[0], info[1]],) + info)
                vigra.impex.writeImage(img, self.output(img_name))
                print 'Exporting gallery matrix image for', info

                
class SaraOutlier(object):
    @staticmethod
    def sara_mitotic_live_selector(pos, plate_name, treatment, outdir):
        pp = pos['object']["primary__primary"]
        
        fret_index = pos.definitions.get_object_feature_idx_by_name('secondary__inside', 'n2_avg')
        topro_ind = pos.definitions.get_object_feature_idx_by_name('quartiary__inside', 'n2_avg')
        yfp_ind = pos.definitions.get_object_feature_idx_by_name('tertiary__inside', 'n2_avg')
        
        fret_inside = pos.get_object_features('secondary__inside')[:, fret_index]
        fret_outside = pos.get_object_features('secondary__outside')[:, fret_index]
        
        yfp_inside = pos.get_object_features('tertiary__inside')[:, yfp_ind]
        yfp_outside = pos.get_object_features('tertiary__outside')[:, yfp_ind]
        
        topro_inside = pos.get_object_features('quartiary__inside')[:, topro_ind]
        topro_outside = pos.get_object_features('quartiary__outside')[:, topro_ind]
        
        try:
            fret_ratio = (fret_inside - fret_outside) / (yfp_inside - yfp_outside)
            
            topro_diff = topro_inside - topro_outside
            
            fret_min = 0.6
            fret_max = 0.82
            idx_1 = numpy.logical_and(fret_ratio > fret_min, fret_ratio < fret_max )
            
            topro_abs_max = 15
            idx_2 = topro_diff < topro_abs_max
            
            idx = numpy.logical_and(idx_1, idx_2)
        except:
            print "@!#$!"*100
            idx = numpy.zeros((len(pp),), dtype=numpy.bool)
        
        print "  %s_%s" % (pos.well, pos.pos), "%d/%d" % (idx.sum(), len(idx)),  'are live mitotic'
        
        # Export images for live mitotic cells
        if False:
            well = pos.well
            site = pos.pos

            ch5_index_mitotic_live = numpy.nonzero(idx)[0]
            ch5_index_not_mitotic_live = numpy.nonzero(numpy.logical_not(idx))[0]
                
            outlier_img = pos.get_gallery_image_matrix(ch5_index_not_mitotic_live, (10, 5)).swapaxes(1,0)
            inlier_img = pos.get_gallery_image_matrix(ch5_index_mitotic_live, (10, 5)).swapaxes(1,0)
            
            img = numpy.concatenate((inlier_img, numpy.ones((5, inlier_img.shape[1]))*255, outlier_img))
            vigra.impex.writeImage(img, os.path.join(outdir, 'mito_live_%s_%s_%s_%s.png' % (plate_name,  well, site, treatment )))
        return idx
    
    def __init__(self, name, mapping_files, ch5_files, rows=None, cols=None, locations=None, gamma=None, nu=None, pca_dims=None, kernel=None):
        self.od = OutlierDetection(name,
                                  mapping_files,
                                  ch5_files,
                                  rows=rows,
                                  cols=cols,
                                  locations=locations,
                                  gamma=gamma,
                                  nu=nu,
                                  pca_dims=pca_dims,
                                  kernel=kernel
                                  )
        self.od.set_read_feature_time_predicate(numpy.equal, 0)
        self.od.read_feature(self.sara_mitotic_live_selector)
        self.od.train_pca()
        self.od.predict_pca()
        self.od.train()
        self.od.predict()
        self.od.compute_outlyingness()
        
        #self.od.cluster_outliers()                    
        #self.od.interactive_plot()
        
       
        
        self.od.make_top_hit_list()
        #self.od.make_hit_list_single_feature('roisize')
        self.od.make_heat_map()
        #self.od.make_outlier_galleries()
        print 'Results:', self.od.output_dir
        os.startfile(os.path.join(os.getcwd(), self.od.output_dir))
      
class MatthiasOutlier(object):
    def __init__(self, name, mapping_files, ch5_files, rows=None, cols=None, locations=None, gamma=None, nu=None, pca_dims=None, kernel=None):
        if True:
            self.od = self._init(name, mapping_files, ch5_files, rows=rows, cols=cols, locations=locations, gamma=gamma, nu=nu, pca_dims=pca_dims, kernel=kernel)
        else:
            for kernel in ['linear', 'rbf']:
                for nu in [0.05, 0.2]:
                    for pca_dims in [2, 20, 100, 239]:
                        for gamma in [0.01, 0.005]:
                            self.od = self._init(name, mapping_files, ch5_files, rows=rows, cols=cols, locations=locations, gamma=gamma, nu=gamma, pca_dims=pca_dims, kernel=kernel)
            
        
        
        
        
    def _init(self, name, mapping_files, ch5_files, rows, cols, locations, gamma, nu, pca_dims, kernel):
        self.od = OutlierDetection(name,
                                  mapping_files,
                                  ch5_files,
                                  rows=rows,
                                  cols=cols,
                                  locations=locations,
                                  gamma=gamma,
                                  nu=nu,
                                  pca_dims=pca_dims,
                                  kernel=kernel
                                  )
        self.od.set_read_feature_time_predicate(numpy.greater, 7)
        self.od.read_feature()
        self.od.train_pca()
        self.od.predict_pca()
        self.od.train()
        self.od.predict()
        self.od.compute_outlyingness()
        
        #self.od.cluster_outliers()                    
        #self.od.make_pca_scatter()
        
        self.od.make_heat_map()
        self.od.make_hit_list()
        self.od.make_outlier_galleries()
        print 'Results:', self.od.output_dir
        os.startfile(os.path.join(os.getcwd(), self.od.output_dir))

    
    
def run_exp(name, lookup, analysis_class):
    ac = analysis_class(name, **lookup[name])
    

if __name__ == "__main__":
    EXPLOOKUP = {'sarax_od':
                     {
                      'mapping_files' : {
                        'SP_9': 'F:/sara_adhesion_screen/sp9.txt',
                        'SP_8': 'F:/sara_adhesion_screen/sp8.txt',
                        'SP_7': 'F:/sara_adhesion_screen/sp7.txt',
                        'SP_6': 'F:/sara_adhesion_screen/sp6.txt',
                        'SP_5': 'F:/sara_adhesion_screen/sp5.txt',
                        'SP_4': 'F:/sara_adhesion_screen/sp4.txt',
                        'SP_3': 'F:/sara_adhesion_screen/sp3.txt',
                        'SP_2': 'F:/sara_adhesion_screen/sp2.txt',
                        'SP_1': 'F:/sara_adhesion_screen/sp1.txt',
                                        },
                      'ch5_files' : {
                            'SP_9': 'F:/sara_adhesion_screen/sp9__all_positions_with_data_combined.ch5',
                            'SP_8': 'F:/sara_adhesion_screen/sp8__all_positions_with_data_combined.ch5',
                            'SP_7': 'F:/sara_adhesion_screen/sp7__all_positions_with_data_combined.ch5',
                            'SP_6': 'F:/sara_adhesion_screen/sp6__all_positions_with_data_combined.ch5',
                            'SP_5': 'F:/sara_adhesion_screen/sp5__all_positions_with_data_combined.ch5',
                            'SP_4': 'F:/sara_adhesion_screen/sp4__all_positions_with_data_combined.ch5',
                            'SP_3': 'F:/sara_adhesion_screen/sp3__all_positions_with_data_combined.ch5',
                            'SP_2': 'F:/sara_adhesion_screen/sp2__all_positions_with_data_combined.ch5',
                            'SP_1': 'F:/sara_adhesion_screen/sp1__all_positions_with_data_combined.ch5',
                                        },
#                     'locations' : (
#                           ("A",  1), ("B", 8), ("C", 8), ("D", 8),
#                           ("H", 6), ("H", 7), ("G", 6), ("G", 7),
#                           ("H",12), ("H",13), ("G",12), ("G",13),
#                       ),
                      'rows' : list("ABCDEFGHIJKLMNOP")[:],
                      'cols' : tuple(range(1,24)),
                      'gamma' : 0.005,
                      'nu' : 0.10,
                      'pca_dims' : 239,
                      'kernel' :'rbf'
                     },
                 'matthias_od':
                     {
                      'mapping_files' : {
                            '002324': 'M:/experiments/Experiments_002300/002324/meta/CellCog/mapping/MD9_Grape_over_Time.txt',
                                       },
                      'ch5_files' : {
                            '002324': 'M:/experiments/Experiments_002300/002324/meta/CellCog/analysis/hdf5/_all_positions.ch5',
                                        },
                      'locations' : (
                              ("A",  8), ("B", 8), ("C", 8), ("D", 8),
                              ("H", 6), ("H", 7), ("G", 6), ("G", 7),
                              ("H",12), ("H",13), ("G",12), ("G",13),
                                    ),
#                       'rows' : list("ABCDEFGHIJKLMNOP")[:3],
#                       'cols' : tuple(range(1,3)),
                      'gamma' : 0.0005,
                      'nu' : 0.15,
                      'pca_dims' : 239,
                      'kernel' :'rbf'
                     }
                  }
    
    run_exp('sarax_od', EXPLOOKUP, SaraOutlier)
    #run_exp('matthias_od', EXPLOOKUP, MatthiasOutlier)


    print 'finished'
