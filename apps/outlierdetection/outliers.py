import numpy
import pylab
import vigra

from sklearn.svm import OneClassSVM
import sklearn.cluster
import sklearn.mixture

import cellh5
import cPickle as pickle
from numpy import recfromcsv
import pandas
import time
from matplotlib.mlab import PCA
from scipy.stats import nanmean
import datetime
import os

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


class OutlierDetection(object):
    classifier_class = OneClassSVM
    def __init__(self, name, mapping_files, cellh5_files, training_sites=(1,), rows=None, cols=None, locations=None, gamma=None, nu=None, pca_dims=None, kernel=None):
        assert gamma != None
        assert pca_dims != None
        assert kernel != None
        assert nu != None
        
        self.name = name
        
        self.mapping_files = mapping_files
        self.cellh5_files = cellh5_files
        
        mappings = []
        for plate_name, mapping_file in mapping_files.items():
            assert plate_name in cellh5_files.keys()
            cellh5_file = cellh5_files[plate_name]
        
            plate_cellh5  = cellh5.CH5MappedFile(cellh5_file)
        
            plate_cellh5.read_mapping(mapping_file, sites=training_sites, rows=rows, cols=cols, locations=locations, plate_name=plate_name)
            plate_mappings = plate_cellh5.mapping
            
            mappings.append(plate_mappings)
            
        self.mapping = pandas.concat(mappings, ignore_index=True)
        del mappings

        self.gamma = gamma
        self.nu = nu
        self.classifier = None
        
        self.pca_dims = pca_dims
        self.kernel = kernel
        plate_id = name
        self.plate_id = name
    
        output_dir = None
        self.set_output_dir(output_dir, plate_id)
        
    def set_output_dir(self, output_dir, plate_id):
        
        self.output_dir = output_dir
        if output_dir is None:
            self.output_dir = plate_id +"/" + datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
            self.output_dir += "-p%d-k%s-n%f-g%f" % (self.pca_dims, self.kernel, self.nu, self.gamma)
            try:
                os.makedirs(self.output_dir)
            except:
        		pass
             
        print "Output Directory: ", self.output_dir
        

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
        
        
        
    def save(self, file_name=None):
        import datetime
        if file_name is None:
            file_name = self.name + "_" + datetime.datetime.now().strftime("%y-%m-%d-%H-%M") + "_g%5.4f_n%5.4f" % (self.gamma, self.nu) + ".pkl"
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)
        return file_name
    
    @staticmethod
    def load(file_name):
        with open(file_name, 'rb') as f:
            obj = pickle.load(f)
        return obj
    
    def train(self, train_on=('neg',)):
        self.feature_set = 'PCA'
        print 'Training OneClass Classifier for', self.feature_set
        training_matrix = self.get_data(train_on, self.feature_set)
        #training_matrix = self.normalize_training_data(training_matrix)
        
        self.train_classifier(training_matrix)
        
        
        
#         testing_matrix, testing_treatment = self.get_data(self.test_on)
#         testing_matrix, testing_treatment = self.normalize_test_data(testing_matrix, testing_treatment)
#         prediction = self.predict(testing_matrix[:,:])
#         self.plot(testing_matrix, prediction)
#         self.group_treatment_outlies(prediction, testing_treatment)
#         
#         self.result = {}
#         self.result[self.test_on] = {}
#         self.result[self.test_on]['matrix'] = testing_matrix
#         self.result[self.test_on]['treatment'] = testing_treatment
#         self.result[self.test_on]['prediction'] = prediction
    
    def predict(self, test_on=('target', 'pos', 'neg')):
        print 'Predicting OneClass Classifier for', self.feature_set
        training_matrix_list = self.mapping[self.mapping['Group'].isin(test_on)][['Well', 'Site', self.feature_set, "Gene Symbol", "siRNA ID"]].iterrows()

        predictions = {}
        distances = {}
        
        log_file_handle = open(self.output('_outlier_detection_log.txt'), 'w')
        
        for idx, (well, site, tm, t1, t2) in training_matrix_list:
            print well, site, t1, t2, "->",
            log_file_handle.write("%s\t%d\t%s\t%s" % (well, site, t1, t2))
            if tm.shape[0] == 0:
                predictions[idx] = numpy.zeros((0, 0))
                distances[idx] = numpy.zeros((0, 0))
            else:
                # tm = self._remove_nan_rows(tm)
                pred, dist = self.predict_with_classifier(tm, log_file_handle)
                predictions[idx] = pred
                distances[idx] = dist
        log_file_handle.close()
        self.mapping['Predictions'] = pandas.Series(predictions)
        self.mapping['Hyperplane distance'] = pandas.Series(distances)
            
    def _remove_nan_rows(self, data):
        data = data[:, self._non_nan_feature_idx]
        if numpy.any(numpy.isnan(data)):
            print 'Warning: Nan values in prediction found. Trying to delete examples:'
            nan_rows = numpy.unique(numpy.where(numpy.isnan(data))[0])
            self._non_nan_sample_idx = [x for x in xrange(data.shape[0]) if x not in nan_rows]
            print 'deleting %d of %d' % (data.shape[0] - len(self._non_nan_sample_idx), data.shape[0])
            
            # get rid of nan samples (still)
            data = data[self._non_nan_sample_idx, :]
        data = (data - self._normalization_means) / self._normalization_stds
        return data
            
    def read_feature(self):
        # init new columns
        self.mapping['Object features'] = 0
        self.mapping['Object count'] = 0
        self.mapping['CellH5 object index'] = 0
        
        # read features from each plate
        for plate_name, cellh5_file in self.cellh5_files.items(): 
            ch5_file = cellh5.CH5File(cellh5_file)
            features = []
            object_counts = []
            c5_object_index = []
            
            for i, row in self.mapping[self.mapping['Plate']==plate_name].iterrows():
                well = row['Well']
                site = str(row['Site'])
                
                ch5_pos = ch5_file.get_position(well, site)
                
                feature_matrix = ch5_pos.get_object_features()
                a = ch5_pos.get_object_table('primary__primary')
                
                time_8_idx = ch5_pos['object']["primary__primary"]['time_idx'] == 7
                
                feature_matrix = feature_matrix[time_8_idx, :]
                
                object_count = len(feature_matrix)
                object_counts.append(object_count)
                
                if object_count > 0:
                    features.append(feature_matrix)
                else:
                    features.append(numpy.zeros((0, features[0].shape[1])))
                c5_object_index.append(numpy.where(time_8_idx)[0])
                
                print 'Reading', plate_name, well, site, len(feature_matrix)
            
            plate_idx = self.mapping['Plate'] == plate_name
            self.mapping.loc[plate_idx, 'Object features'] = features
            self.mapping.loc[plate_idx,'Object count'] = object_counts
            self.mapping.loc[plate_idx,'CellH5 object index'] = c5_object_index
        
    def compute_outlyingness(self):
        def _outlier_count(x):
            res = numpy.float32((x == -1).sum()) / len(x)
            if len(x) == 0:
                print 'a'
            elif numpy.any(numpy.isnan(res)):
                print 'b'
            return res
            
        res = pandas.Series(self.mapping[self.mapping['Group'].isin(('target', 'pos', 'neg'))]['Predictions'].map(_outlier_count))
        self.mapping['Outlyingness'] = res
    
    def train_pca(self, train_on=('neg', 'pos', 'target')):
        training_matrix = self.get_data(train_on)
        training_matrix = self.normalize_training_data(training_matrix)
        print 'Compute PCA', 'found nans in features?', numpy.any(numpy.isnan(training_matrix))
        self.pca = PCA(training_matrix)
        
    def predict_pca(self):
        print 'Project onto PC'
        def _project_on_pca(ma):
            if len(ma) == 0:
                return numpy.NAN
            else:
                ma = self._remove_nan_rows(ma)
                return self.pca.project(ma)[:, :self.pca_dims]
        res = pandas.Series(self.mapping['Object features'].map(_project_on_pca))
        self.mapping['PCA'] = res
    
    def train_classifier(self, training_matrix):
        self.classifier = self.classifier_class(kernel="rbf", nu=self.nu, gamma=self.gamma)
        self.classifier.fit(training_matrix)
#         outlyer_importance = list(numpy.abs(self.classifier.coef_[0]))
#         features = self.mcellh5.object_feature_def()
#         for kk, (a, b) in enumerate([(x,y) for (y,x) in sorted(zip(outlyer_importance,features),reverse=True)]):
#             print "  ", kk, ":", a, b
        
        
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
        
    def _get_mapping_field_of_pos(self, well, pos, field):
        return self.mapping[(self.mapping['Well'] == str(well)) & (self.mapping['Site'] == int(pos))][field].iloc[0]
        
    def get_group_of_pos(self, well, pos):
        return self._get_mapping_field_of_pos(well, pos, 'Group')
    
    def get_treatment_of_pos(self, well, pos, treatment_column=None):
        if treatment_column is None:
            treatment_column = ['siRNA ID', 'Gene Symbol']
        return self._get_mapping_field_of_pos(well, pos, treatment_column)
    
    def get_training_data(self, target='neg'):
        return self.get_data(target)
    
    def get_test_data(self, group):
        data, treatment = self.get_data(group)
        return data, treatment 
        
    def get_data(self, target, type='Object features'):
        print ' get_data for', self.mapping[self.mapping['Group'].isin(target)].reset_index()['siRNA ID']
        return numpy.concatenate(list(self.mapping[self.mapping['Group'].isin(target)].reset_index()[type]))
    
    def normalize_training_data(self, data):
        self._normalization_means = data.mean(axis=0)
        self._normalization_stds = data.std(axis=0)
        
        data = (data - self._normalization_means) / self._normalization_stds
        
        nan_cols = numpy.unique(numpy.where(numpy.isnan(data))[1])
        self._non_nan_feature_idx = [x for x in range(data.shape[1]) if x not in nan_cols]
        data = data[:, self._non_nan_feature_idx]
         
        self._normalization_means = self._normalization_means[self._non_nan_feature_idx]
        self._normalization_stds = self._normalization_stds[self._non_nan_feature_idx]
        print ' normalize: found nans?', numpy.any(numpy.isnan(data))
        
        return data
    
    def normalize_testing_data(self, data):
        return (data - self._normalization_means) / self._normalization_stds
    
#     def normalize_test_data(self, data, treatment_list):
#         # get rid of nan features from training
#         data = data[:, self._non_nan_feature_idx]
#         
#         if numpy.any(numpy.isnan(data)):
#             print 'Warning: Nan values in prediction found. Trying to delete examples:'
#             nan_rows = numpy.unique(numpy.where(numpy.isnan(data))[0])
#             self._non_nan_sample_idx = [x for x in xrange(data.shape[0]) if x not in nan_rows]
#             print 'deleting %d of %d' % (len(self._non_nan_sample_idx), data.shape[0])
#             
#             # get rid of nan samples (still)
#             data = data[self._non_nan_sample_idx, :]
#             treatment_list = treatment_list[self._non_nan_sample_idx]
#         
#         data = (data - self._normalization_means) / self._normalization_stds
#         return data, treatment_list

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
        
    def purge_feature(self):
        del self.mapping['Object features']
        
    def make_heat_map(self):
        rows = sorted(numpy.unique(self.mapping['Row']))
        
        cols = sorted(numpy.unique(self.mapping['Column']))
        
        target_col = 'Outlyingness'
        fig = pylab.figure()
        
        heatmap = numpy.zeros((len(rows), len(cols)), dtype=numpy.float32)
        
        for r_idx, r in enumerate(rows):
            for c_idx, c in enumerate(cols):
                target_value = self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c)][target_col]
                target_count = self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c)]['Object count']
                target_grp = self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c)]['Group']
                
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
        pylab.pcolor(heatmap, cmap=cmap, vmin=0)
        pylab.colorbar()

        for r_idx, r in enumerate(rows):
            for c_idx, c in enumerate(cols):
                try:
                    text_grp = str(self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c)]['Group'].iloc[0])
                    text_gene = str(self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c)]['siRNA ID'].iloc[0])
                    text_gene2 = str(self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c)]['Gene Symbol'].iloc[0])
                    count = self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c)]['Object count'].sum()
                except IndexError:
                    text_grp = "empty"
                    text_gene = "empty"
                    text_gene2 = "empty"
                    count = -1
                    
                t = pylab.text(c_idx + 0.5, r_idx + 0.5, '%s\n%s\n%s' % (text_grp, text_gene, text_gene2), horizontalalignment='center', verticalalignment='center', fontsize=8)
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
             label.set_fontsize(22) 
        
        pylab.tight_layout()
        pylab.savefig(self.output('outlier_heatmap.pdf'))
    
        return heatmap
    
    def make_hit_list(self):
        group_on = 'siRNA ID'
        # group_on = 'Gene Symbol'
        
        group = self.mapping[(self.mapping['Object count'] > 0) & (self.mapping['Group'] == 'target')].groupby([group_on])
        
        neg_group = self.mapping[(self.mapping['Object count'] > 0) & (self.mapping['Group'] == 'neg')].groupby([group_on])
        neg_mean = neg_group.mean()['Outlyingness'].mean()
        
        pos_group = self.mapping[(self.mapping['Object count'] > 0) & (self.mapping['Group'] == 'pos')].groupby([group_on])
        pos_mean = pos_group.mean()['Outlyingness'].mean()        
        
        means = group.mean()['Outlyingness']
        means = means.copy()
        means.sort()
        

        stds = []
        genes = []
        for g, m in means.iteritems():
            print g, group.get_group(g).count()
            std = group.get_group(g).std()['Outlyingness']
            assert m == group .get_group(g).mean()['Outlyingness'] 
            stds.append(std)
            
#             g_ = str(group.get_group(g)['siRNA ID'].iloc[0])
#             genes.append(g + ' ' + g_)
            
            genes.append(g)
            
        ax = pylab.subplot(111)
        ax.errorbar(range(len(means)), means, yerr=stds, fmt='o')
        ax.set_xticks(range(len(means)), minor=False)
        ax.set_xticklabels(genes, rotation=90)
        ax.axhline(means.mean(), label='Target mean')
        ax.axhline(means.mean() + means.std() * 2, color='k', label='Target cutoff at 2 sigma')
        ax.axhline(neg_mean, color='g', label='Negative control mean')
        ax.axhline(pos_mean, color='r', label='Positive control mean')
        
        pylab.legend(loc=2)
        pylab.ylabel('Outlyingness (OC-SVM)')
        pylab.xlabel('Target genes')
        pylab.title('Outliers grouped by gene')
        
        pylab.tight_layout()
        pylab.show()
        
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
        
    def make_outlier_galleries(self):
        for i in range(len(self.mapping['PCA'])):
            prediction = self.mapping['Predictions'][i]
            ge = self.mapping['Gene Symbol'][i]
            si = self.mapping['siRNA ID'][i]
            
            well = str(self.mapping['Well'][i])
            site = str(self.mapping['Site'][i])
            plate_name = str(self.mapping['Plate'][i])
            print 'Exporting gallery matrices for', plate_name, well, site 
            
            ch5_file = cellh5.CH5File(self.cellh5_files[plate_name])
            
            ch5_pos = ch5_file.get_position(well, site)
            
            ch5_index = self.mapping["CellH5 object index"][i][prediction == -1]
            dist =  self.mapping["Hyperplane distance"][i][prediction == -1]
            sorted_ch5_index = zip(*sorted(zip(dist,ch5_index), reverse=True))
            if len(sorted_ch5_index) > 1:
                sorted_ch5_index = sorted_ch5_index[1]
            else:
                sorted_ch5_index = []
                
            img = ch5_pos.get_gallery_image_matrix(sorted_ch5_index, (20, 25))
            vigra.impex.writeImage(img.swapaxes(1,0), self.output('xgal_%s_%s_%s_%s_%s_outlier.png' % (plate_name, well, site, ge, si, )))
            
            ch5_index = self.mapping["CellH5 object index"][i][prediction == 1]
            dist =  self.mapping["Hyperplane distance"][i][prediction == 1]
            sorted_ch5_index = zip(*sorted(zip(dist,ch5_index), reverse=True))
            if len(sorted_ch5_index) > 1:
                sorted_ch5_index = sorted_ch5_index[1]
            else:
                sorted_ch5_index = []
            img = ch5_pos.get_gallery_image_matrix(sorted_ch5_index, (20, 25))
            vigra.impex.writeImage(img.swapaxes(1,0), self.output('xgal_%s_%s_%s_%s_%s_inlier.png' % (plate_name,  well, site, ge, si, )))
             
class OutlierTest(object):
    def __init__(self, name, mapping_file, ch5_file):
        self.name = name
        self.mapping_file = mapping_file 
        self.ch5_file = ch5_file 
        
    def setup(self, rows=None, cols=None, locations=None, gamma=None, nu=None, pca_dims=None, kernel=None):
        self.od = OutlierDetection(self.name,
                              self.mapping_file,
                              self.ch5_file,
                              rows=rows,
                              cols=cols,
                              locations=locations,
                              gamma=gamma,
                              nu=nu,
                              pca_dims=pca_dims,
                              kernel=kernel
                              )
        self.od.read_feature()
        
        self.od.train_pca()
        self.od.predict_pca()
        
        self.od.train()
        self.od.predict()
        
        self.od.compute_outlyingness()
        self.od.purge_feature()
        #self.last_file = self.od.save()
        # prinRt 'Storing to file name', self.last_file
        
    def load_last(self, file=None):
        if file is None:
            import glob
            pkl_list = glob.glob('*.pkl')
            pkl_list.sort()
            file = pkl_list[-1]
        print 'Loading file name', file,
        
        tic = time.time()
        self.od = OutlierDetection.load(file)
        print 'in', time.time() - tic, '[sec]'
        # self.od.__class__ = OutlierDetection
        
        
def sara_screen_analysis_old():
    ot = OutlierTest('testing', 'dc', 'dc')
    ot.load_last('backup/testing_13-07-23-13-24_g0.0050_n0.0500.pkl')
    # ot.load_last()
    # ot.od.plot()
    a = ot.od.make_hit_list()
    a = ot.od.make_heat_map()
    
def matthias_screen_analysis():
    ### Matthias Begin
    if False:
    
        for kernel in ['linear', 'rbf']:
            for nu in [0.05, 0.2]:
                for pca_dims in [2, 20, 100, 239]:
                    for gamma in [0.01, 0.005]:
                        ot = OutlierTest('matthias_outlier_clustering_k_2',
                             'M:/experiments/Experiments_002300/002324/meta/CellCog/mapping/MD9_Grape_over_Time.txt',
                             'M:/experiments/Experiments_002300/002324/meta/CellCog/analysis/hdf5/_all_positions.ch5'
                            )
                        ot.setup(
                                    locations=(
                                               ("A",  8), ("B", 8), ("C", 8), ("D", 8),
                                                 ("H", 6), ("H", 7), #("G", 6), ("G", 7),
                                                 ("H",12), ("H",13), #("G",12), ("G",13),
                                                ),
                                    gamma=gamma,
                                    nu=nu,
                                    pca_dims=pca_dims,
                                    kernel=kernel
                                  )
                        ot.od.cluster_outliers()
    #                     ot.od.make_pca_scatter()
    #                     ot.od.make_heat_map()
    #                     ot.od.make_outlier_galleries()
    else:
        ot = OutlierTest('matthias_od',
                         {'matthias_002324': 'M:/experiments/Experiments_002300/002324/meta/CellCog/mapping/MD9_Grape_over_Time.txt'},
                         {'matthias_002324': 'M:/experiments/Experiments_002300/002324/meta/CellCog/analysis/hdf5/_all_positions.ch5'}
                        ) 
        ot.setup(
                locations=(
                       ("A",  8), ("B", 8), ("C", 8), ("D", 8),
                         ("H", 6), ("H", 7), #("G", 6), ("G", 7),
                         ("H",12), ("H",13), #("G",12), ("G",13),
                        ),
                gamma=2,
                nu=0.2,
                pca_dims=2,
                kernel='rbf'
                )
        ot.od.cluster_outliers()                    
        ot.od.make_pca_scatter()
        ot.od.make_heat_map()
        ot.od.make_outlier_galleries()
### Matthias End
     
     
def sara_screen_analysis():
    ot = OutlierTest('sarax_od',
                     {'SP_9': 'M:/members/SaCl/Adhesion_Screen/6h_noco_timepoint/2013-05-17_SP9_noco01/_meta/MD/SP9.txt'},
                     {'SP_9': 'M:/members/SaCl/Adhesion_Screen/6h_noco_timepoint/2013-05-17_SP9_noco01/_meta/Cellcognition/Analysis1/Analysis1/hdf5/_all_positions.ch5'}
                    ) 
    ot.setup(
            locations=(
                   ("A",  8), ("B", 8), ("C", 8), ("D", 8),
                     ("H", 6), ("H", 7), #("G", 6), ("G", 7),
                     ("H",12), ("H",13), #("G",12), ("G",13),
                    ),
            gamma=2,
            nu=0.2,
            pca_dims=2,
            kernel='rbf'
            )
    ot.od.cluster_outliers()                    
    ot.od.make_pca_scatter()
    ot.od.make_heat_map()
    ot.od.make_outlier_galleries()

if __name__ == "__main__":
    sara_screen_analysis()


    print 'finished'
