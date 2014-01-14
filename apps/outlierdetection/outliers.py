import numpy
import pylab

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


class OutlierDetection(object):
	classifier_class = OneClassSVM
	def __init__(self, name, mapping_file, cellh5_file, training_sites=(1,), rows=None, cols=None, locations=None, gamma=1.0 / 200, nu=0.05):
		self.name = name
		
		self.mapping_file = mapping_file
		self.cellh5_file = cellh5_file
		
		self.mcellh5  = cellh5.CH5MappedFile(cellh5_file)
		
		self.mcellh5.read_mapping(mapping_file, sites=training_sites, rows=rows, cols=cols, locations=locations)
		self.mapping = self.mcellh5.mapping
		
		self.gamma = gamma
		self.nu = nu
		self.classifier = None
		
		self.pca_dims = 2
		
# 	def read_mapping(self, sites=None, rows=None, cols=None):
# 		self.mapping = pandas.read_csv(self.mapping_file, sep='\t')
# 		
# 		if sites is not None:
# 			self.mapping = self.mapping[self.mapping['Site'].isin(sites)]
# 		if rows is not None:
# 			self.mapping = self.mapping[self.mapping['Row'].isin(rows)]
# 		if cols is not None:
# 			self.mapping = self.mapping[self.mapping['Column'].isin(cols)]
# 		
# 		self.mapping.reset_index(inplace=True)
		
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
		
		
		
# 		testing_matrix, testing_treatment = self.get_data(self.test_on)
# 		testing_matrix, testing_treatment = self.normalize_test_data(testing_matrix, testing_treatment)
# 		prediction = self.predict(testing_matrix[:,:])
# 		self.plot(testing_matrix, prediction)
# 		self.group_treatment_outlies(prediction, testing_treatment)
# 		
# 		self.result = {}
# 		self.result[self.test_on] = {}
# 		self.result[self.test_on]['matrix'] = testing_matrix
# 		self.result[self.test_on]['treatment'] = testing_treatment
# 		self.result[self.test_on]['prediction'] = prediction
	
	def predict(self, test_on=('target', 'pos', 'neg')):
		print 'Predicting OneClass Classifier for', self.feature_set
		training_matrix_list = self.mapping[self.mapping['Group'].isin(test_on)][['Well', 'Site', self.feature_set, "Gene Symbol", "siRNA ID"]].iterrows()

		predictions = {}
		distances = {}
		for idx, (well, site, tm, t1, t2) in training_matrix_list:
			print well, site, t1, t2, "->",
			if tm.shape[0] == 0:
				predictions[idx] = numpy.zeros((0, 0))
				distances[idx] = numpy.zeros((0, 0))
			else:
				# tm = self._remove_nan_rows(tm)
				pred, dist = self.predict_with_classifier(tm)
				predictions[idx] = pred
				distances[idx] = dist
			
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
		m = self.mapping
		ch5_file = cellh5.CH5File(self.cellh5_file)
		
		features = []
		object_counts = []
		c5_object_index = []
		for i, row in m.iterrows():
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
			
			print 'Reading', well, site, len(feature_matrix)
		
		m['Object features'] = features
		m['Object count'] = object_counts
		m['CellH5 object index'] = c5_object_index
		
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
# 		outlyer_importance = list(numpy.abs(self.classifier.coef_[0]))
# 		features = self.mcellh5.object_feature_def()
# 		for kk, (a, b) in enumerate([(x,y) for (y,x) in sorted(zip(outlyer_importance,features),reverse=True)]):
# 			print "  ", kk, ":", a, b
		
		
	def predict_with_classifier(self, test_matrix):
		prediction = self.classifier.predict(test_matrix)
		distance = self.classifier.decision_function(test_matrix)[:,0]
		print "%d / %d outliers (%3.2f)" % ((prediction == -1).sum(),
											 len(prediction),
											 (prediction == -1).sum() / float(len(prediction)))
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
		return numpy.concatenate(self.mapping[self.mapping['Group'].isin(target)].reset_index()[type])
	
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
	
# 	def normalize_test_data(self, data, treatment_list):
# 		# get rid of nan features from training
# 		data = data[:, self._non_nan_feature_idx]
# 		
# 		if numpy.any(numpy.isnan(data)):
# 			print 'Warning: Nan values in prediction found. Trying to delete examples:'
# 			nan_rows = numpy.unique(numpy.where(numpy.isnan(data))[0])
# 			self._non_nan_sample_idx = [x for x in xrange(data.shape[0]) if x not in nan_rows]
# 			print 'deleting %d of %d' % (len(self._non_nan_sample_idx), data.shape[0])
# 			
# 			# get rid of nan samples (still)
# 			data = data[self._non_nan_sample_idx, :]
# 			treatment_list = treatment_list[self._non_nan_sample_idx]
# 		
# 		data = (data - self._normalization_means) / self._normalization_stds
# 		return data, treatment_list

	def cluster_get_k(self, training_data):
		max_k = 7
		bics = numpy.zeros(max_k)
		bics[0] = 0
		for k in range(1, max_k):
			gmm = sklearn.mixture.GMM(k)
			gmm.fit(training_data)
			b = gmm.bic(training_data)
			bics[k] = b
		K = numpy.argmin(bics)
		pylab.plot(bics)
		pylab.vlines(K, bics.min(), bics.max())
		pylab.xlabel("Number of clusters (k)")
		pylab.ylabel("Baysian Information Criterion (BIC)")
		pylab.show()
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
		
		print 'Run clustering for training data shape', training_data.shape
		km = sklearn.cluster.KMeans(3)
		km.fit(training_data)
		
		cluster_vectors = {}
		
		print 'Apply Clustering'
		for idx , (data, prediction, g, s)  in self.mapping[['PCA', 'Predictions', 'Gene Symbol', 'siRNA ID']].iterrows():		
			cluster_predict = km.predict(data) + 1
			cluster_predict[prediction==1, :]= 0
			cluster_vectors[idx] = cluster_predict
			
		self.mapping['Outlier clustering'] = pandas.Series(cluster_vectors)

	def plot(self):
		f_x = 1
		f_y = 0
		
		x_min, y_min = 1000000, 100000
		x_max, y_max = -100000, -100000
		
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
			
			ch5_file = cellh5.CH5File(self.cellh5_file)
			
			import vigra
			well = str(self.mapping['Well'][i])
			site = str(self.mapping['Site'][i])
			ch5_pos = ch5_file.get_position(well, site)
			
			img = ch5_pos.get_gallery_image_matrix(numpy.where(prediction == -1)[0], (20, 20))
			vigra.impex.writeImage(img.swapaxes(1,0), '%s_%s_outlier.png' % (well, site))
			
			img = ch5_pos.get_gallery_image_matrix(numpy.where(prediction == 1)[0], (20, 20))
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
		# sites = numpy.unique(self.mapping['Site'])
		# print rows
		# print cols
		
		target_col = 'Outlyingness'
		
		heatmap = numpy.zeros((len(rows), len(cols)), dtype=numpy.float32)
		
		for r_idx, r in enumerate(rows):
			for c_idx, c in enumerate(cols):
				target_value = self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c)][target_col]
				target_count = self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c)]['Object count']
				target_grp = self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c)]['Group']
				
				if target_count.sum() == 0:
					value = -1
				else:
					value = numpy.nansum(target_value * target_count) / float(target_count.sum())
					# value = nanmean(target_value) 
				
				
				if numpy.isnan(value):
					print 'Warning: there are nans...'
				if target_count.sum() > 0:
					heatmap[r_idx, c_idx] = value
# 				else:
# 					heatmap[r_idx, c_idx] = -1
# 					
# 				if target_grp.iloc[0] in ('neg', 'pos'):
# 					heatmap[r_idx, c_idx] = -1
				
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
		pylab.savefig('phenotypic_outlier_heatmap.pdf')
		pylab.show()
	
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
			
# 			g_ = str(group.get_group(g)['siRNA ID'].iloc[0])
# 			genes.append(g + ' ' + g_)
			
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
		KK = 4
			
		pcs = [(x, y) for x in range(2) for y in range(2)]
		legend_there = False
		
		for ii, (f_x, f_y) in enumerate(pcs):
			ax = pylab.subplot(2, 2, ii + 1)
			if f_x == f_y:
				ax.axis('off')
				continue
			
			legend_points = []
			legend_labels = []

			treatment_group = self.mapping.groupby(['Gene Symbol','siRNA ID'])
			
			for tg in treatment_group:
				treatment = "%s %s" % tg[0]
				wells = list(tg[1]['Well'])
				pca_components = numpy.concatenate(list(tg[1]['PCA']))  
				prediction = numpy.concatenate(list(tg[1]['Predictions']))  
		
				x_min, x_max = pca_components[:, f_y].min(), pca_components[:, f_y].max()
				y_min, y_max = pca_components[:, f_x].min(), pca_components[:, f_x].max()
				

			
				if f_y > f_x:	
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
					
				else:
					cluster_vectors = numpy.concatenate(list(tg[1]['Outlier clustering']))
					cluster_colors = {0:'k', 1:'r', 2:'g', 3:'b', 4:'y', 5:'m'}
					for k in range(1, cluster_vectors.max()+1):
						points = ax.scatter(pca_components[cluster_vectors == k, f_y], pca_components[cluster_vectors == k, f_x], c=cluster_colors[k], marker="o", facecolors=cluster_colors[k], zorder=999, edgecolor="none", s=20)
						
					

					
			if not legend_there:
				pylab.figlegend(legend_points, legend_labels, loc = 'lower center', ncol=4, labelspacing=0.1 )
				lengend_there = True
			
			if f_y > f_x:	
				x_min, x_max, y_min, y_max = y_min, y_max, x_min, x_max				
								
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

				pylab.xlim((x_min, x_max))
				pylab.ylim((y_min, y_max))
			else:
				pylab.xlim((x_min, x_max))
				pylab.ylim((y_min, y_max))
			pylab.xticks([])
			pylab.yticks([])
		

		
		pylab.axis('tight')
		pylab.subplots_adjust(wspace=0.01, hspace=0.01)
		pylab.show(block=True)	
		
	def make_outlier_galleries(self):
		for i in range(len(self.mapping['PCA'])):
			prediction = self.mapping['Predictions'][i]
			ch5_file = cellh5.CH5File(self.cellh5_file)
		
			import vigra
			well = str(self.mapping['Well'][i])
			site = str(self.mapping['Site'][i])
			print 'Exporting gallery matrices for', well, site 
			ch5_pos = ch5_file.get_position(well, site)
			
			ch5_index = self.mapping["CellH5 object index"][i][prediction == -1]
			dist =  self.mapping["Hyperplane distance"][i][prediction == -1]
			sorted_ch5_index = zip(*sorted(zip(dist,ch5_index), reverse=True))
			if len(sorted_ch5_index) > 1:
				sorted_ch5_index = sorted_ch5_index[1]
			else:
				sorted_ch5_index = []
				
			img = ch5_pos.get_gallery_image_matrix(sorted_ch5_index, (25, 20))
			vigra.impex.writeImage(img.swapaxes(1,0), '%s_%s_outlier.png' % (well, site))
			
			ch5_index = self.mapping["CellH5 object index"][i][prediction == 1]
			dist =  self.mapping["Hyperplane distance"][i][prediction == 1]
			sorted_ch5_index = zip(*sorted(zip(dist,ch5_index), reverse=True))
			if len(sorted_ch5_index) > 1:
				sorted_ch5_index = sorted_ch5_index[1]
			else:
				sorted_ch5_index = []
			img = ch5_pos.get_gallery_image_matrix(sorted_ch5_index, (50, 20))
			vigra.impex.writeImage(img.swapaxes(1,0), '%s_%s_inlier.png' % (well, site))
		
		
		
class OutlierTest(object):
	def __init__(self, name, mapping_file, ch5_file):
		self.name = name
		self.mapping_file = mapping_file 
		self.ch5_file = ch5_file 
		
	def setup(self, rows=None, cols=None, locations=None):
		self.od = OutlierDetection(self.name,
							  self.mapping_file,
							  self.ch5_file,
							  rows=rows,
							  cols=cols,
							  locations=locations,
							  gamma=0.01,
							  nu=0.2
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
		
		
def sara_screen_analysis():
	ot = OutlierTest('testing', 'dc', 'dc')
	ot.load_last('testing_13-07-23-13-24_g0.0050_n0.0500.pkl')
	# ot.load_last()
	# ot.od.plot()
 	a = ot.od.make_hit_list()
 	a = ot.od.make_heat_map()
 	
if __name__ == "__main__":
	ot = OutlierTest('matthias',
					 'M:/experiments/Experiments_002300/002324/meta/CellCog/mapping/MD9_Grape_over_Time.txt',
					 'M:/experiments/Experiments_002300/002324/meta/CellCog/analysis/hdf5/_all_positions.ch5'
					)
  	ot.setup(
  			 #rows=( "B", 'C', 'D',"E"), 
  			 #cols=(6,7,8,12,13),
    			 locations=(("A",  8), ("B", 8), ("C", 8), ("D", 8),
  						("H", 6), ("H", 7), ("G", 6), ("G", 7),
  						("H",12), ("H",13), ("G",12), ("G",13),)
  			)
	# ot.load_last('matthias_13-12-19-10-43_g0.0167_n0.0800.pkl')
	
	b = ot.od.cluster_outliers()
	
   	b = ot.od.make_pca_scatter()
# 	a = ot.od.make_hit_list()
	a = ot.od.make_heat_map()
   	#b = ot.od.make_outlier_galleries()
   	#b = ot.od.cluster_outliers()

