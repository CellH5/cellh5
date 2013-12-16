import numpy
import pylab as plt
from sklearn.svm import OneClassSVM
import cellh5
import cPickle as pickle
from numpy import recfromcsv
import pandas
import time
from matplotlib.mlab import PCA
from scipy.stats import nanmean


class OutlierDetection(object):
	classifier_class = OneClassSVM
	def __init__(self, name, mapping_file, cellh5_file, training_sites=(1,), rows=None, cols=None, gamma=1.0/200, nu=0.05):
		self.name = name
		
		self.mapping_file = mapping_file
		self.cellh5_file = cellh5_file
		
		self.read_mapping(sites=training_sites, rows=rows, cols=cols)
		
		self.gamma = gamma
		self.nu = nu
		self.classifier = None
		
		self.pca_dims = 4
		
	def read_mapping(self, sites=None, rows=None, cols=None):
		self.mapping = pandas.read_csv(self.mapping_file, sep='\t')
		
		if sites is not None:
			self.mapping = self.mapping[self.mapping['Site'].isin(sites)]
		if rows is not None:
			self.mapping = self.mapping[self.mapping['Row'].isin(rows)]
		if cols is not None:
			self.mapping = self.mapping[self.mapping['Column'].isin(cols)]
		
		self.mapping.reset_index(inplace=True)
		
	def save(self, file_name=None):
		import datetime
		if file_name is None:
			file_name = self.name + "_" + datetime.datetime.now().strftime("%y-%m-%d-%H-%M") + "_g%5.4f_n%5.4f"%(self.gamma, self.nu) + ".pkl"
		with open(file_name, 'wb') as f:
			pickle.dump(self, f)
		return file_name
	
	@staticmethod
	def load(file_name):
		with open(file_name, 'rb') as f:
			obj = pickle.load(f)
		return obj
	
	def train(self, train_on=('neg', )):
		training_matrix = self.get_data(train_on, 'PCA')
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
		training_matrix_list = self.mapping[self.mapping['Group'].isin(test_on)][['Well','Site','PCA']].iterrows()

		predictions = {}
		for idx, (well, site, tm) in training_matrix_list:
			print well, site, tm.shape,
			if tm.shape[0] == 0:
				predictions[idx] = numpy.zeros((0, 0))
			else:
				#tm = self._remove_nan_rows(tm)
				predictions[idx] = self.predict_with_classifier(tm)
			
		self.mapping = self.mapping.join(pandas.DataFrame({'Predictions' : pandas.Series(predictions)}))
			
	def _remove_nan_rows(self, data):
		data = data[:, self._non_nan_feature_idx]
		if numpy.any(numpy.isnan(data)):
			print 'Warning: Nan values in prediction found. Trying to delete examples:'
			nan_rows = numpy.unique(numpy.where(numpy.isnan(data))[0])
			self._non_nan_sample_idx = [x for x in xrange(data.shape[0]) if x not in nan_rows]
			print 'deleting %d of %d' % (data.shape[0]-len(self._non_nan_sample_idx), data.shape[0])
			
			# get rid of nan samples (still)
			data = data[self._non_nan_sample_idx, :]
		data = (data - self._normalization_means) / self._normalization_stds
		return data
			
	def read_feature(self):
		m = self.mapping
		ch5_file = cellh5.CH5File(self.cellh5_file)
		
		features = []
		object_counts = []
		for i, row in m.iterrows():
			well = row['Well']
			site = str(row['Site'])
			
			ch5_pos = ch5_file.get_position(well, site)
			
			feature_matrix = ch5_pos.get_object_features()
			a = ch5_pos.get_object_table('primary__primary')
			
			object_count = len(feature_matrix)
			object_counts.append(object_count)
			
			if object_count > 0:
				features.append(feature_matrix)
			else:
				features.append(numpy.zeros((0, features[0].shape[1]))) 
			
			print well, site, len(feature_matrix)
		
		m['Object features'] = features
		m['Object count'] = object_counts
		
	def compute_outlyingness(self):
		def _outlier_count(x):
			res = numpy.float32((x==-1).sum()) / len(x)
			if len(x)==0:
				print 'a'
			elif numpy.any(numpy.isnan(res)):
				print 'b'
			return res
			
		res = pandas.Series(self.mapping[self.mapping['Group'].isin(('target', 'pos', 'neg'))]['Predictions'].map(_outlier_count))
		self.mapping = self.mapping.join(pandas.DataFrame({'Outlyingness': res}))
	
	def train_pca(self, train_on=('neg', 'pos', 'target')):
		training_matrix = self.get_data(train_on)
		training_matrix = self.normalize_training_data(training_matrix)
		print 'Compute PCA', 'is nan?', numpy.any(numpy.isnan( training_matrix))
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
		self.mapping = self.mapping.join(pandas.DataFrame({'PCA': res}))	
	
	def train_classifier(self, training_matrix):
		self.classifier = self.classifier_class(kernel="rbf", nu=self.nu, gamma=self.gamma)
		self.classifier.fit(training_matrix)
		
	def predict_with_classifier(self, test_matrix):
		prediction = self.classifier.predict(test_matrix)
		print "%d / %d outliers (%3.2f)" % ( (prediction==-1).sum(), 
											 len(prediction), 
											 (prediction==-1).sum()/float(len(prediction)))
		return prediction
		
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
		print 'get_data for', self.mapping[self.mapping['Group'].isin(target)].reset_index()['siRNA ID']
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
		print 'normalize', numpy.any(numpy.isnan(data))
		
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
	

	def plot(self):
		f_x = 1
		f_y = 0
		
		x_min, y_min = 1000000, 100000
		x_max, y_max = -100000, -100000
		
		print len(self.mapping['PCA'])
		for i in range(len(self.mapping['PCA'])):
			data = self.mapping['PCA'][i]
			prediction = self.mapping['Predictions'][i]
			print self.mapping['siRNA ID'][i], data.shape
			
			if self.mapping['Group'][i] in ['pos', 'target']:
				plt.scatter(data[prediction==-1, f_x], data[prediction==-1, f_y], c='red', marker='d', s=42)
				plt.scatter(data[prediction==1, f_x], data[prediction==1, f_y], c='white', marker='d', s=42)
			else:
				plt.scatter(data[prediction==-1, f_x], data[prediction==-1, f_y], c='white', s=42)
				plt.scatter(data[prediction==1, f_x], data[prediction==1, f_y], c='white', s=42)
			
			x_min_cur, x_max_cur = data[:,f_x].min(), data[:,f_x].max()
			y_min_cur, y_max_cur = data[:,f_y].min(), data[:,f_y].max()
		
			x_min = min(x_min, x_min_cur)
			y_min = min(y_min, y_min_cur)
			x_max = max(x_max, x_max_cur)
			y_max = max(y_max, y_max_cur)
			
			ch5_file = cellh5.CH5File(self.cellh5_file)
			
			if True:
				import vigra, os
				for id in numpy.where(prediction==-1)[0]:
					well = str(self.mapping['Well'][i])
					site = str(self.mapping['Site'][i])
					ch5_pos = ch5_file.get_position(well, site)
					img = ch5_pos.get_gallery_image_contour(id, color='#FFFFFF', scale=2.0)
					try:
						os.makedirs('%s/%s/out' % (well, site,) )
					except:
						pass
					vigra.impex.writeImage(img, '%s/%s/out/%6.3f_%6.3f.png' % (well, site, data[id, f_x], data[id, f_y]))
				for id in numpy.where(prediction==1)[0]:
					try:
						os.makedirs('%s/%s/in' % (well, site,))
					except:
						pass
					well = self.mapping['Well'][i]
					site = str(self.mapping['Site'][i])
					ch5_pos = ch5_file.get_position(well, site)
					img = ch5_pos.get_gallery_image_contour(id, color='#FFFFFF', scale=2.0)
					vigra.impex.writeImage(img, '%s/%s/in/%6.3f_%6.3f.png' % (well, site, data[id, f_x], data[id, f_y]))
					
			
		x_min = -12
		y_min = -25
		
		x_max = 42
		y_max = 42	
			
		xx, yy = numpy.meshgrid(numpy.linspace(x_min, x_max, 100), numpy.linspace(y_min, y_max, 100))
		#Z = self.classifier.decision_function(numpy.c_[xx.ravel(), yy.ravel()])
		matrix = numpy.zeros((100*100, self.pca_dims))
		matrix[:, f_x] = xx.ravel()
		matrix[:, f_y] = yy.ravel()
		
		
		Z = self.classifier.decision_function(matrix)
		Z = Z.reshape(xx.shape)
		#print Z
		#Z = (Z - Z.min())
		#Z = Z / Z.max()
		#print Z.min(), Z.max()
		#Z = numpy.log(Z+0.001)
		
		plt.contourf(xx, yy, Z, levels=numpy.linspace(Z.min(), Z.max(), 8), cmap=plt.matplotlib.cm.Greens, hold='on', alpha=0.5)
		#a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
		#plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')
		
		
		plt.axis('tight')
		
		plt.xlim((x_min, x_max))
		plt.ylim((y_min, y_max))
		plt.axis('off')
		plt.show(block=True)
		
	def purge_feature(self):
		del self.mapping['Object features']
		
	def make_heat_map(self):
		rows = numpy.unique(self.mapping['Row'])
		cols = numpy.unique(self.mapping['Column'])
		#sites = numpy.unique(self.mapping['Site'])
		print rows
		print cols
		
		target_col = 'Outlyingness'
		
		heatmap = numpy.zeros((len(rows), len(cols)), dtype=numpy.float32)
		
		for r_idx, r in enumerate(rows):
			for c_idx, c in enumerate(cols):
				target_value = self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c)][target_col]
				target_count = self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c)]['Object count']
				target_grp   = self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c)]['Group']
				
				if target_count.sum() == 0:
					value = -1
				else:
					value = numpy.nansum(target_value * target_count) / float(target_count.sum())
					#value = nanmean(target_value) 
				
				
				if numpy.isnan(value):
					print 'Warning: there are nans...'
				if target_count.sum() > 0:
					heatmap[r_idx, c_idx] = value
# 				else:
# 					heatmap[r_idx, c_idx] = -1
# 					
# 				if target_grp.iloc[0] in ('neg', 'pos'):
# 					heatmap[r_idx, c_idx] = -1
				
		cmap = plt.matplotlib.cm.Greens
		cmap.set_under(plt.matplotlib.cm.Oranges(0))
		#cmap.set_under('w')
					
		print 'Heatmap', heatmap.max(), heatmap.min()	
		
		ax = plt.subplot(111)
		plt.pcolor(heatmap, cmap=cmap, vmin=0)
		plt.colorbar()

		for r_idx, r in enumerate(rows):
			for c_idx, c in enumerate(cols):
				try:
					text_grp = str(self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c)]['Group'].iloc[0])
					text_gene = str(self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c)]['siRNA ID'].iloc[0])
					count = self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c)]['Object count'].sum()
				except IndexError:
					text_grp = "empty"
					text_gene = "empty"
					count = -1
					
				t = plt.text(c_idx + 0.5, r_idx + 0.5, '%s\n%s\n%d' %( text_grp, text_gene, count), horizontalalignment='center', verticalalignment='center', fontsize=8)
				if heatmap[r_idx, c_idx] > 0.1:
					t.set_color('w')
					
		# put the major ticks at the middle of each cell
		ax.set_xticks(numpy.arange(heatmap.shape[1])+0.5, minor=False)
		ax.set_yticks(numpy.arange(heatmap.shape[0])+0.5, minor=False)
		
		# want a more natural, table-like display
		ax.invert_yaxis()
		ax.xaxis.tick_top()
		
		ax.set_xticklabels(list(cols), minor=False)
		ax.set_yticklabels(list(rows), minor=False)
		
		#ax.set_title('Phenotypic Outliers')
		plt.show()
	
		return heatmap
	
	def make_hit_list(self):
		group_on = 'siRNA ID'
		#group_on = 'Gene Symbol'
		
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
			
		ax = plt.subplot(111)
		ax.errorbar(range(len(means)), means, yerr=stds, fmt='o')
		ax.set_xticks(range(len(means)), minor=False)
		ax.set_xticklabels(genes, rotation=90)
		ax.axhline(means.mean(), label='Target mean')
		ax.axhline(means.mean() + means.std()*2, color='k', label='Target cutoff at 2 sigma')
		ax.axhline(neg_mean, color='g', label='Negative control mean')
		ax.axhline(pos_mean, color='r', label='Positive control mean')
		
		plt.legend(loc=2)
		plt.ylabel('Outlyingness (OC-SVM)')
		plt.xlabel('Target genes')
		plt.title('Outliers grouped by gene' )
		
		plt.tight_layout()
		plt.show()
				
		
		
		
class OutlierTest(object):
	def __init__(self, name, mapping_file, ch5_file):
		self.name = name
		self.mapping_file = mapping_file 
		self.ch5_file = ch5_file 
		
	def setup(self, rows=None, cols=None):
		self.od = OutlierDetection(self.name,
							  self.mapping_file, 
							  self.ch5_file,
							  rows=rows,
							  cols=cols,
							  gamma=1.0/60, 
							  nu=0.08
							  )
		self.od.read_feature()
		
		self.od.train_pca()
		self.od.predict_pca()
		
		self.od.train()
		self.od.predict()
		
		self.od.compute_outlyingness()
		#self.od.purge_feature()
		self.last_file = self.od.save()
		print 'Storing to file name', self.last_file
		
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
		#self.od.__class__ = OutlierDetection
		
		
def sara_screen_analysis():
	ot = OutlierTest('matthias', 'dc', 'dc')
	ot.load_last('testing_13-07-23-13-24_g0.0050_n0.0500.pkl')
	#ot.load_last()
	#ot.od.plot()
 	a = ot.od.make_hit_list()
 	a = ot.od.make_heat_map()
 	
if __name__ == "__main__":
	ot = OutlierTest('matthias',
					 'M:/experiments/Experiments_002300/002324/meta/CellCog/mapping/MD9_Grape_over_Time.txt',
					 'M:/experiments/Experiments_002300/002324/meta/CellCog/analysis/hdf5/_all_positions.ch5'
					)
	ot.setup(
			#rows=('A', 'B', 'C', 'D', 'E'), 
			#cols=(8,9,10,11,12,13,14,15,19,24)
			)
	#ot.od.plot()
 	a = ot.od.make_hit_list()
 	a = ot.od.make_heat_map()


