"""
                           The CellCognition Project
                     Copyright (c) 2006 - 2011 Michael Held
                      Gerlich Lab, ETH Zurich, Switzerland
                              www.cellcognition.org

              CellCognition is distributed under the LGPL License.
                        See trunk/LICENSE.txt for details.
                 See trunk/AUTHORS.txt for author contributions.
"""
__all__ = []

#-------------------------------------------------------------------------------
# standard library imports:
#
import os
import zlib
import base64

#-------------------------------------------------------------------------------
# extension module imports:
#
import h5py, \
       numpy, \
       vigra, \
       random

import time as timing

#-------------------------------------------------------------------------------
# cecog imports:
#
from cecog.io.dataprovider import TerminalObjectItem, \
                                  ObjectItem

#-------------------------------------------------------------------------------
# constants:
#

MAX_OBJECT_ITEM_CACHE_SIZE = 3

#-------------------------------------------------------------------------------
# functions:
#

def print_timing(func):
    def wrapper(*arg, **kwargs):
        t1 = timing.time()
        res = func(*arg, **kwargs)
        t2 = timing.time()
        name = str(func.__class__) + ' :: ' + func.func_name if hasattr(func, '__class__') else func.func_name
        print '%s took %0.3f ms' % (name, (t2-t1)*1000.0)
        return res
    return wrapper

import types
def MixIn(pyClass, mixInClass, makeAncestor=0):
    if makeAncestor:
        if mixInClass not in pyClass.__bases__:
            pyClass.__bases__ = pyClass.__bases__ + (mixInClass,)
    else:
        # Recursively traverse the mix-in ancestor
        # classes in order to support inheritance
        baseClasses = list(mixInClass.__bases__)
        baseClasses.reverse()
        for baseClass in baseClasses:
            MixIn(pyClass, baseClass)
        # Install the mix-in methods into the class
        for name in dir(mixInClass):
            if not name.startswith('__'):
            # skip private members
                member = getattr(mixInClass, name)
                if type(member) is types.MethodType:
                    member = member.im_func
                setattr(pyClass, name, member)

class CellTerminalObjectItemMixin(object):
    BOUNDING_BOX_SIZE = 100
    
    @property
    def image(self):
        if not hasattr(self, '_image'):
            channel_idx = self.channel_idx
            self._image = self._get_image(self.time, self.local_idx, channel_idx)
            
#            sib = self.get_siblings()
#            if sib is not None:
#                image_sib = sib.image
#                new_shape = (self.BOUNDING_BOX_SIZE,)*2 + (3,)
#                image = numpy.zeros(new_shape, dtype=numpy.uint8)
#                image[0:image_own.shape[0],0:image_own.shape[1],0] = image_own
#                image[0:image_sib.shape[0],0:image_sib.shape[1],1] = image_sib
#            else:
#                image = image_own
#            self._image = image
        
        return self._image 
    
    @property
    def crack_contour(self):
        crack_contour = self._get_crack_contours(self.time, self.local_idx)
        bb = self.bounding_box
        crack_contour[:,0] -= bb[0][0]
        crack_contour[:,1] -= bb[0][1]  
        return crack_contour.clip(0, self.BOUNDING_BOX_SIZE)
    
    @property
    def predicted_class(self):
        # TODO: This can access can be cached by parent
        if not hasattr(self, '_predicted_class'):
            classifier_idx = self.classifier_idx()
            self._predicted_class = self._get_additional_object_data(self.name, 'classifier', classifier_idx) \
                                        ['prediction'][self.idx]
        return self._predicted_class[0]
    
    @property
    def features(self):
        # TODO: This can access can be cached by parent
        if not hasattr(self, '_features'):
            self._features = self._get_additional_object_data(self.name, 'feature')[self.idx,:]
            self._features.shape = (1,) + self._features.shape
        return self._features
    
    @property
    def feature_names(self):
        return self.get_position().object_feature[self.name]
    
    @property
    def time(self):
        return self._local_idx[0]
    
    @property
    def local_idx(self):
        return self._local_idx[1]
    
    def classifier_idx(self):
        return self.get_plate().object_classifier_index[self.name] \
                if self.name in self.get_plate().object_classifier_index \
                else None
    
    @property
    def channel_idx(self):
        if not hasattr(self, '_channel_idx'):
            self._channel_idx = self.get_plate().regions[self.get_position().sub_objects[self.name]]['channel_idx']
        return self._channel_idx
        
    @property
    def bounding_box(self):
        if not hasattr(self, '_bounding_box'):   
            objects = self.parent.object_np_cache['terminals'][self.time]['object']
            self._bounding_box = (objects['upper_left'][self.local_idx], objects['lower_right'][self.local_idx])
        return self._bounding_box
    
    def _get_image(self, t, obj_idx, c, bounding_box=None):
        self.get_position().read_image_data()
        
        if bounding_box is None:
            ul, lr = self.bounding_box
        else:
            ul, lr = bounding_box
        offset_0 = (self.BOUNDING_BOX_SIZE - lr[0] + ul[0])
        offset_1 = (self.BOUNDING_BOX_SIZE - lr[1] + ul[1]) 
        ul[0] = max(0, ul[0] - offset_0/2 - cmp(offset_0%2,0) * offset_0 % 2) 
        ul[1] = max(0, ul[1] - offset_1/2 - cmp(offset_1%2,0) * offset_1 % 2)      
        lr[0] = min(self.get_position()._hf_group_np_copy.shape[4], lr[0] + offset_0/2) 
        lr[1] = min(self.get_position()._hf_group_np_copy.shape[3], lr[1] + offset_1/2) 
        
        self._bounding_box = (ul, lr)
        # TODO: get_iamge returns am image which might have a smaller shape than 
        #       the requested BOUNDING_BOX_SIZE, I dont see a chance to really
        #       fix it, without doing a copy...
        res = self.get_position()._hf_group_np_copy[c, t, 0, ul[1]:lr[1], ul[0]:lr[0]]
        return res

    def _get_crack_contours(self, t, obj_idx):  
        crack_contours_string = self.parent.object_np_cache['terminals'][t]['crack_contours'][obj_idx]                               
        return numpy.asarray(zlib.decompress( \
                             base64.b64decode(crack_contours_string)).split(','), \
                             dtype=numpy.float32).reshape(-1,2)
        
    def _get_object_data(self, t, obj_idx, c):
        bb = self.get_bounding_box(t, obj_idx, c)
        img, new_bb = self.get_image(t, obj_idx, c, bb)
        cc = self.get_crack_contours(t, obj_idx, c)
        cc[:,0] -= new_bb[0][0] 
        cc[:,1] -= new_bb[0][1]
        return img, cc
    
    def _get_additional_object_data(self, object_name, data_fied_name, index=None):
        if index is None:
            return self.get_position()._hf_group['object'][object_name][data_fied_name]
        else:
            return self.get_position()._hf_group['object'][object_name][data_fied_name][str(index)]
    
    @property
    def class_color(self):
        if not hasattr(self, '_class_color'):
            classifier_idx = self.classifier_idx()
            if classifier_idx is not None:
                classifier = self.get_plate().object_classifier[self.name, classifier_idx]
                self._class_color = dict(enumerate(classifier['schema']['color'].tolist()))[self.predicted_class]
            else:
                self._class_color = None
   
        return self._class_color
    
    def compute_features(self):
#        print 'compute feature call for', self.name, self.id  
        pass
    
class EventObjectItemMixin(object):
    def compute_features(self):
#        print 'compute feature call for', self.name, self.id  
#        for feature in trajectory_features:
#            if isinstance(self, feature.type):
#                self[feature.name] =  feature.compute(self.children())
#                
        self['prediction'] = [x.predicted_class for x in self.children()]
        
    @property
    def item_features(self):
        children = self.children()
        if children is not None:
            return numpy.concatenate([c.features for c in children], axis=0)
        else:
            return None
            
    @property
    def item_labels(self):
        children = self.children()
        if children is not None:
            return [c.predicted_class for c in children]
        else:
            return None
        
    @property
    def sibling_item_features(self):
        children = self.children()
        if children is not None:
            return numpy.concatenate([c.get_siblings().features for c in children], axis=0)
        else:
            return None
        
    @property
    def item_colors(self):
        children = self.children()
        if children is not None:
            return [c.class_color for c in children]
        else:
            return None
        
    @property
    def item_feature_names(self):
        return self.get_plate().object_feature[self.get_position().sub_objects[self.name]]
    
    def item_feature_min_max(self, feature_idx):
        if not hasattr(self.parent, 'feature_min_max'):
            self.parent.feature_min_max = {}
        
        min_ = 1000000
        max_ = - 1000000
        if feature_idx not in self.parent.feature_min_max.keys():
            for p in self.parent:
                tmin = p.item_features[:,feature_idx].min()
                tmax = p.item_features[:,feature_idx].max()
                
                if tmin < min_:
                    min_ = tmin 
                    
                if tmax > max_:
                    max_ = tmax 
                     
            self.parent.feature_min_max[feature_idx] = min_, max_
            
        return self.parent.feature_min_max[feature_idx]
    
    def sibling_item_feature_min_max(self, feature_idx):
        if not hasattr(self.parent, 'sibling_feature_min_max'):
            self.parent.sibling_feature_min_max = {}
        
        min_ = 1000000
        max_ = - 1000000
        if feature_idx not in self.parent.sibling_feature_min_max.keys():
            for p in self.parent:
                tmin = p.sibling_item_features[:,feature_idx].min()
                tmax = p.sibling_item_features[:,feature_idx].max()
                
                if tmin < min_:
                    min_ = tmin 
                    
                if tmax > max_:
                    max_ = tmax 
                     
            self.parent.sibling_feature_min_max[feature_idx] = min_, max_
            
        return self.parent.sibling_feature_min_max[feature_idx]
                
MixIn(TerminalObjectItem, CellTerminalObjectItemMixin, True)
MixIn(ObjectItem, EventObjectItemMixin, True)


class TrajectoryFeatureBase(object):
    def compute(self):
        raise NotImplementedError('TrajectoryFeatureBase.compute() has to be implemented by its subclass')

class TrajectoryFeatureMeanIntensity():
    name = 'Mean intensity'
    type = ObjectItem
    
    def compute(self, trajectory_seq):
        value = 0
        for t in trajectory_seq:
            value += t.image.mean()
        value /= len(trajectory_seq)
        return value
    
trajectory_features = [tf() for tf in TrajectoryFeatureBase.__subclasses__()]