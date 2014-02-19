import sys
from PyQt4 import QtGui, QtCore

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg

from matplotlib import rcParams
import resources
for t in rcParams.keys():
    print t
     

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['pdf.fonttype'] = 42
# rcParams['axes.linewidth'] = rcParams['axes.linewidth']*2
# rcParams['legend.numpoints'] = 1
# rcParams['legend.markerscale'] = 0
# rcParams['xtick.major.width'] = 2
# rcParams['ytick.major.width'] = 2
# rcParams['text.color'] = 'white'
rcParams['xtick.color'] = 'white'
rcParams['ytick.color'] = 'white'
rcParams['ytick.labelsize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['axes.labelsize'] = 18

rcParams['axes.labelcolor'] = 'white'

from matplotlib.widgets import RectangleSelector
from matplotlib.colors import colorConverter
from matplotlib.collections import CircleCollection
from matplotlib import cm 
from matplotlib import colors

import cellh5, numpy

class DataPoint(object):
    colorin = colorConverter.to_rgba('red')
    colorout = colorConverter.to_rgba('blue')
    def __init__(self, x, y, ref, include=False):
        self.x = x
        self.y = y
        self.ref = ref
        if include: 
            self.color = self.colorin
        else: 
            self.color = self.colorout

class IScatter(QtGui.QMainWindow):
    def __init__(self, scatter, image):  
        QtGui.QMainWindow.__init__(self)
        self.setWindowTitle('Test')
        
        self.main_widget = QtGui.QWidget(self)
        self.setCentralWidget(self.main_widget)
        
        layout = QtGui.QHBoxLayout()
        layout.addWidget(scatter)
        layout.addWidget(image)
        self.main_widget.setLayout(layout)  
        
        #widget_1.canvas.setParent(widget_1)
        scatter.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        image.canvas.setFocus()      
        
        self.apply_css("iscatter.css")
        
    def apply_css(self, css_file):
        qss_file = QtCore.QFile(css_file)
        qss_file.open(QtCore.QFile.ReadOnly);
        css = QtCore.QLatin1String(qss_file.readAll());
        self.setStyleSheet(css);    
    
class IScatterWidget(QtGui.QWidget):
    image_changed = QtCore.pyqtSignal(numpy.ndarray)
    def __init__(self, parent=None):   
        QtGui.QWidget.__init__(self, parent)
        self.figure = Figure()
        self.figure.patch.set_alpha(0)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)

        self.selector = RectangleSelector(self.axes, self.selector_cb,
                                       drawtype='box', useblit=True,
                                       button=[1,],
                                       minspanx=5, minspany=5,
                                       spancoords='pixels')
        
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.canvas)
        
        axis_selector = QtGui.QWidget(self)
        axis_selector_layout = QtGui.QHBoxLayout()
        
        axis_selector_layout.addWidget(QtGui.QLabel("X:"))
        self.axis_x_cmb = QtGui.QComboBox(self)
        axis_selector_layout.addWidget(self.axis_x_cmb)
        
        axis_selector_layout.addStretch()
        
        axis_selector_layout.addWidget(QtGui.QLabel("Y:"))
        self.axis_y_cmb = QtGui.QComboBox(self)
        axis_selector_layout.addWidget(self.axis_y_cmb)
        
        axis_selector_layout.addStretch()
        
        axis_selector_layout.addWidget(QtGui.QLabel("C:"))
        self.axis_c_chk = QtGui.QCheckBox(self)
        self.axis_c_chk.setChecked(False)
        self._last_c_dim = 0
        self.axis_c_cmb = QtGui.QComboBox(self)
        self.axis_c_cmb.setEnabled(False)
        axis_selector_layout.addWidget(self.axis_c_chk)
        axis_selector_layout.addWidget(self.axis_c_cmb)
        
        axis_selector.setLayout(axis_selector_layout)

        layout.addWidget(axis_selector)
        
        self.setLayout(layout)
        
        self.canvas.mpl_connect('scroll_event', self.mouse_wheel_zoom)

        
    def selector_cb(self, epress, erelease):
        x_min, x_max = min(epress.xdata, erelease.xdata), max(epress.xdata, erelease.xdata)
        y_min, y_max = min(epress.ydata, erelease.ydata), max(epress.ydata, erelease.ydata)
        
        ind = numpy.array([True if ((x > x_min) and 
                                    (x < x_max) and
                                    (y > y_min) and 
                                    (y < y_max)) 
                                    else False for x, y in self.xys])
        
        edgecolors = self.collection.get_edgecolors()
        facecolors = self.collection.get_facecolors()
        
        ids =  [self.data[k].ref for k in range(len(ind)) if ind[k]] 
        print ind.sum(), 'objects selected with refs', ids
        for i in range(len(self.xys)):
            if ind[i]:
                edgecolors[i] = DataPoint.colorin
            else:
                edgecolors[i] = facecolors[i]
                
        self.canvas.draw()
        self.update_image(ind)
        
        
    def mouse_wheel_zoom(self, event, base_scale=1.2):
        if event.key is None or not event.key.startswith('control'):
            return
        cur_xlim = self.axes.get_xlim()
        cur_ylim = self.axes.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/ base_scale
            xd, yd = xdata, ydata
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
            xd, yd = xdata, ydata
        else:
            # deal with something that should never happen
            scale_factor = 1
            print event.button
        # set new limits
        self.axes.set_xlim([xd - cur_xrange*scale_factor,
                            xd + cur_xrange*scale_factor])
        self.axes.set_ylim([yd - cur_yrange*scale_factor,
                     yd + cur_yrange*scale_factor])
        self.canvas.draw() # force re-draw
        
    def update_image(self, ind):
        self.ind = ind
        ids =  [self.data[k].ref for k in range(len(ind)) if ind[k]]       
        img = self.ch5_pos.get_gallery_image_matrix(ids, (10,12))
        self.image_changed.emit(img)
        
    def set_data(self, data_matrix, data_names, x_dim, y_dim, data_ch5_idx, ch5_pos):
        self.ch5_pos = ch5_pos
        self.data_matrix = data_matrix
        self.data_names = data_names
        self.data_ch5_idx = data_ch5_idx
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.c_dim = None
        self.ind = [False]*data_matrix.shape[0]
        
        for dn in data_names:
            self.axis_x_cmb.addItem(dn)
            self.axis_y_cmb.addItem(dn)
            self.axis_c_cmb.addItem(dn)
            
        self.update_axis()
        
        self.axis_x_cmb.setCurrentIndex(x_dim)
        self.axis_y_cmb.setCurrentIndex(y_dim)
        
        self.axis_x_cmb.currentIndexChanged.connect(self.axis_x_changed)
        self.axis_y_cmb.currentIndexChanged.connect(self.axis_y_changed)
        self.axis_c_cmb.currentIndexChanged.connect(self.axis_c_changed)
        self.axis_c_chk.stateChanged.connect(self.axis_c_toggled)
         
    def axis_x_changed(self, x_dim):
        self.x_dim = x_dim
        self.update_axis()
        
    def axis_y_changed(self, y_dim):
        self.y_dim = y_dim
        self.update_axis()
        
    def axis_c_changed(self, c_dim):
        self.c_dim = c_dim
        self.update_axis()
        
    def axis_c_toggled(self, state):
        if not state:
            self._last_c_dim = self.c_dim
            self.c_dim = None
            self.axis_c_cmb.setEnabled(False)
        else:
            self.c_dim = self._last_c_dim
            self.axis_c_cmb.setEnabled(True)
        
        self.update_axis()
        
    def update_axis(self):
        self.axes.clear()
        self.data = [DataPoint(xy[0], xy[1], self.data_ch5_idx[i], False) 
                     for i, xy in enumerate(self.data_matrix[:, [self.x_dim, self.y_dim]])]
        self.Nxy = len(self.data)

        if self.c_dim is None:
            facecolors = [d.color for d in self.data]
            edgecolors = [d.color for d in self.data]
        else:
            f_0_min = self.data_matrix[:, self.c_dim].min()
            f_0_max = self.data_matrix[:, self.c_dim].max()
            cNorm  = colors.Normalize(vmin=f_0_min, vmax=f_0_max)
            scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.jet)
            facecolors = [scalarMap.to_rgba(c) for c in self.data_matrix[:, self.c_dim]]
            edgecolors = [scalarMap.to_rgba(c) for c in self.data_matrix[:, self.c_dim]]
            
        self.xys = [(d.x, d.y) for d in self.data]
        
        for i in range(len(self.xys)):
            if self.ind[i]:
                edgecolors[i] = DataPoint.colorin
            
        
        self.collection = CircleCollection(
            sizes=(22,),
            facecolors=facecolors,
            edgecolors=edgecolors,
            offsets = self.xys,
            transOffset = self.axes.transData)

        self.axes.add_collection(self.collection)
        
        self.update_axis_lims()
        self.canvas.draw()
        
    def update_axis_lims(self):
        f_0_min = self.data_matrix[:, self.x_dim].min()
        f_0_max = self.data_matrix[:, self.x_dim].max()
        f_1_min = self.data_matrix[:, self.y_dim].min()
        f_1_max = self.data_matrix[:, self.y_dim].max()
        
        self.axes.set_xlim(f_0_min, f_0_max)
        self.axes.set_ylim(f_1_min, f_1_max)
        self.axes.set_xlabel(self.data_names[self.x_dim])
        self.axes.set_ylabel(self.data_names[self.y_dim])
        
class SimpleMplImageViewer(QtGui.QWidget):
    def __init__(self, parent=None):   
        QtGui.QWidget.__init__(self, parent)
        self.figure = Figure()
        self.figure.patch.set_alpha(0)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_axes([0, 0, 1, 1])
        self.axes.axis('off')
        self.axes.set_frame_on(False)
        
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
  
    def show_image(self, img):
        print type(img), img.shape
        self.axes.imshow(img, cm.Greys_r)
        self.canvas.draw()


def start_qt_event_loop():
    app = QtGui.QApplication(sys.argv)
    app.setStyle("plastique")     
    return app
    
if __name__ == "__main__":
    app = start_qt_event_loop()
    ch5_file = cellh5.CH5File("C:/Users/sommerc/cellh5/data/0038.ch5", "r")
    ch5_pos = ch5_file.current_pos
     
    features = ch5_pos.get_object_features()
    times = ch5_pos.get_all_time_idx()
    times_idx = times < 3
    features = features[times_idx, :]
    
    center_x = ch5_pos.get_center(times_idx)['x']
    center_y = ch5_pos.get_center(times_idx)['y']
    predictions = ch5_pos.get_class_prediction()[times_idx]['label_idx']
    
    features = numpy.c_[features, times[times_idx], center_x, center_y, predictions]
    
 
    data_names = ch5_file.object_feature_def()
    
    iscatter = IScatterWidget()
    iscatter.set_data(features, data_names + ['Time', 'Center x', 'Center y', 'Classificaiton'], 222, 236, numpy.nonzero(times_idx)[0], ch5_pos)
    
    image_viewer = SimpleMplImageViewer()
    iscatter.image_changed.connect(image_viewer.show_image)
    
    mw = IScatter(iscatter, image_viewer)
    mw.show()
    app.exec_()
    