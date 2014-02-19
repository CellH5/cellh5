import sys
from PyQt4 import QtGui, QtCore

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg

from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['pdf.fonttype'] = 42
# rcParams['axes.linewidth'] = rcParams['axes.linewidth']*2
# rcParams['legend.numpoints'] = 1
# rcParams['legend.markerscale'] = 0
# rcParams['xtick.major.width'] = 2
# rcParams['ytick.major.width'] = 2
# rcParams['text.color'] = 'white'
# rcParams['xtick.color'] = 'white'
# rcParams['ytick.color'] = 'white'

from matplotlib.widgets import Lasso
from matplotlib.colors import colorConverter
from matplotlib.collections import RegularPolyCollection, CircleCollection
from matplotlib import path
from pylab import cm 

from numpy import nonzero
from numpy.random import rand

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

class LassoManager(QtCore.QObject):
    selectionChanged = QtCore.pyqtSignal(numpy.ndarray)
    def __init__(self, ax, data):
        super(QtCore.QObject, self).__init__()
        self.axes = ax
        self.canvas = ax.figure.canvas
        self.data = data
        
        self.axes.clear()
        self.Nxy = len(data)

        self.facecolors = [d.color for d in data]
        self.xys = [(d.x, d.y) for d in data]
        
        self.collection = CircleCollection(
            sizes=(20,),
            facecolors=self.facecolors,
            offsets = self.xys,
            transOffset = ax.transData)

        ax.add_collection(self.collection)

        self.cid = self.canvas.mpl_connect('button_press_event', self.onpress)
        self.selectionChanged.connect(self.update_colors)

    def callback(self, verts):
        print 'callback'
        p = path.Path(verts)
        ind = p.contains_points(self.xys)
        self.selectionChanged.emit(ind)
        
    def update_colors(self, ind):
        facecolors = self.collection.get_facecolors()
        self.facecolors = facecolors.copy()
        ids =  [self.data[k].ref for k in range(len(ind)) if ind[k]] 
        print ind.sum(), 'objects selected with refs', ids
        for i in range(len(self.xys)):
            if ind[i]:
                facecolors[i] = DataPoint.colorin
            else:
                facecolors[i] = DataPoint.colorout
                
        self.canvas.draw_idle()
        self.canvas.widgetlock.release(self.lasso)
        del self.lasso
        
    def onpress(self, event):
        if self.canvas.widgetlock.locked(): 
            return
        elif event.inaxes is None: 
            return
        elif event.button != 1:
            return 
        self.lasso = Lasso(event.inaxes, (event.xdata, event.ydata), self.callback)
        # acquire a lock on the widget drawing
        self.canvas.widgetlock(self.lasso)

class MainWindow(QtGui.QMainWindow):
    def __init__(self, widget_1, widget_2):  
        QtGui.QMainWindow.__init__(self)
        self.setWindowTitle('Test')
        
        self.main_widget = QtGui.QWidget(self)
        self.setCentralWidget(self.main_widget)
        
        layout = QtGui.QHBoxLayout()
        layout.addWidget(widget_1)
        layout.addWidget(widget_2)
        self.main_widget.setLayout(layout)  
        
        #widget_1.canvas.setParent(widget_1)
        widget_1.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        widget_1.canvas.setFocus()      
        
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
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)

        self.lasso = LassoManager(self.axes, [])
        self.lasso.selectionChanged.connect(self.update_image)
        
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
        axis_selector.setLayout(axis_selector_layout)

        layout.addWidget(axis_selector)
        
        self.setLayout(layout)
        
        self.canvas.mpl_connect('scroll_event', self.mouse_wheel_zoom)
        
    def mouse_wheel_zoom(self, event, base_scale=1.1):
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
        img = self.ch5_pos.get_gallery_image_matrix(ids, (10,10))
        self.image_changed.emit(img)
        
    def set_data(self, data_matrix, data_names, x_dim, y_dim, data_ch5_idx, ch5_pos):
        self.ch5_pos = ch5_pos
        self.data_matrix = data_matrix
        self.data_names = data_names
        self.data_ch5_idx = data_ch5_idx
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.ind = [False]*data_matrix.shape[0]
        
        for dn in data_names:
            self.axis_x_cmb.addItem(dn)
            self.axis_y_cmb.addItem(dn)
            
        self.update_axis()
        
        self.axis_x_cmb.setCurrentIndex(x_dim)
        self.axis_y_cmb.setCurrentIndex(y_dim)
        
        self.axis_x_cmb.currentIndexChanged.connect(self.axis_x_changed)
        self.axis_y_cmb.currentIndexChanged.connect(self.axis_y_changed)
         
    def axis_x_changed(self, x_dim):
        self.x_dim = x_dim
        self.update_axis()
        
    def axis_y_changed(self, y_dim):
        self.y_dim = y_dim
        self.update_axis()
        
    def update_axis(self):
        self.data = [DataPoint(xy[0], xy[1], self.data_ch5_idx[i], self.ind[i]) 
                     for i, xy in enumerate(self.data_matrix[:, [self.x_dim, self.y_dim]])]
        self.lasso = LassoManager(self.axes, self.data)
        self.lasso.selectionChanged.connect(self.update_image)
        self.update_axis_lims()
        self.canvas.draw()
        
    def update_axis_lims(self):
        f_0_min = self.data_matrix[:, self.x_dim].min()
        f_0_max = self.data_matrix[:, self.x_dim].max()
        f_1_min = self.data_matrix[:, self.y_dim].min()
        f_1_max = self.data_matrix[:, self.y_dim].max()
        
        self.axes.set_xlim(f_0_min, f_0_max)
        self.axes.set_ylim(f_1_min, f_1_max)
        
class SimpleMplImageViewer(QtGui.QWidget):
    def __init__(self, parent=None):   
        QtGui.QWidget.__init__(self, parent)
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.axes.axis('off')
        
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
  
    def show_image(self, img):
        print type(img), img.shape
        self.axes.imshow(img, cm.Greys_r)
        self.canvas.draw()


def start_qt_event_loop():
    app = QtGui.QApplication(sys.argv)     
    return app
    
if __name__ == "__main__":
    app = start_qt_event_loop()
    ch5_file = cellh5.CH5File("C:/Users/sommerc/cellh5/data/0038.ch5", "r")
    ch5_pos = ch5_file.current_pos
     
    features = ch5_pos.get_object_features()
    times = ch5_pos.get_all_time_idx() == 1
    features = features[times, :]
 
    data_names = ch5_file.object_feature_def()
    
    iscatter = IScatterWidget()
    iscatter.set_data(features, data_names, 222, 236, numpy.nonzero(times)[0], ch5_pos)
    
    image_viewer = SimpleMplImageViewer()
    iscatter.image_changed.connect(image_viewer.show_image)
    
    mw = MainWindow(iscatter, image_viewer)
    mw.show()
    app.exec_()
    