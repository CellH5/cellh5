import sys
from PyQt4 import QtGui, QtCore

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg
from matplotlib.patches import Rectangle
import resources
from functools import partial

from matplotlib import rcParams

from matplotlib.widgets import RectangleSelector
from matplotlib.colors import colorConverter
from matplotlib.collections import CircleCollection
from matplotlib import cm 
from matplotlib import colors

import cellh5, numpy

import numpy
import matplotlib.pyplot as plt
import qimage2ndarray
from PyQt4 import QtGui, QtCore
import sys
from collections import OrderedDict 
from cecog.util.palette import SingleColorPalette

COLORS = OrderedDict([("red" , '#FF0000'),
                          ("green" , '#00FF00'),
                          ("blue" , '#0000FF'),
                          ("yellow" , '#FFFF00'),
                          ("magenta" ,'#FF00FF'),
                          ("cyan" , '#00FFFF'),
                          ("purple" , '#800080'),
                          ("olive" , '#808000'),
                          ("white" , '#FFFFFF')])



class ImageWithCmap(object):
    def __init__(self, array, cmap, name='Image', norm_min=0, norm_max=None):
        self.array = array.astype(numpy.float64)
        self.count = self.array.sum()
        print 'Total count of cells', self.count
        self.array /= float(self.count)
        
        if norm_max is None:
            self.norm_max = numpy.max(self.array)
            
        self.norm_min = norm_min
        
        if isinstance(cmap, (str,)):
            cmap = generate_standard_palette(cmap, self.norm_min, self.norm_max) 
        self.cmap = cmap
        self._name = name
        
        
        
        
        
    @property
    def name(self):
        return self._name
        
    def qimage(self, ):
        qimg = qimage2ndarray.gray2qimage(self.array, normalize=(self.norm_min, self.norm_max))
        qimg.setColorTable(self.cmap.qt)
        return qimg
    
    def max(self):
        return self.array.max()
    
    def reset_scaling(self):
        self.norm_min = self.array.min()
        self.norm_max = self.array.max()


def generate_palette(name, color, start, stop):
    palette = SingleColorPalette.from_hex_color(name, color)
    palette.qt = [QtGui.qRgb(r, g, b) for r,g,b in palette.lut]
    return palette

def generate_standard_palette(name, start=0.2, stop=1):
    return generate_palette(name, COLORS[name], start, stop)

def blend_images_max(images):
    """
    blend a list of QImages together by "lighten" composition (lighter color
    of source and dest image is selected; same effect as max operation)
    """
    assert len(images) > 0, 'At least one image required for blending.'
    pixmap = QtGui.QPixmap(images[0].width(), images[0].height())
    # for some reason the pixmap is NOT empty
    pixmap.fill(QtCore.Qt.black)
    painter = QtGui.QPainter(pixmap)
    painter.setCompositionMode(QtGui.QPainter.CompositionMode_Lighten)
    for image in images:
        if not image is None:
            painter.drawImage(0, 0, image)
    painter.end()
    return pixmap

def blend_images(image_list):
    blend_img = []
    for img in image_list:
        img_q = img.qimage()
        blend_img.append(img_q)
    
    qimage = blend_images_max(blend_img)
    result =  qimage2ndarray.rgb_view(QtGui.QImage(qimage))

    
    
    mask_img = numpy.zeros(image_list[0].array.shape)
    for img in image_list:
        mask_img += img.array
        
    mask_img = mask_img > 0
    
    tmp_ = result.sum(2) > 0
    
    result[mask_img]+=1
    result[tmp_]-=1
    
    result[result.sum(2) == 0] = 64    
    return result

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
    highlight_changed = QtCore.pyqtSignal(list)
    def __init__(self, scatter, scatter2):  
        QtGui.QMainWindow.__init__(self)
        self.setWindowTitle('Test')
        
        self.scatter = scatter
            
        self.image = SimpleImageViewer()
        self.image_combined = SimpleMplImageViewerWithBlending()
        self.scatter.image_changed.connect(self.image.show_image)
        scatter2.image_changed.connect(self.image.show_image)

        self.table = QtGui.QTableWidget(self)
        
        self.main_widget = QtGui.QWidget(self)
        self.setCentralWidget(self.main_widget)
        
        layout = QtGui.QGridLayout()
        layout.addWidget(self.scatter,0,0)
        layout.addWidget(scatter2, 0,1)
        layout.addWidget(self.table,0,2)
        layout.addWidget(self.image_combined,1,0,1,1, QtCore.Qt.AlignLeft)
        layout.addWidget(self.image,1,1,2,2)
        
        self.scatter.send_current_image.connect(self.image_combined.show_image_with_colormap)
        scatter2.send_current_image.connect(self.image_combined.show_image_with_colormap)
        
        self.main_widget.setLayout(layout)  
        
        #widget_1.canvas.setParent(widget_1)
#         self.scatter.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
#         self.image.canvas.setFocus()      
        
        self.scatter.selection_changed.connect(self.update_table)
        scatter2.selection_changed.connect(self.update_table)
        self.table.itemSelectionChanged.connect(self.extract_selection)
        self.highlight_changed.connect(self.image.highlight_cell)
        
        self.apply_css("C:/Users/sommerc/cellh5/pysrc/iscatter.css")
        
    def extract_selection(self):
        print  self.table.selectionModel().selectedIndexes()
        self.highlight_changed.emit(map(lambda x: x.row(), self.table.selectionModel().selectedIndexes()))
        
    def update_table(self, entries):
        self.table.clearContents()
        self.table.setRowCount(len(entries))
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderItem(0, QtGui.QTableWidgetItem('Plate'))
        self.table.setHorizontalHeaderItem(1, QtGui.QTableWidgetItem('Well'))
        self.table.setHorizontalHeaderItem(2, QtGui.QTableWidgetItem('Site'))
        self.table.setHorizontalHeaderItem(3, QtGui.QTableWidgetItem('Treatment 1'))
        self.table.setHorizontalHeaderItem(4, QtGui.QTableWidgetItem('Treatment 2'))
        if len(entries) == 0:
            return
        
        for i, e in enumerate(entries):
            for c in range(5):
                self.table.setItem(i, c, QtGui.QTableWidgetItem(e[c]))
                
        self.table.horizontalHeader().setResizeMode(QtGui.QHeaderView.ResizeToContents)
        self.table.setSortingEnabled(False)
 
    def apply_css(self, css_file):
        qss_file = QtCore.QFile(css_file)
        qss_file.open(QtCore.QFile.ReadOnly);
        css = QtCore.QLatin1String(qss_file.readAll());
        self.setStyleSheet(css);    
        
    def get_scatter_axes(self):
        return self.scatter.axes
    
class IScatterWidget(QtGui.QWidget):
    image_changed = QtCore.pyqtSignal(numpy.ndarray)
    selection_changed = QtCore.pyqtSignal(list)
    send_current_image = QtCore.pyqtSignal(list)
    
    def __init__(self, parent=None):   
        QtGui.QWidget.__init__(self, parent)
        self.figure = Figure()
        self.figure.patch.set_alpha(0)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111, adjustable='box')
        self.contour_eval_func = None

        self.selector = RectangleSelector(self.axes, self.selector_cb,
                                       drawtype='box', useblit=True,
                                       button=[1,],
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       lineprops=dict(color='orange', linestyle='-',
                                                    inewidth = 2, alpha=0.5),
                                       rectprops = dict(facecolor='orange', edgecolor = 'orange',
                                                        alpha=0.5, fill=True))
        
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self.canvas)
        
        axis_selector = QtGui.QWidget(self)
        axis_selector_layout = QtGui.QVBoxLayout()
        
        self.sample_selection_dct = {}
        for sample_selection in ['Plate', 'Well', 'Site', 'Treatment 1', 'Treatment 2', 'Group']:
            axis_selector_layout.addWidget(QtGui.QLabel(sample_selection))
            self.sample_selection_dct[sample_selection] = QtGui.QComboBox(self)
            axis_selector_layout.addWidget(self.sample_selection_dct[sample_selection])
            self.sample_selection_dct[sample_selection].addItem("All")
            
            self.sample_selection_dct[sample_selection].currentIndexChanged[str].connect(partial(self.sample_selection_changed, sample_selection))
            
        axis_selector_layout.addWidget(QtGui.QLabel("Results by"))
        self.cmb_result_sample_selection = QtGui.QComboBox(self)
        axis_selector_layout.addWidget(self.cmb_result_sample_selection)
        self.cmb_result_sample_selection.addItem("All")
        self.cmb_result_sample_selection.addItem("Clustering")
        self.cmb_result_sample_selection.addItem("Outliers")
        self.cmb_result_sample_selection.addItem("Classification")
        self.cmb_result_sample_selection.currentIndexChanged[str].connect(self.result_selection_changed)
             
        axis_selector_layout.addStretch()
        
        axis_selector_layout.addWidget(QtGui.QLabel("X:"))
        self.axis_x_cmb = QtGui.QComboBox(self)
        axis_selector_layout.addWidget(self.axis_x_cmb)

        axis_selector_layout.addWidget(QtGui.QLabel("Y:"))
        self.axis_y_cmb = QtGui.QComboBox(self)
        axis_selector_layout.addWidget(self.axis_y_cmb)

        self.axis_c_label = QtGui.QLabel("C:")
        axis_selector_layout.addWidget(self.axis_c_label)
        self.axis_c_chk = QtGui.QCheckBox(self)
        self.axis_c_chk.setChecked(False)
        self._last_c_dim = 0
        self.axis_c_cmb = QtGui.QComboBox(self)
        self.axis_c_cmb.setEnabled(False)
        axis_selector_layout.addWidget(self.axis_c_chk)
        axis_selector_layout.addWidget(self.axis_c_cmb)
        
        axis_selector_layout.addStretch()
        
        self.export_to_image_btn = QtGui.QPushButton('Export image File')
        axis_selector_layout.addWidget(self.export_to_image_btn)
        self.export_to_image_btn.clicked.connect(self.export_axes_to_image)
        
        self.export_to_cp_btn = QtGui.QPushButton('Export image clipboard')
        axis_selector_layout.addWidget(self.export_to_cp_btn)
        self.export_to_cp_btn.clicked.connect(self.export_axes_to_clipboard)
        
        
        self.cmb_colormap = QtGui.QComboBox(self)
        for c in SimpleMplImageViewerWithBlending.colors:
            self.cmb_colormap.addItem(c)
        axis_selector_layout.addWidget(self.cmb_colormap)
        
        self.cmb_colormap.currentIndexChanged[str].connect(self.colormap_changed)
        
        self.btn_move_to_combined_image = QtGui.QPushButton('vv')
        axis_selector_layout.addWidget(self.btn_move_to_combined_image)
        self.btn_move_to_combined_image.clicked.connect(self.move_to_combined_image)
          
        axis_selector.setLayout(axis_selector_layout)

        layout.addWidget(axis_selector)
        
        self.setLayout(layout)
        
        self.canvas.mpl_connect('scroll_event', self.mouse_wheel_zoom)
        
    def colormap_changed(self, colormap):
        self.current_cmap = str(colormap)
        self.update_axis()
    def move_to_combined_image(self):
        self.send_current_image.emit(self.current_raw_images)
        
    def export_axes_to_clipboard(self):
        bbox = self.axes.get_window_extent().transformed(self.figure.dpi_scale_trans.inverted())
        x, y, width, height = bbox.x0, bbox.y1, bbox.width, bbox.height
        width *= self.figure.dpi
        height *= self.figure.dpi
        x *= self.figure.dpi
        y *= self.figure.dpi
        
        pixmap = QtGui.QPixmap.grabWidget(self.canvas, int(x)+3, self.canvas.height() - int(y)+3, int(width)-5, int(height)-5)
        QtGui.QApplication.clipboard().setPixmap(pixmap)
        
    def export_axes_to_image(self):
        file_name = QtGui.QFileDialog.getSaveFileName(self, "Select file name", ".", "Image Files (*.png *.jpg *.pdf)")
        print file_name
        if file_name:
            extent = self.axes.get_window_extent().transformed(self.figure.dpi_scale_trans.inverted())
            self.figure.savefig(str(file_name), bbox_inches=extent)
        
    def sample_selection_changed(self, type_, idx):
        self.sample_selection = [True] * len(self.sample_selection)
        lkp_tmp = dict([v, u] for u, v in enumerate(['Plate', 'Well', 'Site', 'Treatment 1', 'Treatment 2', 'Group']))
        for cur_sel_type, cur_sel_box in self.sample_selection_dct.items():
            cur_sel = str(cur_sel_box.currentText())
            if cur_sel.startswith('All'):
                continue
            for i, s in enumerate(self.sample_ids):
                if not s[lkp_tmp[cur_sel_type]].startswith(cur_sel):
                    self.sample_selection[i] = False
                        
        self.update_axis()
        
    def result_selection_changed(self, type_):
        self.update_axis()
        
        
    def set_countour_eval_cb(self, contour_eval_func):
        self.contour_eval_func = contour_eval_func

        
    def selector_cb(self, epress, erelease):
        x_min, x_max = min(epress.xdata, erelease.xdata), max(epress.xdata, erelease.xdata)
        y_min, y_max = min(epress.ydata, erelease.ydata), max(epress.ydata, erelease.ydata)
        
        ind = numpy.array([True if ((x > x_min) and 
                                    (x < x_max) and
                                    (y > y_min) and 
                                    (y < y_max) and
                                    self.sample_selection[k]
                                    ) 
                                    else False for k, (x, y) in enumerate(self.xys)])
        
        edgecolors = self.collection.get_edgecolors()
        facecolors = self.collection.get_facecolors()
        
        ids =  [self.data[k].ref for k in range(len(ind)) if ind[k]] 
        print ind.sum(), 'objects selected with refs', ids
        tmp_idx = numpy.nonzero(self.sample_selection)[0]
        for i in range(len(self.xys[tmp_idx])):
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
        treat = [self.sample_ids[k] for k in range(len(ind)) if ind[k]]    
        img = self.image_generator_callback(treat, ids)
        self.image_changed.emit(img)
        self.selection_changed.emit(treat)
        
    def set_data(self, data_matrix, data_names, sample_ids, x_dim, y_dim, data_ch5_idx, image_generator_callback):
        self.image_generator_callback = image_generator_callback
        self.data_matrix = data_matrix
        self.data_names = data_names
        self.data_ch5_idx = data_ch5_idx
        self.sample_ids = sample_ids
        self.sample_selection = [True] * len(sample_ids) 
        
        self.data_mins, self.data_maxs = self.data_matrix.min(0), self.data_matrix.max(0) 
        
        self.fill_selection_boxes()
        
        assert self.data_matrix.shape[0] == len(self.sample_ids)
        assert self.data_matrix.shape[1] == len(self.data_names)
        
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
        
    def fill_selection_boxes(self):
        s = zip(*self.sample_ids)
        for i, s_name in enumerate(['Plate', 'Well', 'Site', 'Treatment 1', 'Treatment 2', 'Group']): 
            for each in numpy.unique(s[i]):
                self.sample_selection_dct[s_name].addItem(str(each))

         
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
        
        tmp_idx = numpy.nonzero(self.sample_selection)[0]
        self.xys = numpy.array([(d.x, d.y) for d in self.data])  

        if self.c_dim is None:
            facecolors = numpy.array([d.color for d in self.data])[tmp_idx]
            edgecolors = numpy.array([d.color for d in self.data])[tmp_idx]
        else:
            f_0_min = self.data_matrix[:, self.c_dim].min()
            f_0_max = self.data_matrix[:, self.c_dim].max()
            cNorm  = colors.Normalize(vmin=f_0_min, vmax=f_0_max)
            scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.jet)
            facecolors = [scalarMap.to_rgba(c) for c in self.data_matrix[tmp_idx, self.c_dim]]
            edgecolors = [scalarMap.to_rgba(c) for c in self.data_matrix[tmp_idx, self.c_dim]]
            
        
        
        for i in range(len(self.xys[tmp_idx])):
            if self.ind[i]:
                edgecolors[i] = DataPoint.colorin
            
        
        
        self.collection = CircleCollection(
            sizes=(22,),
            facecolors=facecolors,
            edgecolors=edgecolors,
            offsets = self.xys[tmp_idx],
            transOffset = self.axes.transData)

        self.axes.add_collection(self.collection)
        
        self.update_axis_lims()
        if self.contour_eval_func is not None:
            xx, yy, Z = self.contour_eval_func(self.axes.get_xlim(), self.axes.get_ylim(), self.x_dim, self.y_dim)
             
            self.axes.contourf(xx, yy, Z, levels=numpy.linspace(Z.min(), 0, 17), cmap=cm.Reds_r, alpha=0.2)
            self.axes.contour(xx, yy, Z, levels=[0], linewidths=1, colors='k')
            self.axes.contourf(xx, yy, Z, levels=numpy.linspace(0, Z.max(), 17), cmap=cm.Greens, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def update_axis_lims(self):
        f_0_min = self.data_mins[self.x_dim]
        f_0_max = self.data_maxs[self.x_dim]
        f_1_min = self.data_mins[self.y_dim]
        f_1_max = self.data_maxs[self.y_dim]
        
        self.axes.set_xlim(f_0_min, f_0_max)
        self.axes.set_ylim(f_1_min, f_1_max)
        self.axes.set_xlabel(self.data_names[self.x_dim])
        self.axes.set_ylabel(self.data_names[self.y_dim])
        
        self.axes.set_aspect(abs((f_0_max-f_0_min)/(f_1_max-f_1_min))/1)

        
        

class IScatterWidgetHisto(IScatterWidget):
    def __init__(self, *args, **kwargs):
        IScatterWidget.__init__(self, *args, **kwargs)
        self.cbar = None
        self.axis_c_chk.setVisible(False)
        self.axis_c_cmb.setVisible(False)
        self.axis_c_label.setVisible(False)
        
        self.current_cmap = 'red'
        
    def update_axis(self):
        self.axes.clear()
        self.data = [DataPoint(xy[0], xy[1], self.data_ch5_idx[i], False)
                     for i, xy in enumerate(self.data_matrix[:, [self.x_dim, self.y_dim]]) ]
        self.Nxy = len(self.data)
            
        self.xs = numpy.array([d.x for d in self.data])
        self.ys = numpy.array([d.y for d in self.data])
        self.xys = zip(self.xs, self.ys)
          
        tmp_idx = numpy.nonzero(self.sample_selection)[0]  
        
        if str(self.cmb_result_sample_selection.currentText()).startswith("Outliers"):
            image_list = []
            for k, cmap_name in zip([1, -1], ['red', 'green']):
                tmp_idx_2 = numpy.logical_and(self.sample_selection, self.data_matrix[:,-2] == k)
     
                hist_img = numpy.histogram2d(self.xs[tmp_idx_2], self.ys[tmp_idx_2], 
                                             bins=100, range=[[self.data_mins[self.x_dim], self.data_maxs[self.x_dim]],
                                                              [self.data_mins[self.y_dim], self.data_maxs[self.y_dim]]])
                img = numpy.flipud(hist_img[0].swapaxes(1,0))
                image_list.append(ImageWithCmap(img, generate_standard_palette(cmap_name)))

            img = blend_images(image_list)
            self.current_raw_images = image_list
            self.current_image = img
            
        elif str(self.cmb_result_sample_selection.currentText()).startswith("Classi"):
            image_list = []
            for k, cmap_name in zip(numpy.unique(self.data_matrix[:,-1]), ['green','red', 'blue', 'yellow', 'cyan', 'magenta', 'purple', 'olive']):
                tmp_idx_2 = numpy.logical_and(self.sample_selection, self.data_matrix[:,-1] == k)
                hist_img = numpy.histogram2d(self.xs[tmp_idx_2], self.ys[tmp_idx_2], 
                                             bins=100, range=[[self.data_mins[self.x_dim], self.data_maxs[self.x_dim]],
                                                             [self.data_mins[self.y_dim], self.data_maxs[self.y_dim]]])
                img = numpy.flipud(hist_img[0].swapaxes(1,0))
                image_list.append(ImageWithCmap(img, generate_standard_palette(cmap_name)))
    
            img = blend_images(image_list)
            self.current_raw_images = image_list
            self.current_image = img
            
        elif str(self.cmb_result_sample_selection.currentText()).startswith("Clustering"):
            image_list = []
            for k, cmap_name in zip(range(4), ['green','red', 'yellow', 'blue', 'magenta',  'cyan', 'purple']):
                tmp_idx_2 = numpy.logical_and(self.sample_selection, self.data_matrix[:,-3] == k)
                hist_img = numpy.histogram2d(self.xs[tmp_idx_2], self.ys[tmp_idx_2], 
                                             bins=100, range=[[self.data_mins[self.x_dim], self.data_maxs[self.x_dim]],
                                                             [self.data_mins[self.y_dim], self.data_maxs[self.y_dim]]])
                img = numpy.flipud(hist_img[0].swapaxes(1,0))
                image_list.append(ImageWithCmap(img, generate_standard_palette(cmap_name)))
    
            img = blend_images(image_list)
            self.current_raw_images = image_list
            self.current_image = img
        else:
            hist_img = numpy.histogram2d(self.xs[tmp_idx], self.ys[tmp_idx], bins=100, range=[[self.data_mins[self.x_dim], self.data_maxs[self.x_dim]],
                                                                                             [self.data_mins[self.y_dim], self.data_maxs[self.y_dim]]])
            
            img = numpy.flipud(hist_img[0].swapaxes(1,0))
            self.current_raw_images = [ImageWithCmap(img, self.current_cmap)]
    
            img = blend_images(self.current_raw_images)
            self.current_image = img
            
        
        aximg = self.axes.imshow(img, interpolation='nearest', extent=(self.data_mins[self.x_dim], self.data_maxs[self.x_dim],
                                                                            self.data_mins[self.y_dim], self.data_maxs[self.y_dim]),)
        
#         self.axes.xaxis.set_ticks_position('none') 
        for tt in self.axes.xaxis.get_ticklines(): tt.set_visible(False) 
        for tt in self.axes.yaxis.get_ticklines(): tt.set_visible(False) 
        #self.axes.axis('off')
        self.axes.set_frame_on(False)

        self.update_axis_lims()
        
        self.figure.tight_layout()
        self.canvas.draw()     
        
    def selector_cb(self, epress, erelease):
        x_min, x_max = min(epress.xdata, erelease.xdata), max(epress.xdata, erelease.xdata)
        y_min, y_max = min(epress.ydata, erelease.ydata), max(epress.ydata, erelease.ydata)
        
        print x_min, x_max
        print y_min, y_max
        
        ind = numpy.array([True if ((x > x_min) and 
                                    (x < x_max) and
                                    (y > y_min) and 
                                    (y < y_max) and
                                    self.sample_selection[k]
                                    ) 
                                    else False for k, (x, y) in enumerate(self.xys)])
                
        self.canvas.draw()  
        ids =  [self.data[k].ref for k in range(len(self.xys)) if ind[k]] 
        print ind.sum(), 'objects selected with refs', ids
        self.update_image(ind) 

class SimpleMplImageViewer(QtGui.QWidget):
    def __init__(self, parent=None):   
        QtGui.QWidget.__init__(self, parent)
        self.figure = Figure()
        self.figure.patch.set_alpha(0)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(1, 1, 1, axisbg='grey')
        self.axes.axis('off')
        self.axes.set_frame_on(False)
        
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self.canvas)
        layout.setAlignment(QtCore.Qt.AlignLeft)
        self.setLayout(layout)
        self.rects = []
        self.normalize = False
  
    def show_image(self, img):
        print type(img), img.shape
        imgplot = self.axes.imshow(img, cm.Greys_r)
        if not self.normalize:
            imgplot.set_clim(0, 255)
        
        self.canvas.draw()  
        
    def highlight_cell(self, entries):
        for r in self.rects:
            r.remove()
        del self.rects
        self.rects = []
        for e in entries:
            m, q = e / 25, e % 25
            rect = self.axes.add_patch(Rectangle((q*60, m*60), 60, 60, facecolor='none', edgecolor="red"))
            self.rects.append(rect)
        self.canvas.draw()
        
class SimpleImageViewer(QtGui.QWidget):
    def __init__(self, parent=None):   
        QtGui.QWidget.__init__(self, parent)
        self.normalize = False
        
        self.image_pane = QtGui.QLabel('Image')
        self.image_pane.setScaledContents(False)
        self.image_pane.setMaximumWidth(1000)
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self.image_pane)
        self.setLayout(layout)
        
    def show_image(self, img):
        pixmap = QtGui.QPixmap(qimage2ndarray.gray2qimage(img, self.normalize))
        scaledPixmap = pixmap.scaled(self.image_pane.size(), QtCore.Qt.KeepAspectRatio)
        self.image_pane.setPixmap(scaledPixmap)
        self.update() 
        
    def highlight_cell(self, entries):
        pass

class SimpleMplImageViewerWithBlending(SimpleMplImageViewer):
    colors = OrderedDict([("red" , '#FF0000'),
                          ("green" , '#00FF00'),
                          ("blue" , '#0000FF'),
                          ("yellow" , '#FFFF00'),
                          ("magenta" ,'#FF00FF'),
                          ("cyan" , '#00FFFF'),
                          ("purple" , '#800080'),
                          ("olive" , '#808000'),
                          ("white" , '#FFFFFF')])
    
    


    def __init__(self, *args, **kwargs):
        SimpleMplImageViewer.__init__(self, *args, **kwargs)
        layout = self.layout()
        
        self._image_list = []
        
        btn_widget = QtGui.QWidget(self)
        btn_layout = QtGui.QVBoxLayout(btn_widget)
        layout.addWidget(btn_widget)
        
        self.btn_clear = QtGui.QPushButton('Clear')
        btn_layout.addWidget(self.btn_clear)
        self.btn_clear.clicked.connect(self.clear_image)
        
        self.btn_clear = QtGui.QPushButton('Image list')
        self.cmb_image_list = QtGui.QComboBox(self)
        self.cmb_image_list.currentIndexChanged[int].connect(self.image_selector_changed)
        btn_layout.addWidget(self.cmb_image_list)
        
        tmp_widget = QtGui.QWidget(self)
        tmp_layout = QtGui.QHBoxLayout(tmp_widget)
        
        self.cmb_norm_min = QtGui.QDoubleSpinBox(self)
        self.cmb_norm_max = QtGui.QDoubleSpinBox(self)
        
        self.cmb_norm_min.setDecimals(3)
        self.cmb_norm_min.setSingleStep(0.01)
        
        self.cmb_norm_max.setDecimals(3)
        self.cmb_norm_max.setSingleStep(0.001)
        
        
        self.cmb_norm_max.setMaximum(1)
        self.cmb_norm_max.setValue(0.01)
        
        self.btn_norm_reset = QtGui.QPushButton('Reset')
        self.btn_norm_reset.clicked.connect(self.norm_reset)
        
        self.btn_del_img = QtGui.QPushButton('Del')
        self.btn_del_img.clicked.connect(self.remove_image)
        
        self.cmb_norm_min.setValue(0)
        self.cmb_norm_max.setValue(1)
        
        tmp_layout.addWidget(QtGui.QLabel('Min'))
        tmp_layout.addWidget(self.cmb_norm_min)
        tmp_layout.addWidget(QtGui.QLabel('Max'))
        tmp_layout.addWidget(self.cmb_norm_max)   
        tmp_layout.addWidget(self.btn_norm_reset)     
        tmp_layout.addWidget(self.btn_del_img)     
        
        self.cmb_norm_min.valueChanged.connect(self.norm_min_changed)
        self.cmb_norm_max.valueChanged.connect(self.norm_max_changed)
        
        btn_layout.addWidget(tmp_widget)
        
        self.export_to_image_btn = QtGui.QPushButton('Export image File')
        btn_layout.addWidget(self.export_to_image_btn)
        self.export_to_image_btn.clicked.connect(self.export_axes_to_image)
        
        self.export_to_cp_btn = QtGui.QPushButton('Export image clipboard')
        btn_layout.addWidget(self.export_to_cp_btn)
        self.export_to_cp_btn.clicked.connect(self.export_axes_to_clipboard)
        btn_layout.addStretch()
        
    def norm_reset(self):
        current_img_idx = self.cmb_image_list.currentIndex()
        current_img = self._image_list[current_img_idx]
        
        current_img.reset_scaling()
        self.cmb_norm_min.setValue(current_img.norm_min)
        self.cmb_norm_max.setValue(current_img.norm_max)
        
    def remove_image(self):
        current_img_idx = self.cmb_image_list.currentIndex()
        del self._image_list[current_img_idx]
        self.show_image_with_colormap([], normalize_to_max=False)
        
    def image_selector_changed(self, index):
        img = self._image_list[index]
        self.cmb_norm_min.setValue(img.norm_min)
        self.cmb_norm_max.setValue(img.norm_max)
        
    def norm_min_changed(self, value):
        current_img_idx = self.cmb_image_list.currentIndex()
        current_img = self._image_list[current_img_idx]
        current_img.norm_min =  value
        self.show_image_with_colormap([], normalize_to_max=False)
    
    def norm_max_changed(self, value):
        current_img_idx = self.cmb_image_list.currentIndex()
        current_img = self._image_list[current_img_idx]
        current_img.norm_max =  value
        self.show_image_with_colormap([], normalize_to_max=False)
    
    
    def export_axes_to_clipboard(self):
        bbox = self.axes.get_window_extent().transformed(self.figure.dpi_scale_trans.inverted())
        x, y, width, height = bbox.x0, bbox.y1, bbox.width, bbox.height
        width *= self.figure.dpi
        height *= self.figure.dpi
        x *= self.figure.dpi
        y *= self.figure.dpi
        
        pixmap = QtGui.QPixmap.grabWidget(self.canvas, int(x)+3, self.canvas.height() - int(y)+3, int(width)-5, int(height)-5)
        QtGui.QApplication.clipboard().setPixmap(pixmap)
        
    def export_axes_to_image(self):
        file_name = QtGui.QFileDialog.getSaveFileName(self, "Select file name", ".", "Image Files (*.png *.jpg *.pdf)")
        print file_name
        if file_name:
            extent = self.axes.get_window_extent().transformed(self.figure.dpi_scale_trans.inverted())
            self.figure.savefig(str(file_name), bbox_inches=extent)
        

    def clear_image(self):
        self._image_list = []
        self._image_norms = [] 
        self.cmb_image_list.clear()
        self.axes.clear()
        self.axes.axis('off')
        self.axes.set_frame_on(False)
        self.canvas.draw()
        
    
    def show_image_with_colormap(self, img_list, normalize_to_max=True):
        self._image_norms = [] 
        
        if normalize_to_max:
            max_max = numpy.max([img_.max() for img_ in img_list])
            for img in img_list:
                img.norm_max = max_max
            
        for _, img in enumerate(img_list):
            self._image_list.append(img)
            self.cmb_image_list.addItem('%s %d' % (img.name, len(self._image_list)))    
        
        self.axes.imshow(blend_images(self._image_list), interpolation='nearest')
        self.axes.tick_params(color="none")

        self.canvas.draw()
    


def start_qt_event_loop():
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    rcParams['pdf.fonttype'] = 42
    # rcParams['axes.linewidth'] = rcParams['axes.linewidth']*2
    # rcParams['legend.numpoints'] = 1
    # rcParams['legend.markerscale'] = 0
    # rcParams['xtick.major.width'] = 2
    # rcParams['ytick.major.width'] = 2
    rcParams['text.color'] = 'white'
    rcParams['xtick.color'] = 'white'
    rcParams['ytick.color'] = 'white'
    rcParams['ytick.labelsize'] = 14
    rcParams['xtick.labelsize'] = 14
    rcParams['axes.labelsize'] = 18
    
    rcParams['axes.labelcolor'] = 'white'
    app = QtGui.QApplication(sys.argv)
    app.setStyle("plastique")     
    return app

def ch5_scatter(ch5_file,time=0):
    app = start_qt_event_loop()
    ch5_file = cellh5.CH5File(ch5_file, "r")
    
    plate = "Plate"
    all_features = []
    all_sample_names = []
    all_data_names = ch5_file.object_feature_def() + ['Time', 'Center x', 'Center y', 'Classificaiton']
    all_times = []
    for ch5_pos in ch5_file.iter_positions():
        well = ch5_pos.well
        site = ch5_pos.pos

        features = ch5_pos.get_object_features()
        times = ch5_pos.get_all_time_idx()
        times_idx = times > time
        features = features[times_idx, :]
        
        center_x = ch5_pos.get_center(times_idx)['x']
        center_y = ch5_pos.get_center(times_idx)['y']
        predictions = ch5_pos.get_class_prediction()[times_idx]['label_idx']
        
        features = numpy.c_[features, times[times_idx], center_x, center_y, predictions]
        
        all_features.append(features)
        
        sample_names = [(plate, well, str(site), well, str(site), 'pos') for k in xrange(features.shape[0])]
        all_sample_names.append(sample_names)
        
        all_times.append(times_idx)
    
    all_features = numpy.concatenate(all_features)
    all_sample_names = numpy.concatenate(all_sample_names)
    all_times = numpy.concatenate(all_times)
    
    iscatter = IScatterWidgetHisto()
    iscatter.set_data(all_features, all_data_names, all_sample_names, 222, 236, numpy.nonzero(all_times)[0], lambda y, x: ch5_pos.get_gallery_image_matrix(x, (12, 32)))

    iscatter2 = IScatterWidget()
    iscatter2.set_data(all_features, all_data_names, all_sample_names, 222, 236, numpy.nonzero(all_times)[0], lambda y, x: ch5_pos.get_gallery_image_matrix(x, (12, 32)))
    
    mw = IScatter(iscatter, iscatter2)
    mw.show()
    app.exec_()
    
if __name__ == "__main__":
    ch5_scatter("C:/Users/sommerc/cellh5/data/_all_positions.ch5")