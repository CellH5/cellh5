import numpy
import matplotlib

def split_str_into_len(s, l=2):
    """ Split a string into chunks of length l """
    return [s[i:i+l] for i in range(0, len(s), l)]

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
                
cmap3 = matplotlib.colors.ListedColormap(map(lambda x: hex_to_rgb(x), 
                                                   ['#FFFFFF', 
                                                    '#AAAAAA', 
                                                    '#0000FF',
                                                    '#FF0000']), 'classification_cmap')
cmap53 = matplotlib.colors.ListedColormap(map(lambda x: hex_to_rgb(x), 
                                                   ['#FFFFFF', 
                                                    '#AAAAAA', 
                                                    '#0000FF',
                                                    '#FF0000']), 'classification_cmap')     

CMAP17_MULTI = matplotlib.colors.ListedColormap(map(lambda x: hex_to_rgb(x), 
                                                   ['#FFFFFF',
                                                     
                                                    '#E31A1C', 
                                                    '#FD8D3C',
                                                    '#FECC5C',
                                                    '#FFFFB2',
                                                    
                                                    '#238443', 
                                                    '#78C679',
                                                    '#C2E699',
                                                    '#FFFFCC',
                                                    
                                                    '#2171B5', 
                                                    '#6BAED6',
                                                    '#BDD7E7',
                                                    '#EFF3FF',
                                                    
                                                    '#238443', 
                                                    '#78C679',
                                                    '#C2E699',
                                                    '#FFFFCC',
                                                    
                                                    '#000000',
                                                    
                                                    ]), 'cmap17_2')    

CMAP17 = matplotlib.colors.ListedColormap(map(lambda x: hex_to_rgb(x), 
                                                   ['#FFFFFF', 
                                                    '#AAAAAA', 
                                                    '#0000FF',
                                                    '#0000FF',
                                                    '#0000FF',
                                                    '#AAAAAA', 
                                                    '#0000FF',
                                                    '#0000FF',
                                                    '#0000FF',
                                                    '#AAAAAA', 
                                                    '#0000FF',
                                                    '#0000FF',
                                                    '#0000FF',
                                                    '#AAAAAA', 
                                                    '#0000FF',
                                                    '#0000FF',
                                                    '#0000FF',
                                                    '#FF0000',
                                                    '#00FF00']), 'cmap17')   

CMAP17SIMPLE = matplotlib.colors.ListedColormap(map(lambda x: hex_to_rgb(x), 
                                                   ['#FFFFFF', 
                                                    '#AAAAAA', 
                                                    '#FFFFFF',
                                                    '#FFFFFF',
                                                    '#FFFFFF',
                                                    '#FFFFFF', 
                                                    '#FFFFFF',
                                                    '#FFFFFF',
                                                    '#FFFFFF',
                                                    '#FFFFFF', 
                                                    '#FFFFFF',
                                                    '#FFFFFF',
                                                    '#FFFFFF',
                                                    '#FFFFFF', 
                                                    '#FFFFFF',
                                                    '#FFFFFF',
                                                    '#FFFFFF',
                                                    '#FFFFFF',
                                                    '#FFFFFF']), 'cmap17') 

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
from itertools import tee, izip
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


from collections import defaultdict
import pylab

from matplotlib import rcParams
import matplotlib
def setupt_matplot_lib_rc():
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.size'] *= 2
    rcParams['font.sans-serif'] = ['Arial']
    #matplotlib.rc('xtick', labelsize=14) 
    #matplotlib.rc('ytick', labelsize=14) 
    #matplotlib.rc('axes', labelsize=14) 
    matplotlib.rc('legend',**{'fontsize':18})
    rcParams['axes.linewidth'] = rcParams['axes.linewidth']*2
    rcParams['legend.numpoints'] = 1
    #rcParams['legend.markerscale'] = 0
    rcParams['xtick.major.width'] = 2
    rcParams['ytick.major.width'] = 2
    rcParams['pdf.fonttype'] = 42
    rcParams['xtick.major.pad']= 8
    rcParams['ytick.major.pad']= 8


class ConcentrationLine(object):
    def __init__(self, name, wells, labels, source_file, color, has_yerr=True):
        self.wells = wells
        self.name = name
        self.labels = labels
        self.color = color
        self.has_yerr = has_yerr
        self.conc_unit = "nM"
        
        self.source_file = source_file
        self.mean_values = {}
        self.std_values = {}
        self.raw_values = {}
        self.fate_classes = {}
        self.read_values()
        
        self.legend_ncol = 1
        self.legend_loc = 2
        self.legend_draw_frame = True
        
    def mean_functor(self, values):
        return numpy.mean(values)
    
    def std_functor(self, values):
        return numpy.std(values)
        
    def read_values(self):
        v = numpy.recfromcsv(self.source_file, delimiter='\t', filling_values=-1, case_sensitive=True)
        for p in self.wells:
            try:
                tmp = v[p+"_01"]
            except:
                tmp =  v[p] 
            tmp = tmp[numpy.logical_not(tmp < 0)]
            self.mean_values[p] = self.mean_functor(tmp)
            self.std_values[p] = self.std_functor(tmp)
            self.raw_values[p] = tmp
                
    def get_means(self):
        return [self.mean_values[w] for w in self.wells]
    
    def get_raws(self):
        return [self.raw_values[w] for w in self.wells]
    
    def set_concentration_unit(self, unit):
        self.conc_unit = unit
    
    def get_std(self):
        return [self.std_values[w] for w in self.wells]
    
    def plot_error_bar(self, ax, **options):
        values = self.get_means() 
        if self.has_yerr:
            ax.errorbar(range(len(values)), values, yerr=self.get_std(), color=self.color, fmt='o-', label=self.name, markeredgecolor='none', zorder=1, lw=2, **options)
        else:
            ax.errorbar(range(len(values)), values, color=self.color, fmt='o-', label=self.name, markeredgecolor='none', zorder=1, lw=2, **options)
        
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(self.labels, rotation=90)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        # add lines
        handles, labels = ax.get_legend_handles_labels()
        # remove the errorbars
        handles = [h[0] for h in handles]
        lg = pylab.legend(handles, labels, loc=self.legend_loc, ncol=self.legend_ncol)
        lg.draw_frame(self.legend_draw_frame)
        pylab.xlabel('Concentration (%s)' % self.conc_unit)
        pylab.ylabel('Mitotic duration (min)')
        pylab.tight_layout()
        
        
class ColoredConcentrationTimingSpread(ConcentrationLine):
    gauss_spread = 0.05
    
    def read_fate_class(self, file_name):
        v = numpy.recfromcsv(file_name, delimiter='\t', filling_values=-1, case_sensitive=True)
        for p in self.wells:
            try:
                tmp = v[p+"_01"]
            except:
                tmp =  v[p] 
            tmp = tmp[numpy.logical_not(numpy.isnan(tmp))]
            self.fate_classes[p] = tmp
    
    def plot_spread(self, ax, **options):
        values = numpy.array(self.get_raws())
        x_spread = []
        
        for v_idx, v in enumerate(values):
            x_spread.append(numpy.random.randn(len(v)) * self.gauss_spread + v_idx)
            
        fates = [self.fate_classes[w] for w in self.wells]
        
        color_table = {0:'b', 
                       1:'g',
                       2:'r', 
                       }
        fate_lookup = {
                       0: (0,1,2,4,), 
                       1: (3,),
                       2: (5,),
                       }
        width=0.25

        rects = []
        for ind, (x, y, f) in enumerate(zip(x_spread, values, fates)):
            ax.vlines(ind-0.5, 0, 2000, alpha=0.5, linestyle=':')
            f_ = f[:len(y)]
            for wi in range(3):
                y_ = y[(numpy.in1d(f, fate_lookup[wi])).nonzero()[0]]
                x_ = x[(numpy.in1d(f, fate_lookup[wi])).nonzero()[0]]
                rect = ax.scatter(x_+((wi-1)*width), y_, color=color_table[wi], s=20, marker='.', )#facecolor='none')
            
                rects.append(rect)
                
        lg = ax.legend(rects, ('Mitosis - live interphase', 'Mitosis - death in interphase', 'Mitosis - death in mitosis'), 
                       loc=self.legend_loc, 
                       ncol=self.legend_ncol,
                       bbox_to_anchor=(-0.1, -0.3)
                       )
        lg.draw_frame(False)
        ax.set_xlim(-0.5,ind+.5)
                
        ax.set_xticks(numpy.arange(len(values)))
        ax.set_xticklabels(self.labels, rotation=90)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        
        
        pylab.xlabel('Concentration (%s)' % self.conc_unit)
        pylab.ylabel('Mitotic duration (min)')
             
            
class ColoredConcentrationTimingBar(ColoredConcentrationTimingSpread):
    def plot_bar(self, ax, **options):
        def autolabel(rects, txt):
            # attach some text labels
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x()+rect.get_width()/2., height+30, txt,
                ha='center', va='bottom', size=14)
        
        
        values = numpy.array(self.get_raws())
        x_spread = []
        
        for v_idx, v in enumerate(values):
            x_spread.append(v_idx)
            
        fates = [self.fate_classes[w] for w in self.wells]
        
        color_table = {0:'b', 
                       1:'g',
                       2:'r', 
                       }
        fate_lookup = {
                       0: (0,1,2,4,), 
                       1: (3,),
                       2: (5,),
                       }
        
        width=0.25
        rects = []
        for x, y, f in zip(x_spread, values, fates):
            f_ = f[:len(y)]
            for wi in range(3):
                y_ = y[(numpy.in1d(f,fate_lookup[wi])).nonzero()[0]]
                if len(y_) > 0:
                    m = numpy.mean(y_)
                    s= numpy.std(y_)
                else:
                    m = 0
                    s = 0
                rect = ax.bar(x+(wi*width), m, width, color=color_table[wi], yerr=s, ecolor='k')
                autolabel(rect, len(y_))
                rects.append(rect)
                
        lg = ax.legend(rects, ('Mitosis - live interphase', 'Mitosis - death in interphase', 'Mitosis - death in mitosis'), 
                       loc=self.legend_loc, 
                       ncol=self.legend_ncol,
                       bbox_to_anchor=(-0.1, -0.3)
                       )
        lg.draw_frame(False)
                
        ax.set_xticks(numpy.arange(len(values))+3*width/2.0)
        ax.set_xticklabels(self.labels, rotation=90)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        
        pylab.xlabel('Concentration (%s)' % self.conc_unit)
        pylab.ylabel('Mitotic duration (min)')
        
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
                   loc=self.legend_loc, 
                   ncol=self.legend_ncol,
                   bbox_to_anchor=(-0.1, -0.3)
                   )
    lg.draw_frame(False)
            
    #ax.set_xticks(numpy.arange(len(fates))+width/2.0)
    ax.set_xticklabels(labels, rotation=90)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    #ax.set_xlim(-0.2,len(fates)-0.35)
    
    pylab.xlabel('Treatment')
    pylab.ylabel('Cluster')
    
class ConcentrationStackedBar(ColoredConcentrationTimingSpread):    
    def plot_stacked_fate_bar(self, ax, **options):
        def autolabel(rects, txt):
            # attach some text labels
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x()+rect.get_width()/2., height+30, txt,
                ha='center', va='bottom', size=14)
        
        x_spread = []
        fates = [self.fate_classes[w] for w in self.wells]
        for v_idx, v in enumerate(fates):
            x_spread.append(v_idx)
        
        color_table = {0:'b', 1:'#00CD00', 2:'r',}
        fate_lookup = {
                       0: (0,1,2,4,), 
                       1: (3,),
                       2: (5,),
                       }
        
        width=0.9
        rects = []
        
        for x, f in zip(x_spread, fates):
            hs = []
            for wi in range(3):
                h = len((numpy.in1d(f, fate_lookup[wi])).nonzero()[0])
                hs.append(h)
                
            hs = numpy.array(hs).astype(numpy.float32)
            hs /= hs.sum()
            bottom=0
            for wi in range(3):
                rect = ax.bar(x, hs[wi], width, bottom=bottom, color=color_table[wi], edgecolor = "none")
                bottom+=hs[wi]
                #autolabel(rect, len(y_))
                rects.append(rect)
                
        lg = ax.legend(rects, ('Mitosis - live interphase', 'Mitosis - death in interphase', 'Mitosis - death in mitosis'), 
                       loc=self.legend_loc, 
                       ncol=self.legend_ncol,
                       bbox_to_anchor=(-0.1, -0.3)
                       )
        lg.draw_frame(False)
                
        ax.set_xticks(numpy.arange(len(fates))+width/2.0)
        ax.set_xticklabels(self.labels, rotation=90)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlim(-width/2.0,len(fates)+width/2.0)
        
        pylab.xlabel('Concentration (%s)' % self.conc_unit)
        pylab.ylabel('Frequency (%)')
            
        
             
                
if __name__ == "__main__":
    setupt_matplot_lib_rc()
        
    wells = ["G07", "B06", "C06", "D06", "E06", "F06", "G06", "B07", "C07", "D07", "E07", "F07"]
    name = 'Taxol'
    labels = ["%s" % c for c in ['0', '1', '2.5'] + map(str, [5,25,50,100,200,400,800,1600,3200])]
    color = 'r'

    ax = pylab.gca()
    
    cl = ConcentrationLine(name, wells, labels, 'mito_timing_2307.txt', color)
    cl2 = ConcentrationLine('blub', wells, labels, 'mito_timing_2308.txt', 'b', has_yerr=False)
    cl2.mean_functor = lambda x: x[0] / float(x[0]) *20
    cl2.read_values()
    print cl2.get_means()
    
    cl.plot_error_bar(ax)
    cl2.plot_error_bar(ax)
    pylab.show()
    
    
    
    
    




