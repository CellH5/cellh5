import numpy
import matplotlib
import pylab

def spreadplot(y, x=None, spread=0.3, spread_type='u', colors=None, marker='.', xticklabels=None):
    if x is None:
        x = range(len(y))
    if colors is None:
        colors = ['k' for _ in xrange(len(y))]
        
    for i, (y_data, x_center) in enumerate(zip(y,x)):
        
        if spread_type == 'u':
            rand_gen = lambda tmp: (numpy.random.rand(tmp)-0.5)*spread
        elif spread_type == 'g':
            rand_gen = lambda tmp: (numpy.random.randn(tmp))*spread
        else:
            raise AttributeError('Unknown spread type. Choose u or g...')
        
        x_data = rand_gen(len(y_data)) + x_center
        pylab.plot(x_data, y_data, color=colors[i], marker=marker, linestyle='None')
        
        ax = pylab.gca()
        ax.set_xticks(x)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels, rotation=90)
    return ax

def line_errbar_plot(y, x=None, color=None, marker='.', xticklabels=None, *args, **kwargs):
    if x is None:
        x = range(len(y))
    if color is None:
        color = 'k'
        
    y_means = []
    y_stds = []
    for i, (y_data, x_center) in enumerate(zip(y, x)):
        y_means.append(numpy.array(y_data).mean())
        y_stds.append(numpy.array(y_data).std())
        
    pylab.errorbar(x, y_means, yerr=y_stds, color=color, *args, **kwargs)
        
    ax = pylab.gca()
    ax.set_xticks(x)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, rotation=90)
    return ax




class Test_line_errbar_plot(object):
    def __init__(self):
        self.x = [numpy.random.randn(40) + u for u in range(10)]
        
    def test_1(self):
        ax = line_errbar_plot(self.x)
        pylab.show()
        
    def test_2(self):
        ax = line_errbar_plot(self.x, color='r')
        pylab.show()
        
    def test_3(self):
        ax = line_errbar_plot(self.x, color='r', ecolor='k')
        pylab.show()
        
class Test_spreadplot(object):
    def __init__(self):
        self.x = [range(10), range(5), range(5,15), numpy.arange(2,20,0.2)]
        
    def test_1(self):
        ax = spreadplot(self.x)
        pylab.show()
        
    def test_2(self):
        ax = spreadplot(self.x, spread=0.1, spread_type='g', colors=('g', 'r', 'k', 'b'))
        pylab.show()
    

if __name__ == "__main__":
    t = Test_line_errbar_plot()
    t.test_1()
    t.test_2()
    t.test_3()
    
    t = Test_spreadplot()
    t.test_1()
    t.test_2()