from fate_utils import (ConcentrationLine, 
                        ColoredConcentrationTimingSpread, 
                        ColoredConcentrationTimingBar,
                        ConcentrationStackedBar,
                        setupt_matplot_lib_rc)
import pylab
setupt_matplot_lib_rc()

def boehringer_concentration_cells_dying_in():
    # BI
    pylab.figure()
    mt_fle = "__mito_timing_old.txt"
    
    BI_no  =  ["D11", "B04", "C04", "D04", "E04",]
    BI_mad  = ["D11", "B07", "C07", "D07", "E07",]
    BI_plk  = ["D11", "B10", "C10", "D10", "E10",]
    
    cons_nm = ["%s" % c for c in ['0',] + map(str, [1,5,25,125])]
    ax = pylab.gca()
    bi_1 = ConcentrationLine("BI2536 no siRNA", BI_no , cons_nm, mt_fle, "g")
    bi_2 = ConcentrationLine("BI2536 siMad2", BI_mad, cons_nm, mt_fle, "b")
    bi_3 = ConcentrationLine("BI2536 siPlk1", BI_plk, cons_nm, mt_fle, "r")
    bi_1.plot_error_bar(ax, marker='o')
    bi_2.plot_error_bar(ax, marker='d')
    bi_3.plot_error_bar(ax, marker='s')
    ax.set_ylim(0,1200)
    pylab.savefig('mito_timing_BI2536.pdf')
    
    # Taxol
    pylab.figure()
    mt_fle = "__mito_timing.txt"
    
    TA_no  =  ["D11", "C03", "D03", "E03", "F03",]
    TA_mad  = ["D11", "C06", "D06", "E06", "F06",]
    TA_plk  = ["D11", "C09", "D09", "E09", "F09",]
    
    cons_nm = ["%s" % c for c in ['0',] + map(str, [10,100,1000, 10000])]
    
    ta_1 = ConcentrationLine("Taxol no siRNA", TA_no , cons_nm, '__mito_timing.txt', "g")
    ta_2 = ConcentrationLine("Taxol siMad2", TA_mad, cons_nm, '__mito_timing.txt', "b")
    ta_3 = ConcentrationLine("Taxol siPlk1", TA_plk, cons_nm, '__mito_timing.txt', "r")
    ax = pylab.gca()
    ta_1.plot_error_bar(ax, marker='o')
    ta_2.plot_error_bar(ax, marker='d')
    ta_3.plot_error_bar(ax, marker='s')
    ax.set_ylim(0,1200)
    pylab.savefig('mito_timing_Taxol.pdf')
    
    # Nocodazole
    pylab.figure()
    mt_fle = "__mito_timing.txt"
    
    NO_no  =  ["D11", "C02", "D02", "E02", "F02", "G02", "B03",]
    NO_mad  = ["D11", "C05", "D05", "E05", "F05", "G05", "B06",]
    NO_plk  = ["D11", "C08", "D08", "E08", "F08", "G08", "B09",]
    
    cons_nm = ["%s" % c for c in ['0',] + map(str, [12,25,50, 100, 200, 400])]
    
    ax = pylab.gca()
    no_1 = ConcentrationLine("Nocodazole no siRNA", NO_no , cons_nm, '__mito_timing.txt', "g")
    no_2 = ConcentrationLine("Nocodazole siMad2", NO_mad, cons_nm, '__mito_timing.txt', "b")
    no_3 = ConcentrationLine("Nocodazole siPlk1", NO_plk, cons_nm, '__mito_timing.txt', "r")
    
    no_1.set_concentration_unit('ng/ml')
    no_2.set_concentration_unit('ng/ml')
    no_3.set_concentration_unit('ng/ml')
    
    no_1.plot_error_bar(ax, marker='o')
    no_2.plot_error_bar(ax, marker='d')
    no_3.plot_error_bar(ax, marker='s')
    ax.set_ylim(0,1200)
    pylab.savefig('mito_timing_Nocodazole.pdf')
    
    # BI dying in Mitosis
    pylab.figure()
    
    BI_no  =  ["D11", "B04", "C04", "D04", "E04",]
    BI_mad  = ["D11", "B07", "C07", "D07", "E07",]
    BI_plk  = ["D11", "B10", "C10", "D10", "E10",]
    
    cons_nm = ["%s" % c for c in ['0',] + map(str, [1,5,25,125])]
    ax = pylab.gca()
    bi_1 = ConcentrationLine("BI2536 no siRNA", BI_no , cons_nm, '__dyinging_in_mito_or_apo.txt', "g", has_yerr=False)
    bi_2 = ConcentrationLine("BI2536 siMad2", BI_mad, cons_nm, '__dyinging_in_mito_or_apo.txt', "b", has_yerr=False)
    bi_3 = ConcentrationLine("BI2536 siPlk1", BI_plk, cons_nm, '__dyinging_in_mito_or_apo.txt', "r", has_yerr=False)
    
    div = lambda x: ((x[0] / float(x[2])) * 100.0)
    bi_1.mean_functor = div
    bi_1.read_values()
    bi_2.mean_functor = div
    bi_2.read_values()
    bi_3.mean_functor = div
    bi_3.read_values()
    
    bi_1.plot_error_bar(ax, marker='o')
    bi_2.plot_error_bar(ax, marker='d')
    bi_3.plot_error_bar(ax, marker='s')
    ax.set_ylim(0,100)
    ax.set_ylabel("Cells dying in mitosis (%)")
    pylab.savefig('dying_mitosis_BI2536.pdf')
    
    pylab.figure()
    ax = pylab.gca()
    div = lambda x: ((x[1] / float(x[2])) * 100.0)
    bi_1.mean_functor = div
    bi_1.read_values()
    bi_2.mean_functor = div
    bi_2.read_values()
    bi_3.mean_functor = div
    bi_3.read_values()
    
    bi_1.plot_error_bar(ax, marker='o')
    bi_2.plot_error_bar(ax, marker='d')
    bi_3.plot_error_bar(ax, marker='s')
    ax.set_ylim(0,100)
    ax.set_ylabel("Cells dying in interphase (%)")
    pylab.savefig('dying_interphase_BI2536.pdf')
    
    # Taxol dying
    pylab.figure()
    
    TA_no  =  ["D11", "C03", "D03", "E03", "F03",]
    TA_mad  = ["D11", "C06", "D06", "E06", "F06",]
    TA_plk  = ["D11", "C09", "D09", "E09", "F09",]
    
    cons_nm = ["%s" % c for c in ['0',] + map(str, [10,100,1000, 10000])]
    
    ta_1 = ConcentrationLine("Taxol no siRNA", TA_no , cons_nm, '__dyinging_in_mito_or_apo.txt', "g", has_yerr=False)
    ta_2 = ConcentrationLine("Taxol siMad2", TA_mad, cons_nm, '__dyinging_in_mito_or_apo.txt', "b", has_yerr=False)
    ta_3 = ConcentrationLine("Taxol siPlk1", TA_plk, cons_nm, '__dyinging_in_mito_or_apo.txt', "r", has_yerr=False)
    div = lambda x: ((x[0] / float(x[2])) * 100.0)
    ta_1.mean_functor = div
    ta_1.read_values()
    ta_2.mean_functor = div
    ta_2.read_values()
    ta_3.mean_functor = div
    ta_3.read_values()
    
    ax = pylab.gca()
    ta_1.plot_error_bar(ax, marker='o')
    ta_2.plot_error_bar(ax, marker='d')
    ta_3.plot_error_bar(ax, marker='s')
    ax.set_ylim(0,100)
    ax.set_ylabel("Cells dying in mitosis (%)")
    pylab.savefig('dying_mitosis_Taxol.pdf')
    
    pylab.figure()
    div = lambda x: ((x[1] / float(x[2])) * 100.0)
    ta_1.mean_functor = div
    ta_1.read_values()
    ta_2.mean_functor = div
    ta_2.read_values()
    ta_3.mean_functor = div
    ta_3.read_values()
    
    ax = pylab.gca()
    ta_1.plot_error_bar(ax, marker='o')
    ta_2.plot_error_bar(ax, marker='d')
    ta_3.plot_error_bar(ax, marker='s')
    ax.set_ylim(0,100)
    ax.set_ylabel("Cells dying in interphase (%)")
    pylab.savefig('dying_interphase_Taxol.pdf')
    
    # Nocodazole dying
    pylab.figure()
    mt_fle = "__mito_timing.txt"
    
    NO_no  =  ["D11", "C02", "D02", "E02", "F02", "G02", "B03",]
    NO_mad  = ["D11", "C05", "D05", "E05", "F05", "G05", "B06",]
    NO_plk  = ["D11", "C08", "D08", "E08", "F08", "G08", "B09",]
    
    cons_nm = ["%s" % c for c in ['0',] + map(str, [12,25,50, 100, 200, 400])]
    
    ax = pylab.gca()
    no_1 = ConcentrationLine("Nocodazole no siRNA", NO_no , cons_nm, '__dyinging_in_mito_or_apo.txt', "g", has_yerr=False)
    no_2 = ConcentrationLine("Nocodazole siMad2", NO_mad, cons_nm, '__dyinging_in_mito_or_apo.txt', "b", has_yerr=False)
    no_3 = ConcentrationLine("Nocodazole siPlk1", NO_plk, cons_nm, '__dyinging_in_mito_or_apo.txt', "r", has_yerr=False)
    div = lambda x: ((x[0] / float(x[2])) * 100.0)
    no_1.mean_functor = div
    no_1.read_values()
    no_2.mean_functor = div
    no_2.read_values()
    no_3.mean_functor = div
    no_3.read_values()
    
    no_1.set_concentration_unit('ng/ml')
    no_2.set_concentration_unit('ng/ml')
    no_3.set_concentration_unit('ng/ml')
    
    no_1.plot_error_bar(ax, marker='o')
    no_2.plot_error_bar(ax, marker='d')
    no_3.plot_error_bar(ax, marker='s')
    ax.set_ylim(0,100)
    ax.set_ylabel("Cells dying in mitosis (%)")
    pylab.savefig('dying_mitosis_Nocodazole.pdf')
    
    pylab.figure()
    ax = pylab.gca()
    div = lambda x: ((x[1] / float(x[2])) * 100.0)
    no_1.mean_functor = div
    no_1.read_values()
    no_2.mean_functor = div
    no_2.read_values()
    no_3.mean_functor = div
    no_3.read_values()
    
    no_1.set_concentration_unit('ng/ml')
    no_2.set_concentration_unit('ng/ml')
    no_3.set_concentration_unit('ng/ml')
    
    no_1.plot_error_bar(ax, marker='o')
    no_2.plot_error_bar(ax, marker='d')
    no_3.plot_error_bar(ax, marker='s')
    ax.set_ylim(0,100)
    ax.set_ylabel("Cells dying in interphase (%)")
    pylab.savefig('dying_interphase_Nocodazole.pdf')
    pylab.show()


def boehringer_mitotic_timing():
    """ plots concentration versus mitotic timing for Taxol, Noco and BI
        color code == fate of beeing either Gascoigne class 1,3,5 (only first mitosis)
    """
    mt_fle = "__mito_timing.txt"
    mc_fle = "__mito_classes.txt"
     
    def plot_spreads(title, wells, conc, conc_label):
        fig = pylab.figure(figsize=(9,6))
        ax = pylab.gca()
        bi_1 = ColoredConcentrationTimingSpread("BI2536 no siRNA", wells , conc, '__mito_timing.txt', "g")
        bi_1.set_concentration_unit(conc_label)
        bi_1.read_fate_class("__mito_classes.txt")
        
        bi_1.plot_spread(ax, marker='o')
        ax.set_ylim(0,2000)
        ax.set_title(title)
        pylab.tight_layout()
        lgd = ax.get_legend()
        pylab.savefig('%s.pdf' % title, bbox_extra_artists=(lgd,), bbox_inches='tight')
        pylab.savefig('%s.png' % title, bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    def plot_bars(title, wells, conc, conc_label):
        fig = pylab.figure(figsize=(9,6))
        ax = pylab.gca()
        bi_1 = ColoredConcentrationTimingBar("BI2536 no siRNA", wells , conc, '__mito_timing.txt', "g")
        bi_1.set_concentration_unit(conc_label)
        bi_1.read_fate_class("__mito_classes.txt")
        
        bi_1.plot_bar(ax, marker='o')
        ax.set_ylim(0,2000)
        ax.set_title(title)
        pylab.tight_layout()
        lgd = ax.get_legend()
        pylab.savefig('%s.pdf' % title, bbox_extra_artists=(lgd,), bbox_inches='tight')
        pylab.savefig('%s.png' % title, bbox_extra_artists=(lgd,), bbox_inches='tight')
        
    def plot_stacked_bars(title, wells, conc, conc_label):
        fig = pylab.figure(figsize=(9,6))
        ax = pylab.gca()
        bi_1 = ConcentrationStackedBar("BI2536 no siRNA", wells , conc, '__mito_timing.txt', "g")
        bi_1.set_concentration_unit(conc_label)
        bi_1.read_fate_class("__mito_classes.txt")
        
        bi_1.plot_stacked_fate_bar(ax, marker='o')
        ax.set_ylim(0,1.1)
        ax.set_title(title)
        pylab.tight_layout()
        lgd = ax.get_legend()
        pylab.savefig('%s.pdf' % title, bbox_extra_artists=(lgd,), bbox_inches='tight')
        pylab.savefig('%s.png' % title, bbox_extra_artists=(lgd,), bbox_inches='tight')
        
    plot_ = plot_spreads
    plot_ = plot_bars
    plot_ = plot_stacked_bars
        
        
    BI_no  =  ["D11", "B04", "C04", "D04", "E04",]
    BI_mad  = ["D11", "B07", "C07", "D07", "E07",]
    BI_plk  = ["D11", "B10", "C10", "D10", "E10",]
    
    cons_nm = ["%s" % c for c in ['0',] + map(str, [1,5,25,125])]
        
    plot_('no siRNA - BI2536', BI_no, cons_nm, 'nM')
    plot_('siMad2 - BI2536 ', BI_mad, cons_nm, 'nM')
    plot_('siPlk1 - BI2536', BI_plk, cons_nm, 'nM')
    
    NO_no  =  ["D11", "C02", "D02", "E02", "F02", "G02", "B03",]
    NO_mad  = ["D11", "C05", "D05", "E05", "F05", "G05", "B06",]
    NO_plk  = ["D11", "C08", "D08", "E08", "F08", "G08", "B09",]
    
    cons_nm = ["%s" % c for c in ['0',] + map(str, [12,25,50, 100, 200, 400])]
    plot_('no siRNA - Nocodazole', NO_no, cons_nm, 'ng/ml')
    plot_('siMad2 - Nocodazole', NO_mad, cons_nm, 'ng/ml')
    plot_('siPlk1 - Nocodazole', NO_plk, cons_nm, 'ng/ml')
    
    TA_no  =  ["D11", "C03", "D03", "E03", "F03",]
    TA_mad  = ["D11", "C06", "D06", "E06", "F06",]
    TA_plk  = ["D11", "C09", "D09", "E09", "F09",]
    
    cons_nm = ["%s" % c for c in ['0',] + map(str, [10,100,1000, 10000])]
    
    plot_('no siRNA - Taxol', TA_no, cons_nm, 'nM')
    plot_('siMad2 - Taxol ', TA_mad, cons_nm, 'nM')
    plot_('siPlk1 - Taxol', TA_plk, cons_nm, 'nM')
    
    pylab.show()

if __name__ == "__main__":
    boehringer_mitotic_timing()