import numpy as np
import os, sys
from collections import OrderedDict
import string
import matplotlib.ticker as ticker
from plot import *
    
def calc_rank_histogram(truth, x_sorted, rank_hist):
    x_sorted.sort(axis=0)
    nem, ndim = x_sorted.shape
    
    for i, t in enumerate(truth):
        if t < x_sorted[0,i]:
            rank_hist[0,i] += 1
        elif (t >= x_sorted[nem-2,i]) and (t <= x_sorted[nem-1,i]):
            rank_hist[nem-1,i] += 1
        elif t > x_sorted[nem-1,i]:
            rank_hist[nem,i] += 1
        else:
            for j in range(1,nem-1):
                if (t >= x_sorted[j-1,i]) and (t <= x_sorted[j,i]):
                    rank_hist[j,i] += 1

                    
def calc_rank_histogram_rate(rank_hist):
    rank_hist_rate = rank_hist.astype(float)/rank_hist.sum(axis=0)
    rank_hist_all_state = rank_hist.sum(axis=1)
    rank_hist_all_state_rate = rank_hist_all_state.astype(float)/rank_hist_all_state.sum()

    return rank_hist_rate, rank_hist_all_state, rank_hist_all_state_rate 


def plot_rank_histogram_each_state(iEnKF, enkf_method, h_type, nem, nobs, rank_hist, plot_show, plot_outdir, plot_save, plot_data_outdir, plot_data_save):
    rank_hist_rate, rank_hist_all_state, rank_hist_all_state_rate = calc_rank_histogram_rate(rank_hist)

    import matplotlib
    if os.name != 'nt':
        matplotlib.use('Agg')                                                           # Generating matplotlib graphs without a running X server
    import matplotlib.pyplot as plt
    
    majors = [0, nem//2, nem]
    set_plot_font(plt,0)
    fig_nrows = 8
    fig_ncols = 5
    fig, axs = plt.subplots(fig_nrows, fig_ncols, figsize=(20,15))

    x = np.arange(start=0,stop=nem+1,step=1,dtype=int)
    title_fontsize = 14
    tick_fontsize = 12

    label_fontsize = 15

    text_pos = (1.0, 0.83)
    text_list = list(string.ascii_lowercase)
    text_fontsize = 14

    for irow in range(fig_nrows):
        for icol in range(fig_ncols):
            i = irow * fig_ncols + icol
            axs[irow, icol].set_title('$\mathrm{X}_\mathrm{%d}$'%(i+1), fontsize=title_fontsize)
            if float(matplotlib.__version__[:3]) >= 2.0:
                axs[irow, icol].bar(x=x, height=rank_hist_rate[:,i], width=1.0, color='k')
            else:
                axs[irow, icol].bar(x=x-0.5, height=rank_hist_rate[:,i], width=1.0, color='k')
            #axs[irow, icol].text(text_pos[0], text_pos[1], text_list[i], fontsize=text_fontsize)
            axs[irow, icol].xaxis.set_major_locator(ticker.FixedLocator(majors))
            axs[irow, icol].tick_params(direction='in', bottom=True, top=True, left=True, right=True)
            axs[irow, icol].tick_params(axis='x', labelsize=tick_fontsize)
            axs[irow, icol].tick_params(axis='y', labelsize=tick_fontsize)
            axs[irow, icol].set_xlim(-0.5, nem + 0.5)
            
            if icol == 0:
                axs[irow, icol].set_ylabel('{}\nFrequency'.format(h_mathexp_str(h_type)), fontsize=label_fontsize)
                
            if irow == fig_nrows - 1:
                axs[irow, icol].set_xlabel('Rank', fontsize=label_fontsize)
              
    fig.tight_layout()

    fn = '8_1_rank_histogram_each_state_iEnKF_{}_enkf_method_{}_h_type_{}_nem_{}_nobs_{}'.format(iEnKF,enkf_method,h_type,nem,nobs)
    if plot_save:
        save_plot(plt, os.path.join(plot_outdir,fn), ['png','svg','eps'])
    if plot_show: 
        plt.show()
    if plot_data_save:
        np.savetxt(os.path.join(plot_data_outdir,'{}_x.txt'.format(fn)), x)
        np.savetxt(os.path.join(plot_data_outdir,'{}_y.txt'.format(fn)), rank_hist[:,:])
        np.savetxt(os.path.join(plot_data_outdir,'{}_y_rate.txt'.format(fn)), rank_hist_rate[:,:])
        

def plot_rank_histogram_all_state(iEnKF, enkf_method, h_type, nem, nobs, rank_hist, plot_show, plot_outdir, plot_save, plot_data_outdir, plot_data_save):
    rank_hist_rate, rank_hist_all_state, rank_hist_all_state_rate = calc_rank_histogram_rate(rank_hist)

    import matplotlib
    if os.name != 'nt':
        matplotlib.use('Agg')                                                           # Generating matplotlib graphs without a running X server
    import matplotlib.pyplot as plt
    
    set_plot_font(plt,0)
    plt.figure()
    x = np.arange(start=0,stop=nem+1,step=1,dtype=int)
    title_fontsize = 14
    tick_fontsize = 12
    
    if float(matplotlib.__version__[:3]) >= 2.0:
        plt.bar(x=x, height=rank_hist_all_state_rate[:], width=1.0, color='k')  
    else:
        plt.bar(left=x-0.5, height=rank_hist_all_state_rate[:], width=1.0, color='k')
        
    plt.title('$\mathrm{X}_\mathrm{all}$', fontsize=title_fontsize)    
    plt.ylabel('{}\nFrequency'.format(h_mathexp_str(h_type)), fontsize=13)
    plt.xlabel('Rank', fontsize=13)
    plt.xticks(x)
    plt.tick_params(direction='in', bottom=True, top=True, left=True, right=True)
    plt.tick_params(axis='x', labelsize=tick_fontsize)
    plt.tick_params(axis='y', labelsize=tick_fontsize)
    plt.xlim(-0.5, nem + 0.5)
    plt.tight_layout()

    fn = '8_2_rank_histogram_all_state_iEnKF_{}_enkf_method_{}_h_type_{}_nem_{}_nobs_{}'.format(iEnKF,enkf_method,h_type,nem,nobs)
    if plot_save:
        save_plot(plt, os.path.join(plot_outdir,fn), ['png','svg','eps'])
    if plot_show: 
        plt.show()
    if plot_data_save:
        np.savetxt(os.path.join(plot_data_outdir,'{}_x.txt'.format(fn)), x)
        np.savetxt(os.path.join(plot_data_outdir,'{}_y.txt'.format(fn)), rank_hist_all_state[:])  
        np.savetxt(os.path.join(plot_data_outdir,'{}_y_rate.txt'.format(fn)), rank_hist_all_state_rate[:])  
                            
                            
def plot_rank_histogram_specific_state(iEnKF, enkf_method, nem, nobs, rank_hist_rate, rank_hist_all_state_rate, plot_show, plot_outdir, plot_save): 
    """
    figure for paper
    """
    import matplotlib
    if os.name != 'nt':
        matplotlib.use('Agg')                                                       # Generating matplotlib graphs without a running X server
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    
    set_plot_font(plt,0)
    fig_nrows = 3
    fig_ncols = 6
    fig, axs = plt.subplots(fig_nrows, fig_ncols, figsize=(15,6))
    
    x = np.arange(start=0,stop=nem+1,step=1,dtype=int)                           # x axis values
    if float(matplotlib.__version__[:3]) < 2.0:
        x -= -0.5    
    xi = range(1,40,8)                                                              # specific state where rank histogram is illustrated
    ylim_range = {3:{12:(0.0, 0.21),50:(0.0, 0.06)},5:{4:(0.0, 0.75)},18:{50:(0.0, 0.05)},24:{12:(0.0, 0.2)}}                # y axis range
    enkf_method_label_list = {3:'LETKF', 5:'LUTKF', 18:'GETKF', 24:'MLUTKF'}

    majors = [0, nem//2, nem]
    title_fontsize = 14
    tick_fontsize = 12
    label_fontsize = 15
    text_fontsize = 13
    
    text_pos = (-0.3, 0.9)
    text_list = list(string.ascii_lowercase)
    
    for irow in range(fig_nrows):
        for icol in range(fig_ncols):
            if icol == (fig_ncols - 1):
                axs[irow, icol].set_title('$\mathrm{X}_\mathrm{all}$', fontsize=title_fontsize)
                axs[irow, icol].bar(x=x, height=rank_hist_all_state_rate[enkf_method][nem][irow][nobs][:], width=1.0, color='k')
            else:
                axs[irow, icol].set_title('$\mathrm{X}_\mathrm{%d}$'%(xi[icol]+1), fontsize=title_fontsize)
                axs[irow, icol].bar(x=x, height=rank_hist_rate[enkf_method][nem][irow][nobs][:,xi[icol]], width=1.0, color='k')
                    
            if icol == 0:
                axs[irow, icol].set_ylabel('{}\nFrequency'.format(h_mathexp_str(irow)), fontsize=label_fontsize)      
                
            if irow == (fig_nrows - 1):
                axs[irow, icol].set_xlabel('Rank', fontsize=label_fontsize)

            axs[irow, icol].set_xlim(-0.5, nem + 0.5)
            axs[irow, icol].set_ylim(ylim_range[enkf_method][nem])
            if enkf_method == 9:
                axs[irow, icol].xaxis.set_major_locator(ticker.FixedLocator(majors))
                #axs[irow, icol].xaxis.set_major_locator(ticker.MaxNLocator(2))
            else:
                #axs[irow, icol].set_xticks(x)
                axs[irow, icol].xaxis.set_major_locator(ticker.FixedLocator(majors))
            axs[irow, icol].tick_params(direction='in', bottom=True, top=True, left=True, right=True)
            axs[irow, icol].tick_params(axis='x', labelsize=tick_fontsize)
            axs[irow, icol].tick_params(axis='y', labelsize=tick_fontsize)
            #axs[irow, icol].text(text_pos[enkf_method][nem][0], text_pos[enkf_method][nem][1], text_list[irow*fig_ncols+icol]+')', fontsize=text_fontsize)
            #axs[irow, icol].text(text_pos[enkf_method][nem][0], text_pos[enkf_method][nem][1], text_list[irow*fig_ncols+icol]+')', fontsize=text_fontsize)
            #axs[irow, icol].text(text_pos[0], text_pos[1],text_list[irow*fig_ncols+icol]+')',fontsize=text_fontsize
            axs[irow, icol].text(0.02, 0.95,text_list[irow*fig_ncols+icol]+')',transform=axs[irow, icol].transAxes,fontsize=text_fontsize,va='top', ha='left')

    fig.tight_layout()

    fn = '8_3_rank_histogram_specific_state_iEnKF_{}_enkf_method_{}_h_type_all_nem_{}_nobs_{}'.format(iEnKF,enkf_method,nem,nobs)
    if plot_save:
        save_plot(plt, os.path.join(plot_outdir,fn), ['pdf','svg','eps'])
    if plot_show:
        plt.show()

if __name__ == "__main__":
    enkf_method_list = [3, 5, 18, 24]                                            # letkf (3), lutkf (5), getkf (18), mlutkf (24)
    h_type_list = [0,1,2]
    nem_list = {3:[12,50],5:[4],18:[50],24:[12]}                      # letkf: 12, 50 ensembles, lutkf: 4 ensembles, getkf: 50 ensembles, mlutkf: 12 ensembles
    #nobs_list = [100,200]
    nobs_list = [100]

    #---------------------------------------------------------------------------------------------------
    # 1. plot rank histogram for each state and all states
    #---------------------------------------------------------------------------------------------------
    for iEnKF in [0]:   
        for enkf_method in enkf_method_list:
            for nem in nem_list[enkf_method]:
                for h_type in h_type_list:
                    for nobs in nobs_list:
                        rank_hist=np.loadtxt(os.path.join("data","8_1_rank_histogram_each_state_iEnKF_{}_enkf_method_{}_h_type_{}_nem_{}_nobs_{}_y.txt".format(iEnKF,enkf_method,h_type,nem,nobs)))
                        plot_rank_histogram_each_state(iEnKF, enkf_method, h_type, nem, nobs, rank_hist, False, "fig", True, "data", False)
                        plot_rank_histogram_all_state(iEnKF, enkf_method, h_type, nem, nobs, rank_hist, False, "fig", True, "data", False)
                   
    #---------------------------------------------------------------------------------------------------
    # 2. plot rank_histogram for specific_state (for paper)
    #---------------------------------------------------------------------------------------------------
    rank_hist_rate = {}
    rank_hist_all_state_rate = {}

    for iEnKF in [0]:   
        for enkf_method in enkf_method_list:
            rank_hist_rate[enkf_method]={}
            rank_hist_all_state_rate[enkf_method]={}
            for nem in nem_list[enkf_method]:
                rank_hist_rate[enkf_method][nem]={}
                rank_hist_all_state_rate[enkf_method][nem]={}            
                for h_type in h_type_list:
                    rank_hist_rate[enkf_method][nem][h_type]={}
                    rank_hist_all_state_rate[enkf_method][nem][h_type]={}
                    for nobs in nobs_list:
                        rank_hist_rate[enkf_method][nem][h_type][nobs]=np.loadtxt(os.path.join("data","8_1_rank_histogram_each_state_iEnKF_{}_enkf_method_{}_h_type_{}_nem_{}_nobs_{}_y_rate.txt".format(iEnKF,enkf_method,h_type,nem,nobs)))
                        rank_hist_all_state_rate[enkf_method][nem][h_type][nobs]=np.loadtxt(os.path.join("data","8_2_rank_histogram_all_state_iEnKF_{}_enkf_method_{}_h_type_{}_nem_{}_nobs_{}_y_rate.txt".format(iEnKF,enkf_method,h_type,nem,nobs)))

    for iEnKF in [0]:
        for enkf_method in enkf_method_list:
            for nem in nem_list[enkf_method]:
                for nobs in nobs_list:
                    plot_rank_histogram_specific_state(iEnKF, enkf_method, nem, nobs, rank_hist_rate, rank_hist_all_state_rate, False, "fig", True)           
