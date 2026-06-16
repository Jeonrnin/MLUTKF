import numpy as np
import os, sys
from collections import OrderedDict
import string
from glob import glob
from plot import *

def plot_hovmuller_diagram(x, y, val, colorbar_range, colorbar_extend, plt, title, fn, plot_outdir, plot_save, plot_show, plot_data_outdir, plot_data_save):
    plt.figure()
    plt.contourf(x,y,val,colorbar_range,cmap=plt.cm.bwr,extend=colorbar_extend)
    plt.xlabel('$\mathrm{x}$')
    plt.ylabel('Time steps')
    plt.colorbar()
    plt.title(title)
    if plot_save:
        save_plot(plt,os.path.join(plot_outdir,fn),['png','svg','eps'])
    if plot_show:     
        plt.show()
    if plot_data_save:
        np.savetxt(os.path.join(plot_data_outdir,'{}_x_axis.txt'.format(fn)), x)
        np.savetxt(os.path.join(plot_data_outdir,'{}_y_axis.txt'.format(fn)), y)
        np.savetxt(os.path.join(plot_data_outdir,'{}_value.txt'.format(fn)), val)

def plot_hovmuller_diagram_for_paper(nobs, x_axis, y_axis, err_value, colorbar_range, fig_prefix, plot_show, plot_outdir, plot_save): 
    """
    figure for paper
    """
    import matplotlib
    if os.name != 'nt':                                                                 # in case of linux system    
        matplotlib.use('Agg')                                                           # Generating matplotlib graphs without a running X server
    import matplotlib.pyplot as plt
    
    set_plot_font(plt,0)
    fig_nrows = 1
    fig_ncols = 4
    fig, axs = plt.subplots(fig_nrows, fig_ncols, figsize=(15,3.5))
    
    title_fontsize = 14
    tick_fontsize = 11
    label_fontsize = 15
    text_fontsize = 14
    
    text_pos = (2, 3920) 
    text_list = list(string.ascii_lowercase)
                    
    for icol in range(fig_ncols):
        if icol == 0:
            im = axs[icol].contourf(x_axis[3][3][2][nobs],y_axis[3][3][2][nobs],err_value[3][3][2][nobs],colorbar_range,cmap=plt.cm.bwr,extend='both')
            axs[icol].set_ylabel('Time steps', fontsize=label_fontsize)  
        elif icol == 1:
            im = axs[icol].contourf(x_axis[3][10][2][nobs],y_axis[3][10][2][nobs],err_value[3][10][2][nobs],colorbar_range,cmap=plt.cm.bwr,extend='both')                    
        elif icol == 2:
            im = axs[icol].contourf(x_axis[7][3][2][nobs],y_axis[7][3][2][nobs],err_value[7][3][2][nobs],colorbar_range,cmap=plt.cm.bwr,extend='both')                    
        elif icol == 3:
            im = axs[icol].contourf(x_axis[6][81][2][nobs],y_axis[6][81][2][nobs],err_value[6][81][2][nobs],colorbar_range,cmap=plt.cm.bwr,extend='both')                    
                       
        #axs[icol].set_xlabel('$\mathrm{x}$', fontsize=label_fontsize)
        axs[icol].set_xlabel('Variables', fontsize=label_fontsize)
        fig.colorbar(im, ax=axs[icol])    
        #axs[icol].tick_params(direction='in', bottom=True, top=True, left=True, right=True)
        axs[icol].tick_params(axis='x', labelsize=tick_fontsize)
        axs[icol].tick_params(axis='y', labelsize=tick_fontsize)
        axs[icol].text(text_pos[0], text_pos[1], text_list[icol]+')', fontsize=text_fontsize)

    fig.tight_layout()

    fn = '{}_nobs{}_for_paper'.format(fig_prefix,nobs)
    if plot_save:
        save_plot(plt, os.path.join(plot_outdir,fn), ['png','svg','eps'])
    if plot_show:
        plt.show()

def plot_all_hovmuller_diagram_for_paper(nobs, x_axis_berr, y_axis_berr, err_value_berr, colorbar_range_berr, x_axis_ainc, y_axis_ainc, err_value_ainc, colorbar_range_ainc, \
                                              fig_prefix, plot_show, plot_outdir, plot_save): 
    """
    figure for paper
    """
    import matplotlib
    if os.name != 'nt':                                                                 # in case of linux system    
        matplotlib.use('Agg')                                                           # Generating matplotlib graphs without a running X server
    import matplotlib.pyplot as plt
    
    set_plot_font(plt,0)
    fig_nrows = 2
    fig_ncols = 4
    fig, axs = plt.subplots(fig_nrows, fig_ncols, figsize=(15,6))
    
    title_fontsize = 14
    tick_fontsize = 11
    label_fontsize = 15
    text_fontsize = 14
    
    text_pos = (2.2, 3905) 
    text_list = list(string.ascii_lowercase)

    for irow in range(fig_nrows):                    
        for icol in range(fig_ncols):
            if irow == 0:
                if icol == 0:
                    im = axs[irow,icol].contourf(x_axis_berr[3][3][2][nobs],y_axis_berr[3][3][2][nobs],err_value_berr[3][3][2][nobs],colorbar_range_berr,cmap=plt.cm.bwr,extend='both')
                    axs[irow,icol].set_ylabel('Time steps', fontsize=label_fontsize)  
                elif icol == 1:
                    im = axs[irow,icol].contourf(x_axis_berr[3][10][2][nobs],y_axis_berr[3][10][2][nobs],err_value_berr[3][10][2][nobs],colorbar_range_berr,cmap=plt.cm.bwr,extend='both')                    
                elif icol == 2:
                    im = axs[irow,icol].contourf(x_axis_berr[7][3][2][nobs],y_axis_berr[7][3][2][nobs],err_value_berr[7][3][2][nobs],colorbar_range_berr,cmap=plt.cm.bwr,extend='both')                    
                elif icol == 3:
                    im = axs[irow,icol].contourf(x_axis_berr[6][81][2][nobs],y_axis_berr[6][81][2][nobs],err_value_berr[6][81][2][nobs],colorbar_range_berr,cmap=plt.cm.bwr,extend='both')                    
            elif irow == 1:
                if icol == 0:
                    im = axs[irow,icol].contourf(x_axis_ainc[3][3][2][nobs],y_axis_ainc[3][3][2][nobs],err_value_ainc[3][3][2][nobs],colorbar_range_ainc,cmap=plt.cm.bwr,extend='both')
                    axs[irow,icol].set_ylabel('Time steps', fontsize=label_fontsize)  
                elif icol == 1:
                    im = axs[irow,icol].contourf(x_axis_ainc[3][10][2][nobs],y_axis_ainc[3][10][2][nobs],err_value_ainc[3][10][2][nobs],colorbar_range_ainc,cmap=plt.cm.bwr,extend='both')                    
                elif icol == 2:
                    im = axs[irow,icol].contourf(x_axis_ainc[7][3][2][nobs],y_axis_ainc[7][3][2][nobs],err_value_ainc[7][3][2][nobs],colorbar_range_ainc,cmap=plt.cm.bwr,extend='both')                    
                elif icol == 3:
                    im = axs[irow,icol].contourf(x_axis_ainc[6][81][2][nobs],y_axis_ainc[6][81][2][nobs],err_value_ainc[6][81][2][nobs],colorbar_range_ainc,cmap=plt.cm.bwr,extend='both')                    
       
                #axs[irow,icol].set_xlabel('$\mathrm{x}$', fontsize=label_fontsize)
                axs[irow,icol].set_xlabel('Variables', fontsize=label_fontsize)

            fig.colorbar(im, ax=axs[irow,icol])    
            #axs[icol].tick_params(direction='in', bottom=True, top=True, left=True, right=True)
            axs[irow,icol].tick_params(axis='x', labelsize=tick_fontsize)
            axs[irow,icol].tick_params(axis='y', labelsize=tick_fontsize)
            axs[irow,icol].text(text_pos[0], text_pos[1], text_list[irow*fig_ncols+icol]+')', fontsize=text_fontsize)

    fig.tight_layout()

    fn = '{}_nobs{}_for_paper'.format(fig_prefix,nobs)
    if plot_save:
        save_plot(plt, os.path.join(plot_outdir,fn), ['png','svg','eps'])
    if plot_show:
        plt.show()
        
if __name__ == "__main__":
    enkf_method_list = [3,7,6]                                       # letkf, slutkf, sutkf
    nem_list = {3:[3,10],7:[3],6:[81]}                               # letkf: 3, 10 ensembles, slutkf: 3 ensembles, sutkf: 81 ensembles
    h_type_list = [2]
    nobs_list = [40]                                                 # case B: nobs 200, case C: nobs 40

    #---------------------------------------------------------------------------------------------------
    # plot background error (for paper)
    #---------------------------------------------------------------------------------------------------
    err_value_berr = {}
    x_axis_berr = {}
    y_axis_berr = {}

    for enkf_method in enkf_method_list:
        err_value_berr[enkf_method]={}
        x_axis_berr[enkf_method]={}
        y_axis_berr[enkf_method]={}
        for nem in nem_list[enkf_method]:
            err_value_berr[enkf_method][nem]={}
            x_axis_berr[enkf_method][nem]={}
            y_axis_berr[enkf_method][nem]={}
            for h_type in h_type_list:
                err_value_berr[enkf_method][nem][h_type]={}
                x_axis_berr[enkf_method][nem][h_type]={}
                y_axis_berr[enkf_method][nem][h_type]={}
                for nobs in nobs_list:
                    x_axis_berr[enkf_method][nem][h_type][nobs]=np.loadtxt(glob(os.path.join("data_background_error_paper","1_3_background_error_enkf_method_{}_h_type_{}_nem_{}_nobs_{}_obs_loc_dist_*_x_axis.txt".format(enkf_method,h_type,nem,nobs)))[0]).astype(int)
                    y_axis_berr[enkf_method][nem][h_type][nobs]=np.loadtxt(glob(os.path.join("data_background_error_paper","1_3_background_error_enkf_method_{}_h_type_{}_nem_{}_nobs_{}_obs_loc_dist_*_y_axis.txt".format(enkf_method,h_type,nem,nobs)))[0]).astype(int)
                    err_value_berr[enkf_method][nem][h_type][nobs]=np.loadtxt(glob(os.path.join("data_background_error_paper","1_3_background_error_enkf_method_{}_h_type_{}_nem_{}_nobs_{}_obs_loc_dist_*_value.txt".format(enkf_method,h_type,nem,nobs)))[0])

    for nobs in nobs_list:
        plot_hovmuller_diagram_for_paper(nobs, x_axis_berr, y_axis_berr, err_value_berr, np.linspace(-18,18,41), '1_4_background_error', True, "fig", True)

    #---------------------------------------------------------------------------------------------------
    # plot analysis increment (for paper)
    #---------------------------------------------------------------------------------------------------
    err_value_ainc = {}
    x_axis_ainc = {}
    y_axis_ainc = {}

    for enkf_method in enkf_method_list:
        err_value_ainc[enkf_method]={}
        x_axis_ainc[enkf_method]={}
        y_axis_ainc[enkf_method]={}
        for nem in nem_list[enkf_method]:
            err_value_ainc[enkf_method][nem]={}
            x_axis_ainc[enkf_method][nem]={}
            y_axis_ainc[enkf_method][nem]={}
            for h_type in h_type_list:
                err_value_ainc[enkf_method][nem][h_type]={}
                x_axis_ainc[enkf_method][nem][h_type]={}
                y_axis_ainc[enkf_method][nem][h_type]={}
                for nobs in nobs_list:
                    x_axis_ainc[enkf_method][nem][h_type][nobs]=np.loadtxt(glob(os.path.join("data_analysis_increment_paper","1_5_analysis_increment_enkf_method_{}_h_type_{}_nem_{}_nobs_{}_obs_loc_dist_*_x_axis.txt".format(enkf_method,h_type,nem,nobs)))[0]).astype(int)
                    y_axis_ainc[enkf_method][nem][h_type][nobs]=np.loadtxt(glob(os.path.join("data_analysis_increment_paper","1_5_analysis_increment_enkf_method_{}_h_type_{}_nem_{}_nobs_{}_obs_loc_dist_*_y_axis.txt".format(enkf_method,h_type,nem,nobs)))[0]).astype(int)
                    err_value_ainc[enkf_method][nem][h_type][nobs]=np.loadtxt(glob(os.path.join("data_analysis_increment_paper","1_5_analysis_increment_enkf_method_{}_h_type_{}_nem_{}_nobs_{}_obs_loc_dist_*_value.txt".format(enkf_method,h_type,nem,nobs)))[0])

    for nobs in nobs_list:
        plot_hovmuller_diagram_for_paper(nobs, x_axis_ainc, y_axis_ainc, err_value_ainc, np.linspace(-4.0,4.0,41), '1_5_analysis_increment', True, "fig", True)


    #---------------------------------------------------------------------------------------------------
    # plot both background error and analysis increment in one figure (for paper)
    #---------------------------------------------------------------------------------------------------
    for nobs in nobs_list:
        plot_all_hovmuller_diagram_for_paper(nobs, x_axis_berr, y_axis_berr, err_value_berr, np.linspace(-18,18,41), x_axis_ainc, y_axis_ainc, err_value_ainc, np.linspace(-4.0,4.0,41), \
                                             '1_6_background_error_analysis_increment', True, "fig", True)          
