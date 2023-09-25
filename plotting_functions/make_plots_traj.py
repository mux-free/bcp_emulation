import pandas as pd
import numpy as np
import xarray as xr
import seaborn as sns
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
# from matplotlib.gridspec import GridSpec
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors


from sklearn.metrics import r2_score, mean_squared_error

sys.path.append('/home/freimax/msc_thesis/scripts/plotting_functions/')
import plot_helpers





def get_quantiles(df, bc_processes=['Atsubsi', 'Atmeltsi']):
    bcp_mean   = df.groupby(['time'])[bc_processes].mean().reset_index()
    bcp_median = df.groupby(['time'])[bc_processes].median().reset_index()
    bcp_q25    = df.groupby(['time'])[bc_processes].quantile(0.25).reset_index()
    bcp_q75    = df.groupby(['time'])[bc_processes].quantile(0.75).reset_index()
    bcp_q10    = df.groupby(['time'])[bc_processes].quantile(0.10).reset_index()
    bcp_q90    = df.groupby(['time'])[bc_processes].quantile(0.90).reset_index()
    bcp_q1     = df.groupby(['time'])[bc_processes].quantile(0.01).reset_index()
    bcp_q99    = df.groupby(['time'])[bc_processes].quantile(0.99).reset_index()
    # print("Return order: bcp_mean, bcp_median, bcp_q25, bcp_q75, bcp_q10, bcp_q90, bcp_q1, bcp_q99")
    return bcp_mean, bcp_median, bcp_q25, bcp_q75, bcp_q10, bcp_q90, bcp_q1, bcp_q99

def calculate_rmse(y, y_pred):
    return np.sqrt(np.mean((y-y_pred)**2))

def subplot_traj_distri(ax, x_axis, metric_list, bc_process, ylabel=True, xlabel=True, legend=False, residual=False, title='', set_ylim=None):
    ## get_qunatiles -> metirc_list       (function output-orer: bcp_mean, bcp_median, bcp_q25, bcp_q75, bcp_q10, bcp_q90, bcp_q1, bcp_q99)
    ax.plot( x_axis , metric_list[0][bc_process] , c='orange' , label='mean'    ,   linewidth=3   ,  alpha=0.95, zorder=2)
    ax.plot( x_axis , metric_list[1][bc_process] , c='k'      , label='median'  ,   linewidth=3   ,  alpha=0.9 , zorder=1)
    ax.plot( x_axis , metric_list[2][bc_process] , c='k'      , label='Q25, Q75',   linewidth=1.5 ,  alpha=0.95)
    ax.plot( x_axis , metric_list[3][bc_process] , c='k'      ,                     linewidth=1.5 ,  alpha=0.95)
    ax.plot( x_axis , metric_list[4][bc_process] , c='k'      , label='Q10, Q90',   linewidth=0.75,  alpha=0.95)
    ax.plot( x_axis , metric_list[5][bc_process] , c='k'      ,                     linewidth=0.75,  alpha=0.95)
    ax.plot( x_axis , metric_list[6][bc_process] , c='k'      , label='Q1 , Q99',   linewidth=1, linestyle='--', dashes=(5,5))
    ax.plot( x_axis , metric_list[7][bc_process] , c='k'      ,                     linewidth=1, linestyle='--', dashes=(5,5))
    ax.fill_between( x_axis , metric_list[4][bc_process], metric_list[5][bc_process]   ,   color='blue', alpha=0.25)
    ax.fill_between( x_axis , metric_list[2][bc_process], metric_list[3][bc_process]   ,   color='blue', alpha=0.25)
    
    if  set_ylim:
        if set_ylim <= -5:
            y_lower_bound = 0.05
        else:
            y_lower_bound = 0.25
        ax.set_ylim(y_lower_bound,set_ylim)
    else:
        ax.set_ylim(0.25,-10.25)

    ax.set_title(title)  
    if ylabel:
        ax.set_ylabel('Accumulated Cooling [K/h]')
    if xlabel:
        ax.set_xlabel('Time [h]')
    if legend:
        ax.legend(loc='upper left', bbox_to_anchor=(0, 0.95), shadow=True, edgecolor='gray', )
    if residual:            
        ## Handle axes-extend in resid_plots
        resid_limit = []
        resid_limit.append(np.floor(np.min( metric_list[6][bc_process] ) *2) /2)
        resid_limit.append(np.ceil(np.max( metric_list[7][bc_process] ) *2) /2)
    
        ax_limit = resid_limit[np.argmax(np.abs(resid_limit))]
        if np.abs(ax_limit ) < 1:
            ax_limit = -1    
        # ax.set_ylim(-np.abs(ax_limit), np.abs(ax_limit))
        ax.set_ylim(-1.25, 1.25)




def plot_accu_cooling_over_time(df_tr, df_tr_rf, 
                                date=None, save_path=None, 
                                truth_available=True, set_ylim=None, 
                                text_box=True,
                                set_axes=None):    
    """
    This function creates a figure (and can save it), that shows the time evolution of the Accumulated cooling of all trajectories in df_tr. 
    Additionally it displays the distribution by showing the mean, IQR, 1st and 99th quantile etc.

    df_tr (pd.DataFrame): 
        Containing the cooling value (not necessarily accumulated) and a time column

    df_tr_rf (pd.DataFrame): 
        Same as df_tr, but with the predicted cooling values

    date (str): 
        If present, a suptitle with the date is included (Make sense if we cosnider only a single case)
    
        set_axes (list):    First element   (Axes objects): List of plt.axes element of length 3 (evry row has 3 columns)
                            Second element  (int):          The current row 
                            Third element   (Boolean):      Last row flag (is this row the last one)
                            Fourth element  (str):          Title appendix (can be left empty)
    """
    
    #=======================================================================================================================================================================================================================
    #=======================================================================================================================================================================================================================
    if truth_available:
    ##   FOR IFS DATA (ground truth available:)
    #----------------------------------------------
        fig,ax = plt.subplots(3,3, figsize=(14,14), sharex=True, sharey=False)
        if date:
            plt.suptitle(f'{date}')
        
        # Share y-axis for first two plots in each row
        ax[0,0].get_shared_y_axes().join(ax[0,0], ax[0,1])
        ax[1,0].get_shared_y_axes().join(ax[1,0], ax[1,1])
        ax[1,0].get_shared_y_axes().join(ax[2,0], ax[2,1])

        # time_axis = np.arange(-48,1,1)#bcp_mean['time']
        time_axis = np.unique(df_tr['time'])

          
        ## Calculate quantiles for residuals
        df_tr_rf['Atsubsi_resid']  = df_tr['Atsubsi']  - df_tr_rf['Atsubsi_pred']
        df_tr_rf['Atmeltsi_resid'] = df_tr['Atmeltsi'] - df_tr_rf['Atmeltsi_pred']
        df_tr_rf['Atevr_resid']    = df_tr['Atevr']    - df_tr_rf['Atevr_pred']

        metric_list       = get_quantiles(df_tr, bc_processes=['Atsubsi', 'Atmeltsi', 'Atevr'])                      #(function output-orer: bcp_mean, bcp_median, bcp_q25, bcp_q75, bcp_q10, bcp_q90, bcp_q1, bcp_q99)
        metric_list_pred  = get_quantiles(df_tr_rf, bc_processes=['Atsubsi_pred', 'Atmeltsi_pred', 'Atevr_pred'])
        metric_list_resid = get_quantiles(df_tr_rf, bc_processes=['Atsubsi_resid', 'Atmeltsi_resid', 'Atevr_resid'])
        
        ## Call plotting function
        # For true values (tmeltsi and tsubsi and tevr)
        subplot_traj_distri(ax=ax[0,0], x_axis=time_axis, metric_list=metric_list, bc_process = 'Atsubsi' , ylabel=True, xlabel=False, legend=True  , title=r'AQ$_{subsi}$')
        subplot_traj_distri(ax=ax[1,0], x_axis=time_axis, metric_list=metric_list, bc_process = 'Atmeltsi', ylabel=True, xlabel=False, legend=False , title=r'AQ$_{meltsi}$')
        subplot_traj_distri(ax=ax[2,0], x_axis=time_axis, metric_list=metric_list, bc_process = 'Atevr'   , ylabel=True, xlabel=True , legend=False , title=r'AQ$_{evr}$')
        # For predictions (tmeltsi and tsubsi and tevr)
        subplot_traj_distri(ax=ax[0,1], x_axis=time_axis, metric_list=metric_list_pred, bc_process = 'Atsubsi_pred' , ylabel=False, xlabel=False, legend=False, title=r'Predicted AQ$_{subsi}$')
        subplot_traj_distri(ax=ax[1,1], x_axis=time_axis, metric_list=metric_list_pred, bc_process = 'Atmeltsi_pred', ylabel=False, xlabel=False, legend=False, title=r'Predicted AQ$_{meltsi}$')
        subplot_traj_distri(ax=ax[2,1], x_axis=time_axis, metric_list=metric_list_pred, bc_process = 'Atevr_pred'   , ylabel=False, xlabel=True , legend=False, title=r'Predicted AQ$_{evr}$')
        # For residual (tmeltsi and tsubsi and tevr)
        subplot_traj_distri(ax=ax[0,2], x_axis=time_axis, metric_list=metric_list_resid, bc_process = 'Atsubsi_resid' , ylabel=False, xlabel=False, legend=False, residual=True, title='Residual (true - prediction)')
        subplot_traj_distri(ax=ax[1,2], x_axis=time_axis, metric_list=metric_list_resid, bc_process = 'Atmeltsi_resid', ylabel=False, xlabel=False, legend=False, residual=True, title='Residual (true - prediction)')
        subplot_traj_distri(ax=ax[2,2], x_axis=time_axis, metric_list=metric_list_resid, bc_process = 'Atevr_resid'   , ylabel=False, xlabel=True , legend=False, residual=True, title='Residual (true - prediction)')
        ## Cacluate RMSE of the accumulated cooling-tendencies
        Atsubsi_rmse  = calculate_rmse(df_tr['Atsubsi'] , df_tr_rf['Atsubsi_pred'])
        Atmeltsi_rmse = calculate_rmse(df_tr['Atmeltsi'], df_tr_rf['Atmeltsi_pred'])
        Atevr_rmse    = calculate_rmse(df_tr['Atevr']   , df_tr_rf['Atevr_pred'])

        ## Write RMSE in box
        if text_box:
            plot_text = True
            if text_box == 'rmse':
                text_Atsubsi  = f'RMSE: {Atsubsi_rmse:.3f}'  
                text_Atevr    = f'RMSE: {Atevr_rmse:.3f}'    
                text_Atmeltsi = f'RMSE: {Atmeltsi_rmse:.3f}'
            elif text_box == 'traj_count':
                mask = (df_tr['time'] == 0)
                traj_count = mask.sum()
                text_Atsubsi  = f'Traj. count: {traj_count:d}'  
                text_Atevr    = f'Traj. count: {traj_count:d}'    
                text_Atmeltsi = f'Traj. count: {traj_count:d}'
            else:
                print('Not correct text-box name, no text-box is plotted')
                plot_text = False
            
            if plot_text:
                plot_helpers.plot_shadow_box(ax=ax[0,2], x_pos=0.03, y_pos=0.9, textstr=text_Atsubsi)
                plot_helpers.plot_shadow_box(ax=ax[1,2], x_pos=0.03, y_pos=0.9, textstr=text_Atmeltsi)
                plot_helpers.plot_shadow_box(ax=ax[2,2], x_pos=0.03, y_pos=0.9, textstr=text_Atevr)

        ## Add label and grid for every plot
        labels = [['a','b','c'], ['d','e','f'], ['g','h','i']]
        for row in range(3):
            for col in range(3):
                ax[row,col].grid()
                label= labels[row][col]
                ax[row,col].text(0.012, 0.99, label, transform=ax[row,col].transAxes, fontsize=16, fontweight='bold', va='top')

    #=======================================================================================================================================================================================================================
    else:
    ## For ERA5 Data (No ground truth available)
    #-----
        if set_axes is not None:
            assert (isinstance(set_axes, list) or isinstance(set_axes, np.ndarray)) and len(set_axes[0]) == 3, \
                'set_axes must be a list or np.ndarray of length 3'
            ax = set_axes[0]
            current_row = set_axes[1]
            last_row_flag = set_axes[2]
            title_appendix = set_axes[3]
        else:
            fig,ax = plt.subplots(1,3, figsize=(14,4.5), sharey=True)
        
        if date:
            plt.suptitle(f'{date}')
        # Share y-axis for first two plots in each row
        # ax[0].get_shared_y_axes().join(ax[0], ax[0,1])
        # ax[1].get_shared_y_axes().join(ax[1], ax[1,1])

        # time_axis = np.arange(-48,1,1)#bcp_mean['time']
        time_axis = np.unique(df_tr['time'])
        metric_list_pred  = get_quantiles(df_tr_rf, bc_processes=['Atsubsi_pred', 'Atmeltsi_pred', 'Atevr_pred'])     #(function output-orer: bcp_mean, bcp_median, bcp_q25, bcp_q75, bcp_q10, bcp_q90, bcp_q1, bcp_q99)
        
        ## Call plotting function
        if set_axes:
            if current_row == 0:
                subplot_traj_distri(ax=ax[0], x_axis=time_axis, metric_list=metric_list_pred, bc_process = 'Atsubsi_pred' , ylabel=True, xlabel=False , set_ylim=set_ylim, legend=True , title=r'Pred $AQ_{subsi}$'+' '+title_appendix)
                subplot_traj_distri(ax=ax[1], x_axis=time_axis, metric_list=metric_list_pred, bc_process = 'Atmeltsi_pred', ylabel=False, xlabel=False , set_ylim=set_ylim, legend=False, title=r'Pred $AQ_{meltsi}$'+' '+title_appendix)
                subplot_traj_distri(ax=ax[2], x_axis=time_axis, metric_list=metric_list_pred, bc_process = 'Atevr_pred'   , ylabel=False, xlabel=False , set_ylim=set_ylim, legend=False, title=r'Pred $AQ_{evr}$'+' '+title_appendix)
            elif last_row_flag:
                subplot_traj_distri(ax=ax[0], x_axis=time_axis, metric_list=metric_list_pred, bc_process = 'Atsubsi_pred' , ylabel=True, xlabel=True , set_ylim=set_ylim, legend=False,  title=r'Pred $AQ_{subsi}$'+' '+title_appendix)
                subplot_traj_distri(ax=ax[1], x_axis=time_axis, metric_list=metric_list_pred, bc_process = 'Atmeltsi_pred', ylabel=False, xlabel=True , set_ylim=set_ylim, legend=False, title=r'Pred $AQ_{meltsi}$'+' '+title_appendix)
                subplot_traj_distri(ax=ax[2], x_axis=time_axis, metric_list=metric_list_pred, bc_process = 'Atevr_pred'   , ylabel=False, xlabel=True , set_ylim=set_ylim, legend=False, title=r'Pred $AQ_{evr}$'+' '+title_appendix)
            else:
                subplot_traj_distri(ax=ax[0], x_axis=time_axis, metric_list=metric_list_pred, bc_process = 'Atsubsi_pred' , ylabel=True, xlabel=False , set_ylim=set_ylim, legend=False, title=r'Pred $AQ_{subsi}$'+' '+title_appendix)
                subplot_traj_distri(ax=ax[1], x_axis=time_axis, metric_list=metric_list_pred, bc_process = 'Atmeltsi_pred', ylabel=False, xlabel=False , set_ylim=set_ylim, legend=False, title=r'Pred $AQ_{meltsi}$'+' '+title_appendix)
                subplot_traj_distri(ax=ax[2], x_axis=time_axis, metric_list=metric_list_pred, bc_process = 'Atevr_pred'   , ylabel=False, xlabel=False , set_ylim=set_ylim, legend=False, title=r'Pred $AQ_{evr}$'+' '+title_appendix)

        else:
            subplot_traj_distri(ax=ax[0], x_axis=time_axis, metric_list=metric_list_pred, bc_process = 'Atsubsi_pred' , ylabel=True, xlabel=True , set_ylim=set_ylim, legend=True , title=r'Predicted $AQ_{subsi}$')
            subplot_traj_distri(ax=ax[1], x_axis=time_axis, metric_list=metric_list_pred, bc_process = 'Atmeltsi_pred', ylabel=False, xlabel=True , set_ylim=set_ylim, legend=False, title=r'Predicted $AQ_{meltsi}$')
            subplot_traj_distri(ax=ax[2], x_axis=time_axis, metric_list=metric_list_pred, bc_process = 'Atevr_pred'   , ylabel=False, xlabel=True , set_ylim=set_ylim, legend=False, title=r'Predicted $AQ_{evr}$')

        ## Write RMSE in box
        if text_box:
            plot_text = True
            if text_box == 'traj_count':
                mask = (df_tr['time'] == 0)
                traj_count = mask.sum()
                text_Atsubsi  = f'Traj. count: {traj_count:d}'  
                text_Atevr    = f'Traj. count: {traj_count:d}'    
                text_Atmeltsi = f'Traj. count: {traj_count:d}'
            else:
                print('Not correct text-box name, no text-box is plotted')
                plot_text = False
            
            if plot_text:
                plot_helpers.plot_shadow_box(ax=ax[2], x_pos=0.03, y_pos=0.9, textstr=text_Atsubsi)

        ## Include a grid for every plot

        if set_axes is None:
            labels = ['a','b','c']
            for i in range(3):
                ax[i].grid()
                ax[i].text(0.012, 0.99, labels[i], transform=ax[i].transAxes, fontsize=16, fontweight='bold', va='top')
        else:
            labels = [['a','b','c'], ['d','e','f'], ['g','h','i']]
            for col in range(3):
                label = labels[current_row][col]
                ax[col].grid()
                ax[col].text(0.012, 0.99, label, transform=ax[col].transAxes, fontsize=16, fontweight='bold', va='top')
       
    #=======================================================================================================================================================================================================================

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=250)






#==========================================================================================================================================================================================================================================
#==========================================================================================================================================================================================================================================



def plot_prediction_vs_truth_2dhist(y_true, y_pred, title='', save_path=False, width_boxplot=0.25, count_per_boxplot=True):
    """
    Function for creating a scatter plot of actual vs predicted values and a residual plot, with optional save functionality.

    Parameters
    ----------
    y_true : array-like
        Array of actual values. Can be any array-like object compatible with matplotlib.

    y_pred : array-like
        Array of predicted values. Must be of the same shape as y_true and can be any array-like object compatible with matplotlib.

    title : str, optional
        Title for the plot. Default is an empty string.

    save_path : str, optional
        If provided, the function will save the plot to the location specified by this path. Default is False.

    width_boxplot : float, optional
        Width of the boxplots to be added to the residual plot. Default is 0.25.
    
    count_per_boxplot : Bool, optional
        Add a textbox with the number of samples within the range of one boxplot. Default is True

    Returns
    -------
    None. This function outputs a matplotlib plot and saves it to a file if a save_path is provided.

    Notes
    -----
    The function calculates Root Mean Square Error (RMSE) and R2 Score (R^2) metrics and displays these on the plot.
    The plot has two subplots: 
    1. A scatter plot with a 2D histogram showing actual vs predicted values.
    2. A scatter plot of residuals (differences between actual and predicted values) with (optinally) added boxplots.
    Boxplots are added to the residual plot in locations with a number of points greater than the minimum threshold (`min_points_boxplot`).
    Each boxplot spans a range of `width_boxplot` on the x-axis.
    The x and y axis limits of the subplots are set based on the `right_edge` and `cooling_limit` variables, which are calculated based on the minimum of the actual and predicted values.
    """



    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Create a scatter plot of actual vs predicted values
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

    ##----------------------------------------------------------------------
    ## Plot comparison plot
    ##---------------------
    plt.suptitle(title)
    # Scatter plot with hist2d
    hist = ax[0].hist2d(y_pred, y_true, bins=40, cmap='Blues', norm=colors.LogNorm(vmax=1e3), )
    # Create a colorbar for the 2D histogram
    cbar = plt.colorbar(hist[3], ax=ax[0])
    cbar.set_label('Counts')
    cbar.ax.yaxis.set_minor_locator(plt.LogLocator(base=10, subs=(0.2, 0.4, 0.6, 0.8)))
    
    ax[0].set_xlabel('Predicted Cooling [K/h]')
    ax[0].set_ylabel('Actual Cooling [K/h]')

    ## Handle the axes-extend for different cooling magnitudes (accumulated and instantaneous) Round to next 10th if cooling is biiger than 10 (12->20), else round to next 5 (7 -> 10)
    max_cooling = np.min([y_true.min(), y_pred.min()])
    right_edge, cooling_limit = (0.5, np.floor(max_cooling / 10)*10) if max_cooling < -10 else (0.1, np.floor(max_cooling))
    ax[0].set_xlim(cooling_limit, right_edge)
    ax[0].set_ylim(cooling_limit, right_edge)
    ax[0].plot([cooling_limit,right_edge], [cooling_limit,right_edge], linewidth=1.5, c='k', dashes=(5,5), alpha=0.75)


    # Add text box with metrics
    textstr = f'RMSE: {rmse:.3f}\n$R^2$     : {r2:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax[0].text(0.05, 0.95, textstr, transform=ax[0].transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    ax[0].grid(zorder=0)

    #=================================================================================================================================================
    ## Plot residual plot
    #-------------------------------------------------------------------------------------------------------------------------------------------------

    residuals = y_true - y_pred

    # Create a scatter plot of residuals
    sns.scatterplot(x=y_true, y=residuals, ax=ax[1], alpha=0.5, color='lightslategray')
    ax[1].set_xlabel('Actual Cooling [K/h]')
    ax[1].set_ylabel('Residual [K/h]')
    max_residuals = np.ceil(np.max(np.abs( residuals))/2)*2
    ax[1].set_xlim(cooling_limit, right_edge)
    ax[1].set_ylim(-max_residuals, max_residuals)
    ax[1].hlines(y=0, xmin=cooling_limit, xmax=right_edge, linestyle='--', color='k', alpha=0.75)
    #---------------------------------------------------------
    ## PLOT BOXPLOTS THAT SHOW DISTRIBUTION OF RESIUDAL POINTS
    plot_helpers.add_boxplots(x=y_true, y=residuals, width_boxplot=width_boxplot, ax=ax[1], min_points_boxplot=100, count_per_boxplot=count_per_boxplot, fontsize_text=10, input_detail_mode=False, )
    #---------------------------------------------------------
    ax[1].grid(zorder=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path,bbox_inches='tight',dpi=200)




## If possible use directly the function given by plot.helpers module
def get_qq_plot(y_true, y_pred, bcp='Atmeltsi', save_path=False, title=None):
    fig,ax = plt.subplots(figsize=(8,8))
    # plt.scatter(q_pred, q_val, color='k', s=10, alpha=0.5)
    nr_points=plot_helpers.plot_modified_qq_plot(ax, y_pred, y_true, subsample_factor=0.1)
    if title: 
        ax.set_title(f'QQ plot: {bcp} (# qunatiles: {nr_points:2.1e} from {len(y_pred):2.1e} total samples)')
    if save_path:
        plt.savefig(save_path,bbox_inches='tight', dpi=200)



#==========================================================================================================================================================================================================================================
#==========================================================================================================================================================================================================================================





def plot_synoptic_accumulated_cooling(df_center, df_center_pred, month=None, cooling_var='Atsubsi', date=None, save_fig=False, composite=False):

    """
    Plots the accumulated cooling at two different pressure levels for extratropical cyclones.

    Parameters
    ----------
    df_center : DataFrame
        DataFrame containing trajectory output data including longitude, latitude, pressure and accumulated cooling values.
    df_center_pred : DataFrame
        DataFrame containing predicted trajectory output data.
    month : str or int, optional
        Specifies the month for which the plot is generated.
    cooling_var : str, default='Atsubsi'
        The specific cooling process to focus on, either sublimation or melting.
    date : str or datetime, optional
        Specific date for which the plot is generated. 
    save_fig : str or bool, default=False
        If a string is provided, it is assumed to be the path and filename where the figure should be saved. If False, the figure is not saved.
    composite : bool, default=False
        If True, data is scaled to make them evenly spaced in terms of longitude, which is useful for higher latitudes that have more points in the longitude direction.

    Returns
    -------
    None
        The function generates a plot and optionally saves it, but does not return any values.

    Notes
    -----
    The function generates a multi-panel plot for each pressure level (1000-850 hPa and 850-700 hPa). The left panels show the actual accumulated cooling, the middle panels show the predicted accumulated cooling, 
    and the right panels show the residuals (difference between actual and predicted cooling). The contour plot in the background represents the sea level pressure.

    The term 'accumulated cooling' here refers to the sum of the cooling effect along a backward trajectory in the study of extratropical cyclones.

    """

    #==========================================================================================================================
    # Define the colors for our new colormap
    #---------------------------------------
    # Define widht of white space
    white_fraction = 1/16
    #---------------------------------------

    c_frac = int(1/white_fraction)
    c_len = int(100*c_frac)
    c_mid = int(100*c_frac / 2)


    cmap_seismic = plt.cm.get_cmap('seismic')                                                                               
    # Define the colors for our new colormap                                                                                
    colors = [(cmap_seismic(i/c_len)) for i in range(c_mid)] + [(1,1,1,1)]*100 + [(cmap_seismic(i/c_len)) for i in range(c_mid, c_len)]
    cmap_residual = mcolors.LinearSegmentedColormap.from_list("new_PRGn", colors)     
    #==========================================================================================================================




    if composite:
        fig_size = (12,8)
    else:
        fig_size = (18, 8)

    # Create a gridspec to handle layout
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 0.05])
    fig = plt.figure(figsize=fig_size)  


    if date:
        plt.suptitle(f'{date}')
        ifs_path = f'/net/helium/atmosdyn/IFS-1Y/{month}/cdf/P{date}'
        slp = xr.open_dataset(ifs_path)
        slp = slp.SLP.squeeze()

    df_lowlevel = df_center[(df_center['p'] > 850) & (df_center['p'] < 1000)]
    df_midlevel = df_center[(df_center['p'] <= 850) & (df_center['p'] > 700)]

    df_lowlevel_pred = df_center_pred[(df_center_pred['p'] > 850) & (df_center_pred['p'] < 1000)]
    df_midlevel_pred = df_center_pred[(df_center_pred['p'] <= 850) & (df_center_pred['p'] > 700)]

    df_2d_lowlevel = df_lowlevel.groupby(['lon', 'lat'])[['Atsubsi', 'Atmeltsi']].mean().reset_index()
    df_2d_midlevel = df_midlevel.groupby(['lon', 'lat'])[['Atsubsi', 'Atmeltsi']].mean().reset_index()

    df_2d_lowlevel_pred = df_lowlevel_pred.groupby(['lon', 'lat'])[['Atsubsi_pred', 'Atmeltsi_pred']].mean().reset_index()
    df_2d_midlevel_pred = df_midlevel_pred.groupby(['lon', 'lat'])[['Atsubsi_pred', 'Atmeltsi_pred']].mean().reset_index()

    # df_2dcenter = df_center.groupby(['lon', 'lat'])[['Atsubsi', 'Atmeltsi']].mean().reset_index()
    # df_2dcenter_pred = df_center_pred.groupby(['lon', 'lat'])[['Atsubsi_pred', 'Atmeltsi_pred']].mean().reset_index()

    # Define data sets and variable names
    data_sets = [[df_2d_lowlevel, df_2d_lowlevel_pred], [df_2d_midlevel, df_2d_midlevel_pred]]

    var_names = [[f'{cooling_var}', f'{cooling_var}_pred'], [f'{cooling_var}', f'{cooling_var}_pred']]
    
    
    titles = [[f'Acc. Cooling {cooling_var} (1000-850 hPa)', f'Predicted Acc. Cooling {cooling_var} (1000-850 hPa)'],
              [f'Acc. Cooling {cooling_var} (850-700 hPa)', f'Predicted Acc. Cooling {cooling_var} (850-700 hPa)']]
    
    colorbar_labels = [f'{cooling_var}', f'{cooling_var}']


    vmins = [[-5,-5],[-5,-5]]
    vmaxs = [[0,0],[0,0]]

    # Add titles and colorbar labels for the difference plots
    diff_titles = [f'Residual ({cooling_var} - {cooling_var}_pred) (1000-850 hPa)', f'Residual ({cooling_var} - {cooling_var}_pred) (850-700 hPa)']
    colorbar_diff_labels = [f'Δ {cooling_var}', 'Δ {cooling_var}']


    lon_min = np.floor(df_center.lon.min() - 3)
    lon_max = np.ceil(df_center.lon.max() + 3)

    lat_min = np.floor(df_center.lat.min() - 3)
    lat_max = np.ceil(df_center.lat.max() + 3)
    
    
    sc_list, sc_diff_list = [], []
    for i in range(2):
        for j in range(2):
            ax = plt.subplot(gs[i, j])  # create subplot using gridspec
            data = data_sets[i][j]
            var = var_names[i][j]

            if composite:
                m = Basemap(projection='cyl', resolution=None,
                            llcrnrlat=lat_min, urcrnrlat=lat_max,
                            llcrnrlon=lon_min, urcrnrlon=lon_max, ax=ax)

            else:
                CS = slp.plot.contour(ax=ax, alpha=0.5, colors='k', zorder=0, levels=np.arange(slp.min(),slp.max(),5))
                m.drawcoastlines()
                m.drawcountries()


                m = Basemap(projection='cyl', resolution='l',
                            llcrnrlat=lat_min, urcrnrlat=lat_max,
                            llcrnrlon=lon_min, urcrnrlon=lon_max, ax=ax)

            m.drawparallels(np.arange(-90., 91., 5.), labels=[True, False, False, False],
                            ax=ax)  
            m.drawmeridians(np.arange(-180., 181., 5.), labels=[False, False, False, True],
                            ax=ax)  

            ax.grid(True, alpha=0.5)  # Add grid

            ax.set_title(titles[i][j])

            vmin, vmax = vmins[i][j], vmaxs[i][j] 
            bounds = np.arange(vmin, vmax+0.001, 0.25)
            my_cmap = plt.cm.get_cmap('plasma', len(bounds))
            # create a BoundaryNorm instance
            norm = BoundaryNorm(bounds, my_cmap.N)

            x, y = m(data['lon'].values, data['lat'].values) 
            sc = ax.scatter(x, y, c=data[var], cmap=my_cmap, norm=norm, marker='s', alpha=0.85 )
            sc_list.append(sc)


        # Add colorbar for each row
        cax1 = plt.subplot(gs[2, 0:2])  
        # cax1 = fig.add_axes([0.25, 0.15, 0.25, 0.02]) # left, bottom, width, height
        cbar = fig.colorbar(sc_list[0], cax=cax1, orientation='horizontal', extend='min', shrink=0.5)  # Change orientation to horizontal
        cbar.set_label(f'{colorbar_labels[0]} [K]')

        # Add difference plot in 4th column
        ax_diff = plt.subplot(gs[i, 2])
        
        
        diff = data_sets[i][0][var_names[i][0]] - data_sets[i][1][var_names[i][1]]

        m = Basemap(projection='cyl', resolution='l',
                    llcrnrlat=lat_min, urcrnrlat=lat_max,
                    llcrnrlon=lon_min, urcrnrlon=lon_max, ax=ax)

        ax_diff.grid(True, alpha=0.5)  # Add grid

        if date:
            slp.plot.contour(ax=ax_diff, alpha=0.5, colors='k', zorder=0, levels=np.arange(slp.min(),slp.max(),5))

        m.drawcoastlines()
        m.drawcountries()
        m.drawparallels(np.arange(-90., 91., 5.), labels=[True, False, False, False],
                        ax=ax_diff)  # Draw parallels (latitude lines) for values (in degrees).
        m.drawmeridians(np.arange(-180., 181., 5.), labels=[False, False, False, True],
                        ax=ax_diff)  # Draw meridians (longitude lines) for values (in degrees).

        ax_diff.set_title(diff_titles[i])

        x, y = m(data['lon'].values, data['lat'].values) 
        sc_diff = ax_diff.scatter(x, y, c=diff, cmap=cmap_residual, marker='s', vmin=-2, vmax=2, edgecolors='k', linewidths=0.1) 

        sc_diff_list.append(sc_diff)

        # Add colorbar for difference plot
        cax2 = plt.subplot(gs[2, 2])  # place colorbar in 3rd row, 3rd column
        cbar_diff = fig.colorbar(sc_diff_list[0], cax=cax2, orientation='horizontal', extend='both')  # Change orientation to horizontal
        cbar_diff.set_label(f'{colorbar_diff_labels[0]} [K]')

    
    plt.tight_layout(h_pad=4, w_pad=1)

    if save_fig:
        print(f'Save figure.')
        fig.savefig(save_fig, bbox_inches='tight', dpi=150)





#==========================================================================================================================================================================================================================================
#==========================================================================================================================================================================================================================================


def plot_pred_vs_true(ax, y_true, y_pred, xrange=None, xaxis_label=True, ylabel_bcp='', unit='[K/h]', vmax=1e3):
    # Scatter plot with hist2d
    if vmax >= 100:
        hist = ax.hist2d(y_pred, y_true, bins=40, cmap='Blues', norm=colors.LogNorm(vmax=vmax), )
    else:
        hist = ax.hexbin(y_pred, y_true, gridsize=50, cmap='Blues', vmax=vmax )

    if xaxis_label:
        ax.set_xlabel(f'Predicted Cooling {unit}')
    ax.set_ylabel(ylabel_bcp)
    
    if xrange is None:
        xrange = -5    
    ax.set_xlim(xrange, 0.1)
    ax.set_ylim(xrange, 0.1)
    ax.plot([xrange,1], [xrange,1], linewidth=1.5, c='k', linestyle='--', dashes=(5,5), alpha=0.75)
    
    ## Add text box with metrics
    if y_true.shape[0] >= 1000:
        count_str = f"{y_true.shape[0]//1000}k"  # Compact representation for large counts
    else:
        count_str = str(y_true.shape[0])
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # if r2 is None:
    #     textstr = (f'N: {count_str}\n'
    #                 f'RMSE: {rmse:.3f}')
    # else:
    textstr = (f'N: {count_str}\n'
            f'RMSE: {rmse:.3f}\n'
            f'$R^2$:       {r2:.3f}')
        
    props = dict(boxstyle='round, pad=0.3, rounding_size=0.2', facecolor='white', edgecolor='gray', alpha=0.8)
    shadowprops = dict(boxstyle='round, pad=0.35, rounding_size=0.2', facecolor='gray', alpha=0.2)
    # Shadow text (for a shadow effect)
    ax.text(0.027, 0.898, textstr, transform=ax.transAxes, fontsize=11, 
            fontname='Arial', ha='left', va='top', bbox=shadowprops)
    # Actual text
    ax.text(0.025, 0.9, textstr, transform=ax.transAxes, fontsize=11, 
            fontname='Arial', ha='left', va='top', bbox=props)
    
    ## Set Grid
    ax.grid(zorder=0)

    return hist


def plot_residuals(ax, 
                   y_true, 
                   y_pred, 
                   width_boxplot=0.25, 
                   min_points_boxplot=100,
                   xrange=None, yrange=None, 
                   xaxis_label=True, ylabel_bcp='', unit='[K/h]'):
    ## Calculate residuals
    residuals = y_true - y_pred
    # Create a scatter plot of residuals
    sns.scatterplot(x=y_true, y=residuals, alpha=0.5, color='lightslategray', ax=ax)
    ax.set_ylabel(ylabel_bcp)
    if xaxis_label: 
        ax.set_xlabel(f'Actual Cooling {unit}')
    else:
        ax.set_xlabel('')
    
    if xrange is None:
        xrange = np.round(min(y_true))
    if yrange is None:
        yrange = max(abs(min(residuals)), max(residuals))
    
    ax.set_xlim(xrange, 0.1)
    ax.set_ylim(-yrange, yrange)
    ax.hlines(y=0, xmin=xrange, xmax=0.1, linestyle='--', color='k', alpha=0.75,  linewidth=1.5) 
    plot_helpers.add_boxplots(x=y_true,y=residuals, ax=ax, 
                              width_boxplot=width_boxplot, min_points_boxplot=min_points_boxplot,
                                count_per_boxplot=True, fontsize_text=10, input_detail_mode=False)
    ax.grid(zorder=0)














def plot_time_series(df_tr_ifs, df_tr_era, set_ylim=None, plot_text=True,save_fig=False):    
    """
    This function is to create a figure fopr the thesis and largely builds on plot_accu_cooling_over_time
    It encorperates IFS, IFS predictions and ERA5 prediciton
    """

    fig,ax = plt.subplots(3,3, figsize=(12,12), sharex=False, sharey=False)

        
    # Share y-axis for first two plots in each row
    # ax[0,0].get_shared_y_axes().join(ax[0,0], ax[0,1])
    # ax[1,0].get_shared_y_axes().join(ax[1,0], ax[1,1])
    # ax[1,0].get_shared_y_axes().join(ax[2,0], ax[2,1])

    # time_axis = np.arange(-48,1,1)#bcp_mean['time']
    time_axis = np.unique(df_tr_ifs['time'])


    metric_list           = get_quantiles(df_tr_ifs, bc_processes=['Atsubsi', 'Atmeltsi', 'Atevr'])                      #(function output-orer: bcp_mean, bcp_median, bcp_q25, bcp_q75, bcp_q10, bcp_q90, bcp_q1, bcp_q99)
    metric_list_pred_ifs  = get_quantiles(df_tr_ifs, bc_processes=['Atsubsi_pred', 'Atmeltsi_pred', 'Atevr_pred'])
    metric_list_pred_era  = get_quantiles(df_tr_era, bc_processes=['Atsubsi_pred', 'Atmeltsi_pred', 'Atevr_pred'])

    ## Call plotting function
    # Plot IFS cplumn
    subplot_traj_distri(ax=ax[0,0], x_axis=time_axis, metric_list=metric_list, bc_process = 'Atsubsi' , ylabel=True, xlabel=False, legend=True  , title=r'IFS $AQ_{subsi}$')
    subplot_traj_distri(ax=ax[1,0], x_axis=time_axis, metric_list=metric_list, bc_process = 'Atmeltsi', ylabel=True, xlabel=False, legend=False , title=r'IFS $AQ_{meltsi}$')
    subplot_traj_distri(ax=ax[2,0], x_axis=time_axis, metric_list=metric_list, bc_process = 'Atevr'   , ylabel=True,  xlabel=True , legend=False , title=r'IFS $AQ_{evr}$')
    # For predictions (tmeltsi and tsubsi and tevr)
    subplot_traj_distri(ax=ax[0,1], x_axis=time_axis, metric_list=metric_list_pred_ifs, bc_process = 'Atsubsi_pred' , ylabel=False , xlabel=False, legend=False, title=r'IFS Prediction $AQ_{subsi}$')
    subplot_traj_distri(ax=ax[1,1], x_axis=time_axis, metric_list=metric_list_pred_ifs, bc_process = 'Atmeltsi_pred', ylabel=False, xlabel=False, legend=False, title=r'IFS Prediction $AQ_{meltsi}$')
    subplot_traj_distri(ax=ax[2,1], x_axis=time_axis, metric_list=metric_list_pred_ifs, bc_process = 'Atevr_pred'   , ylabel=False, xlabel=True , legend=False, title=r'IFS Prediction $AQ_{evr}$')
    # For ERA5 (tmeltsi and tsubsi and tevr)
    subplot_traj_distri(ax=ax[0,2], x_axis=time_axis, metric_list=metric_list_pred_era, bc_process = 'Atsubsi_pred' , ylabel=False, xlabel=False , set_ylim=set_ylim, legend=False , title=r'ERA5 Prediction $AQ_{subsi}$')
    subplot_traj_distri(ax=ax[1,2], x_axis=time_axis, metric_list=metric_list_pred_era, bc_process = 'Atmeltsi_pred', ylabel=False, xlabel=False , set_ylim=set_ylim, legend=False, title=r'ERA5 Prediction $AQ_{meltsi}$')
    subplot_traj_distri(ax=ax[2,2], x_axis=time_axis, metric_list=metric_list_pred_era, bc_process = 'Atevr_pred'   , ylabel=False, xlabel=True , set_ylim=set_ylim, legend=False, title=r'ERA5 Prediction $AQ_{evr}$')
    


    if plot_text:
        mask = (df_tr_ifs['time'] == 0)
        traj_count = mask.sum()
        text_ifs  = f'IFS Traj. count: {traj_count:d}'  
        mask = (df_tr_era['time'] == 0)
        traj_count = mask.sum()   
        text_era = f'ERA5 Traj. count: {traj_count:d}'

        plot_helpers.plot_shadow_box(ax=ax[2,0], x_pos=0.03, y_pos=0.83, textstr=text_ifs)
        plot_helpers.plot_shadow_box(ax=ax[2,1], x_pos=0.03, y_pos=0.83, textstr=text_ifs)
        plot_helpers.plot_shadow_box(ax=ax[2,2], x_pos=0.03, y_pos=0.83, textstr=text_era)




    ## Add label and grid for every plot
    labels = [['a','b','c'], ['d','e','f'], ['g','h','i']]
    for row in range(3):
        for col in range(3):
            ax[row,col].grid()
            label= labels[row][col]
            ax[row,col].text(0.012, 0.99, label, transform=ax[row,col].transAxes, fontsize=16, fontweight='bold', va='top')


    if save_fig:
        print('Figure is saved')
        plt.savefig(save_fig, bbox_inches='tight',dpi=150)