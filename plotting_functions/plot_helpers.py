import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.colors as colors
from sklearn.metrics import r2_score, mean_squared_error


###############################################
## This script contains functions that are often used across different topics 
    # - Model evaluation
    # - Trajectory evaluation







def plot_shadow_box(ax, x_pos, y_pos, textstr, fontsize=11, fontname='Arial',ha='left', va='top'):
    props = dict(boxstyle='round, pad=0.3, rounding_size=0.2', facecolor='white', edgecolor='gray', alpha=0.8)
    shadowprops = dict(boxstyle='round, pad=0.35, rounding_size=0.2', facecolor='gray', alpha=0.2)  # Shadow text (for a shadow effect)
    ax.text(x_pos, y_pos, textstr, transform=ax.transAxes, fontsize=fontsize, 
            fontname=fontname, ha=ha, va=va, bbox=shadowprops)
    # Actual text
    ax.text(x_pos, y_pos, textstr, transform=ax.transAxes, fontsize=fontsize, 
            fontname=fontname, ha=ha, va=va, bbox=props)





def add_boxplots(x, y, width_boxplot, ax, min_points_boxplot=100, count_per_boxplot=True, fontsize_text=10, input_detail_mode=False, ):
    """
    This function adds boxplots to an existing matplotlib axis at positions determined by the values in 'x'. 

    Parameters:
    x (array-like): Sequence of numerical data.
    y (array-like): Sequence of numerical data  (values to be boxed).
    width_boxplot (float): Width of the boxplots.
    ax (matplotlib Axes object): The Axes object to which the boxplots are added.
    min_points_boxplot (int, optional): The minimum number of data points required to generate a boxplot. 
                                        If the number of data points in a bin is less than this, the boxplot is skipped. 
                                        Default value is 100.
    count_per_boxplot (bool, optional): If True, the count of data points in each boxplot is displayed above the boxplot.
                                        Default value is True.
    input_detail_mode (bool, optional): If True, the function will prompt for user inputs to modify visual aspects such as colors, fontsize, etc. 
                                        If False, these details will be set to default values. 
                                        Default value is False.

    Returns:
    None

    This function modifies the provided Axes object (ax) in-place by adding boxplots. 
    The boxplots are grouped by bin, with the width of every boxplot determined by the 'width_boxplot' argument. 
    Each boxplot's position is at the mid-point of the corresponding bin. 
    """
    
    if input_detail_mode:
        box_color = input("Enter box color: ")
        box_alpha = float(input("Enter box alpha (0 to 1): "))
        line_width = float(input("Enter line width: "))
    else:
        box_color = 'skyblue'
        box_alpha = 0.75
        line_width = 1.5
        

    df_boxplot = pd.DataFrame({'X_value': x, 'Y_value': y})
    
    ## Define max and min value
    min_value, max_value = np.nanmin(x), np.nanmax(x)

    ## Define bins and assign every data-pari to a bin-range
    bins = np.arange(np.floor(min_value), max_value+width_boxplot, width_boxplot) 
    df_boxplot['bin'] = pd.cut(df_boxplot['X_value'], bins=bins, labels=bins[:-1])


    # Group the data by bin and iterate over the groups
    for (bin_val, group) in df_boxplot.groupby('bin'):
        count = group['Y_value'].count()
        # Skip this group if it has less than the minimum number of points
        if count < min_points_boxplot:
            continue

        # Calculate the position of the boxplot (mid-point of each bin)
        position = bin_val + width_boxplot/2
        # Create a boxplot for the group at the calculated position
        boxplot = ax.boxplot(group['Y_value'], positions=[position], widths=width_boxplot, patch_artist=True, showfliers=False, manage_ticks=False)
        
        # Change the color of the boxes
        for patch in boxplot['boxes']:
            patch.set_facecolor(box_color)
            patch.set_alpha(box_alpha) 
            patch.set_linewidth(line_width)
        
        # Make the median line thicker
        for median in boxplot['medians']:
            median.set_linewidth(line_width+1)
        
        # Increase the linewidth of the whiskers
        for whisker in boxplot['whiskers']:
            whisker.set_linewidth(line_width)

        # Increase the linewidth of the caps
        for cap in boxplot['caps']:
            cap.set_linewidth(line_width)
            

        # Add text above the boxplot with the count of points
        if count_per_boxplot:
            if count >= 1000:
                count_str = f"{count//1000}k"  # Compact representation for large counts
            else:
                count_str = str(count)
            # ax.text(position, ax.get_ylim()[1]*0.98, f'N={count_str}', ha='center', va='top', rotation=90, fontsize=12)
            ax.text(position, ax.get_ylim()[1]*0.95, f' N={count_str}', ha='center', va='top', rotation=90, fontsize=fontsize_text,
                bbox=dict(boxstyle="round,pad=0.3", fc=(1, 1, 1, 0.8), ec="black", lw=1))  # Increase the fontsize here








def plot_modified_qq_plot(ax, y_pred, y_true, subsample_factor=1, show_ylabel=True, unit='[K/hr]'):
    """
    This function generates a modified quantile-quantile (Q-Q) plot where the quantiles of 
    the actual values are plotted against the quantiles of the predicted values. 

    Parameters:
    - ax : matplotlib Axes
        The Axes object to draw the plot onto.

    - y_pred : array-like
        The array containing the predicted values.

    - y_true : array-like
        The array containing the actual (true) values.
        
    - subsample_factor : float, optional
        The fraction of point that will be plotted (default=1 -> all points are plotted)

    The function plots a scatter plot of the quantiles, highlights specific 
    percentiles (0.1th, 1st, 10th, 25th, and the median) with differently colored points, 
    and adds a red dashed line for the perfect theoretical Q-Q relationship (y = x). 

    Returns:
    - nr_points : int
        The number of points that were plotted.

    Note:
    The quantiles are calculated from the sorted data, so this function can handle missing values 
    (which are sorted to the end of the array by numpy.sort()).
    """

    # Calculate the percentiles
    y_val_sorted = np.sort(y_true)
    y_pred_sorted = np.sort(y_pred)

    # Generate percentiles
    nr_points = int(len(y_pred) * subsample_factor)
    percentiles = np.linspace(0, 100, nr_points ) 

    # Calculate the Q-Q pairs
    q_val = np.percentile(y_val_sorted, percentiles)
    q_pred = np.percentile(y_pred_sorted, percentiles)

    # Plot scatter of prediciton vs true values
    ax.scatter(q_pred, q_val, color='k', s=10, alpha=0.5)


    median_pred = np.median(q_pred)
    median_val = np.median(q_val)
    ax.scatter(median_pred, median_val, color='blue', edgecolor='k', label='median', s=100, zorder=5, alpha=0.95)

    p25_pred = np.percentile(q_pred, 25)
    p25_val = np.percentile(q_val, 25)
    ax.scatter(p25_pred, p25_val, color='green', edgecolor='k', label='25th percentile', s=100, zorder=5, alpha=0.95)

    p10_pred = np.percentile(q_pred, 10)
    p10_val = np.percentile(q_val, 10)
    ax.scatter(p10_pred, p10_val, color='red', edgecolor='k', label='10th percentile', s=100, zorder=5, alpha=0.95)


    p1_pred = np.percentile(q_pred, 1)
    p1_val = np.percentile(q_val, 1)
    ax.scatter(p1_pred, p1_val, color='yellow', edgecolor='k', label='1st percentile', s=100, zorder=5, alpha=0.95)

    p01_pred = np.percentile(q_pred, 0.1)
    p01_val = np.percentile(q_val, 0.1)
    ax.scatter(p01_pred, p01_val, color='white', edgecolor='k', label='0.1 percentile', s=100, zorder=5, alpha=0.95)

    # Diagonal line
    diag_min, diag_max = np.min((q_pred.min(),q_val.min())), np.max((q_pred.max(),q_val.max()))
    ax.plot([diag_min, diag_max], [diag_min, diag_max], '--', color='k', dashes=(5,5), alpha=0.75)

    if show_ylabel:
        ax.set_ylabel(f'True quantiles {unit}')
    ax.set_xlabel(f'Predicted quantiles {unit}')
    ax.grid(zorder=0)
    # ax.legend(bbox_to_anchor=(0.0, 0.45, 0.5, 0.5, alignment='left'), shadow=True, edgecolor='gray', ) 
    ax.legend(loc='upper left', bbox_to_anchor=(0, 0.96), shadow=True, edgecolor='gray')
    
    return nr_points









def plot_pred_vs_true(ax, y_true, y_pred, xrange=None, xaxis_label=True, ylabel_bcp=''):
    # Scatter plot with hist2d
    hist = ax.hist2d(y_pred, y_true, bins=40, cmap='Blues', norm=colors.LogNorm(vmax=1e3), )
    if xaxis_label:
        ax.set_xlabel('Predicted Cooling [K/hr]')
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


def plot_residuals(ax, y_true, y_pred, width_boxplot=0.25, xrange=None, yrange=None, xaxis_label=True, ylabel_bcp=''):
    ## Calculate residuals
    residuals = y_true - y_pred
    # Create a scatter plot of residuals
    sns.scatterplot(x=y_true, y=residuals, alpha=0.5, color='lightslategray', ax=ax)
    ax.set_ylabel(ylabel_bcp)
    if xaxis_label: 
        ax.set_xlabel('Actual Cooling [K/hr]')
    else:
        ax.set_xlabel('')
    
    if xrange is None:
        xrange = np.round(min(y_true))
    if yrange is None:
        yrange = max(abs(min(residuals)), max(residuals))
    
    ax.set_xlim(xrange, 0.1)
    ax.set_ylim(-yrange, yrange)
    ax.hlines(y=0, xmin=xrange, xmax=0.1, linestyle='--', color='k', alpha=0.75,  linewidth=1.5) 
    add_boxplots(x=y_true,y=residuals, ax=ax, width_boxplot=width_boxplot, min_points_boxplot=100, count_per_boxplot=True, fontsize_text=10, input_detail_mode=False)
    ax.grid(zorder=0)






