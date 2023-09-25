## Import modules

import numpy as np 
import matplotlib.pyplot as plt 
import pickle

from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import metpy.calc as mpcalc
from metpy.interpolate import cross_section

#==================================================================================================================================================================================================================
#==================================================================================================================================================================================================================
#==================================================================================================================================================================================================================

def plot_cross_section(cross_p, 
                       data_p, 
                       start, 
                       end,
                       plot_contourf,
                       plot_contour,
                       ax=None,
                       show_wind_barbs=False,
                       show_clouds=False,
                       show_precip=False,
                       contourf_colorbar=True,
                       inset_contourf='all_bcp_sums',
                       rf_bcp = False,
                       baseline_bcp = None,
                       ):
    ## Define what happens if fiels is not defined
    if ax is None:
        fig, ax = plt.subplots()
    if plot_contourf is None:
        plot_contourf = []
    if plot_contour is None:
        plot_contour = []


    #---------------------------------------------------------------
    # Determine if cross-section x-axis is shown in lon or lat
    lon_difference = abs(max(cross_p['lon']) - min(cross_p['lon']))
    lat_difference = abs(max(cross_p['lat']) - min(cross_p['lat']))
    
    x_axis_name = 'lon'
    if lon_difference >= lat_difference:
        x_axis = cross_p['lon']
    else:
        #ax.invert_xaxis()
        x_axis = cross_p['lat'] 
        x_axis_name = 'lat'
    #---------------------------------------------------------------        

    #====================================================================================================================================
    ###       Properties of plotted fields:
    #----------------------------------------------
    plasma_r = plt.cm.get_cmap('plasma_r', 100)
    skyblue = plt.cm.get_cmap('Blues_r', 40)

    colors0 = np.linspace([1, 1, 1, 1], [1, 1, 1, 1], 10)   #np.array([[1, 1, 1, 1]])  # White color
    white_to_yellow = np.linspace([1, 1, 1, 1], plasma_r(0), 30)
    colors1 = white_to_yellow
    colors2 = plasma_r(np.linspace(0, 1, 60))
    colors3 = skyblue(np.linspace(0.1, 1, 40))

    colors = np.vstack((colors0, colors1, colors2, colors3))
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    
    ## Define properties for most common contourf plots 
    contourf_properties = {
        'tevr'   :  {'levels': np.arange(-0.5, -0.049, 0.025) , 'cmap': 'YlGn_r'   , 'alpha': 0.75,  'cbar_title': 'T evR [K/hr]'     , 'extend':'min'},
        'tsubsi' :  {'levels': np.arange(-1.5, -0.074, 0.025) , 'cmap': 'inferno'  , 'alpha': 0.75,  'cbar_title': 'T sub S&I [K/hr]' , 'extend':'min'},
        'tmeltsi':  {'levels': np.arange(-1.5, -0.074, 0.025) , 'cmap': 'Blues_r'  , 'alpha': 0.75,  'cbar_title': 'T melt S&I [K/hr]', 'extend':'min'},
        'RH_ifs' :  {'levels': np.linspace(0,140,15)          , 'cmap': custom_cmap, 'alpha': 0.3 ,  'cbar_title': 'RH_ifs in %'      , 'extend':'neither'},
        'OMEGA'  :  {'levels': np.linspace(-1,1,10)           , 'cmap':'PuOr'      , 'alpha': 0.2 ,  'cbar_title': 'Omega [Pa/s]'           , 'extend':'both'},

        'tevr_pred'   :  {'levels': np.arange(-0.5, -0.049, 0.025) , 'cmap': 'YlGn_r'   , 'alpha': 0.75,  'cbar_title': 'Pred TevR [K/hr]'     , 'extend':'min'},
        'tsubsi_pred' :  {'levels': np.arange(-1.5, -0.074, 0.025) , 'cmap': 'inferno'  , 'alpha': 0.75,  'cbar_title': 'Pred Tsub S&I [K/hr]' , 'extend':'min'},
        'tmeltsi_pred':  {'levels': np.arange(-1.5, -0.074, 0.025) , 'cmap': 'Blues_r'  , 'alpha': 0.75,  'cbar_title': 'Pred Tmelt S&I [K/hr]', 'extend':'min'},

        'residual_tevr'   :  {'levels': np.arange(-1.0, 1.1, 0.1) , 'cmap': 'PRGn'      , 'alpha': 0.75,  'cbar_title': 'Residual TevR (true-pred)'   , 'extend':'both'},
        'residual_tsubsi' :  {'levels': np.arange(-1.0, 1.1, 0.1) , 'cmap': 'seismic'   , 'alpha': 0.75,  'cbar_title': 'Residual Tsubsi (true-pred)' , 'extend':'both'},
        'residual_tmeltsi':  {'levels': np.arange(-1.0, 1.1, 0.1) , 'cmap': 'RdYlGn'    , 'alpha': 0.75,  'cbar_title': 'Residual Tmeltsi (true-pred)', 'extend':'both'},
    }

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    ## Define properties of contour plots 
    contour_properties = {
        'RH': {
            'variable_name': 'RH',
            'levels': np.arange(0, 101, 10),
            'cmap': 'Blues',
            'alpha': 0.75,
            'linewidths': 1,
            'linestyles' : '-',
        },
        'RH_ifs': {
            'variable_name': 'RH_ifs',
            'levels': [80,100],
            'colors': 'blue',
            'alpha': 0.75,
            'linewidths': 1,
            'linestyles' : '-',
        },
        'TH': {
            'variable_name': 'TH',
            'levels': np.arange(270, 365, 10),
            'colors': 'k',
            'alpha': 0.3,
            'linewidths': 0.75,
            'linestyles' : '-',
        },
        'isotherms': {
            'variable_name': 'T',
            'levels': [-38, -23, 0],
            'colors': 'k',
            'alpha': 0.65,
            'linewidths': 2,
            'linestyles' : '-',
        },
        'CC': {
            'variable_name': 'CC',
            'levels': [0.8, 1],
            'colors': 'k',
            'alpha': 0.99,
            'linewidths': 0.75,
            'linestyles' : '-',
        },

        'tsubsi_thr_weak': {
            'variable_name': 'tsubsi',
            'levels': [-0.075],
            'colors': 'gold',
            'alpha': 0.99,
            'linewidths': 2.,
            'linestyles' : '-',
        },
        'tsubsi_thr_strong': {
            'variable_name': 'tsubsi',
            'levels': [-0.75],
            'colors': 'indigo',
            'alpha': 0.99,
            'linewidths': 2.,
            'linestyles' : '-',
        },

        'tmeltsi_thr_weak': {
            'variable_name': 'tmeltsi',
            'levels': [-0.075],
            'colors': 'skyblue',
            'alpha': 0.99,
            'linewidths': 2.,
            'linestyles' : '-',
        },
        'tmeltsi_thr_strong': {
            'variable_name': 'tmeltsi',
            'levels': [-0.75],
            'colors': 'navy',
            'alpha': 0.99,
            'linewidths': 2.,
            'linestyles' : '-',
        },

        'tevr_thr_weak': {
            'variable_name': 'tevr',
            'levels': [-0.05],
            'colors': 'palegreen',
            'alpha': 0.99,
            'linewidths': 2.,
            'linestyles' : '-',
        },
        'tevr_thr_strong': {
            'variable_name': 'tevr',
            'levels': [-0.5],
            'colors': 'green',
            'alpha': 0.99,
            'linewidths': 2.,
            'linestyles' : '-',
        },

    }

    #====================================================================================================================================
    ###       Properties of plotted fields:
    #----------------------------------------------
    ## Plot all plot_contourf-fields and store the returned contour plot objects
    contourf_plots = []
    for field in plot_contourf:
        if field in contourf_properties:
            
            #-----------------------------------------------------------------------     
            ## Check if field is a residual field:Fill prediciton field with 0 where it is nan, otherwise difference is also nan
            if 'residual' in field:
                if 'tsubsi' in field:
                    y_actual = cross_p['tsubsi']
                    y_pred  = (cross_p['tsubsi_pred']).fillna(0)
                    res_field = (y_actual - y_pred)

                elif 'tmeltsi' in field:
                    y_actual = cross_p['tmeltsi']
                    y_pred  = (cross_p['tmeltsi_pred']).fillna(0)
                    res_field = (y_actual - y_pred)
                elif 'tevr' in field:
                    y_actual = cross_p['tevr']
                    y_pred  = (cross_p['tevr_pred']).fillna(0)
                    res_field = (y_actual - y_pred)
                print(f'Residual field will be plotted: {field}')
                
                alpha = contourf_properties[field]['alpha']
                levels = contourf_properties[field]['levels']
                cmap = contourf_properties[field]['cmap']
                
                contourf = ax.contourf(
                    x_axis,
                    cross_p['level'],
                    res_field,
                    levels=levels,
                    cmap=cmap,
                    alpha=alpha,
                    extend=contourf_properties[field]['extend'],
                )
                contourf_plots.append(contourf)            
            #-----------------------------------------------------------------------
            # Non-residual fields
            else:
                print(f'Plot Contour {field}')
                alpha = contourf_properties[field]['alpha']
                levels = contourf_properties[field]['levels']
                cmap = contourf_properties[field]['cmap']
                
                contourf = ax.contourf(
                    x_axis,
                    cross_p['level'],
                    cross_p[field],
                    levels=levels,
                    cmap=cmap,
                    alpha=alpha,
                    extend=contourf_properties[field]['extend'],
                )
                contourf_plots.append(contourf)
        else:
            print(f'\n FIELD ({field}) DOES NOT HAVE DEFINED PROPERTIES AND WAS THEREFOR NOT PLOTTED! \n Modify "filed_properties" dictonary to add this field... \n')
    

    #### Handle ColorBars
    if contourf_colorbar:
        # Add the colorbars for each contourf plot
        colorbar_width = 0.2
        colorbar_space = 0.03
        colorbar_start_x = 0.05

        bbox = ax.get_position()
        if bbox.x0 > 0.35:
            colorbar_start_x += 3 * (colorbar_width + colorbar_space)
        cbar_count=0
        for i, (contourf_plot, field) in enumerate(zip(contourf_plots, plot_contourf)):
            if field in contourf_properties:
                # Increase cbar count by 1
                cbar_count += 1
                if cbar_count <= 3:
                    fig = ax.get_figure()
                    # Create an axis for the colorbar at the specified position
                    cbar_ax = fig.add_axes([colorbar_start_x + i * (colorbar_width + colorbar_space),   #left
                                            0.05,                                                       # bottom
                                            colorbar_width,                                             # width
                                            0.015])                                                      # height
                    # Create the colorbar using the contourf plot
                    cbar = fig.colorbar(contourf_plot, cax=cbar_ax, orientation='horizontal')
                    cbar_label = contourf_properties[field]['cbar_title']
                    cbar.set_label(cbar_label)
                else:
                    print(f'\n\nYAASS\n')
                    cbar_ax = fig.add_axes([0.05, 0.15, 0.0075, 0.4 ])                                                  
                    # Create the colorbar using the contourf plot
                    cbar = fig.colorbar(contourf_plot, cax=cbar_ax, orientation='vertical')
                    cbar_label = contourf_properties[field]['cbar_title']
                    cbar.set_label(cbar_label)
                    
    
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    
    for field in plot_contour:
        if field in contour_properties:
            print(f'Plot Contour {field}')
            levels = contour_properties[field].get('levels')
            alpha = contour_properties[field].get('alpha')
            variable_name = contour_properties[field].get('variable_name')
            linewidths = contour_properties[field].get('linewidths')
            linestyles = contour_properties[field].get('linestyles')

            if 'colors' in contour_properties[field]:
                colors = contour_properties[field].get('colors')
                contour = ax.contour(
                    #cross_p['lon'],
                    x_axis,
                    cross_p['level'],
                    cross_p[variable_name],
                    levels=levels,
                    colors=colors,
                    alpha=alpha,
                    linewidths=linewidths,
                    linestyles=linestyles,
                )
                if field == 'tsubsi_thr_weak' or field == 'tsubsi_thr_strong':
                    print('No contour')
                elif len(levels) < 4: #field == 'RH_ifs':
                    contour.clabel(contour.levels, fontsize=10, inline=1, inline_spacing=8)
                else:
                    contour.clabel(contour.levels[1::2], fontsize=10, inline=1, inline_spacing=8)
            
            elif 'cmap' in contour_properties[field]:
                contour = ax.contour(
                    #cross_p['lon'],
                    x_axis,
                    cross_p['level'],
                    cross_p[variable_name],
                    levels=contour_properties[field].get('levels'),
                    cmap=contour_properties[field].get('cmap'),
                    alpha=alpha,
                    linewidths=linewidths,
                    linestyles=linestyles,
                )
                contour.clabel(contour.levels[1::2], fontsize=10, inline=1, inline_spacing=8)
        
        else:
            print(f'\n FIELD ({field}) DOES NOT HAVE DEFINED PROPERTIES AND WAS THEREFOR NOT PLOTTED! \n Modify "filed_properties" dictopnary to add this field... \n')

    ax.set_ylim(cross_p['level'].max(), cross_p['level'].min())
    
    # Plot orientaiton dots
    midpoint_1 = [start[0] + 1/3*(end[0]-start[0]), start[1] + 1/3*(end[1]-start[1])]
    midpoint_2 = [start[0] + 2/3*(end[0]-start[0]), start[1] + 2/3*(end[1]-start[1])]
    midpoints = np.vstack([midpoint_1, midpoint_2]).transpose()[::-1]

    lowest_pres = ax.get_ylim()[0]

    if x_axis_name == 'lon':
        ax.plot(midpoints[0, 0], lowest_pres, color='k', marker='v', markersize=8,  clip_on=False, zorder=5)
        ax.plot(midpoints[0, 1], lowest_pres, color='k', marker='d', markersize=8,  clip_on=False, zorder=5)
    else:
        ax.plot(midpoints[1, 0], lowest_pres, color='k', marker='v', markersize=8,  clip_on=False)
        ax.plot(midpoints[1, 1], lowest_pres, color='k', marker='d', markersize=8,  clip_on=False)


    #====================================================================================================================================
    #====================================================================================================================================
    ## Add windbarbs, cloud and precip filed if requested
    #----------------------------------------------------
    if show_wind_barbs:
        tan_norm_wind=True
        if tan_norm_wind == False:
            cross_p['t_wind'], cross_p['n_wind'] = mpcalc.cross_section_components(cross_p['U'], cross_p['V'])
        # Customize the wind_slc_vert and wind_slc_horz variables as needed
        wind_slc_vert = list(range(0, 81, 10))
        wind_slc_horz = slice(5, 100, 10)
        ax.barbs(x_axis[wind_slc_horz], cross_p['level'][wind_slc_vert],
                 #cross_p['lon'][wind_slc_horz], cross_p['level'][wind_slc_vert],
                cross_p['t_wind'][wind_slc_vert, wind_slc_horz],
                cross_p['n_wind'][wind_slc_vert, wind_slc_horz], color='k')


    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    #- - - - - - - - - - - - - - - - - - - - - - - - Show Rain/Cloud contours - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    if show_precip:
        # Retrieve the fig object from the ax object
        fig = ax.get_figure()
        #-------------------------------------- Create colormaps --------------------------------------------#
        # Create custom colormaps
        rwc_cmap = LinearSegmentedColormap.from_list("rwc_cmap", ["#C5B358", "#574B1C"], N=8)
        swc_cmap = LinearSegmentedColormap.from_list("swc_cmap", ["lightcoral", "darkred"], N=8)
        # Create custom colormap
        rwc_lev = np.linspace(0.005, 0.3, 8)
        swc_lev = np.linspace(0.005, 0.3, 8)
        # Draw colorbars
        colorbar_ax = fig.add_axes([0.95, 0.525, 0.05, 0.15])  # Adjust these numbers for the position and size of the colorbar
        draw_color_blocks(colorbar_ax, rwc_cmap, rwc_lev, title='RWC g/kg', orientation='vertical')
        colorbar_ax = fig.add_axes([0.95, 0.725, 0.05, 0.15])  # Adjust these numbers for the position and size of the colorbar
        draw_color_blocks(colorbar_ax, swc_cmap, swc_lev, title='SWC g/kg', orientation='vertical')

        #----------------------------------------- Create Plots -----------------------------------------------#        
        rwc_contour = ax.contour(x_axis,
                                 #cross_p['lon'], 
                                 cross_p['level'], cross_p['RWC'],
                                levels=rwc_lev, cmap=rwc_cmap, linewidths=3, alpha=0.7)   
        swc_contour = ax.contour(x_axis, 
                                 #cross_p['lon'], 
                                 cross_p['level'], cross_p['SWC'],
                                levels=swc_lev, cmap=swc_cmap, linewidths=3, alpha=0.7)
        
    if show_clouds:
        # Retrieve the fig object from the ax object
        fig = ax.get_figure()
        # Create custom colormaps
        lwc_cmap = LinearSegmentedColormap.from_list("lwc_cmap", ["lightgreen", "darkgreen"], N=8)
        iwc_cmap = LinearSegmentedColormap.from_list("iwc_cmap", ["lightblue", "darkblue"], N=8)
        # Create custom colormap
        lwc_lev = np.linspace(0.005, 0.3, 8)
        iwc_lev = np.linspace(0.005, 0.3, 8)
        # Draw colorbars
        colorbar_ax = fig.add_axes([0.95, 0.125, 0.05, 0.15])  # Adjust these numbers for the position and size of the colorbar
        draw_color_blocks(colorbar_ax, lwc_cmap, lwc_lev, title='LWC g/kg', orientation='vertical')        
        colorbar_ax = fig.add_axes([0.95, 0.325, 0.05, 0.15])  # Adjust these numbers for the position and size of the colorbar
        draw_color_blocks(colorbar_ax, iwc_cmap, iwc_lev, title='IWC g/kg', orientation='vertical')

        #----------------------------------------- Create Plots -----------------------------------------------#        
        lwc_contour = ax.contour(x_axis,
                                 #cross_p['lon'], 
                                 cross_p['level'], cross_p['LWC'],
                                levels=lwc_lev, cmap=lwc_cmap, linewidths=3, alpha=0.7)
        iwc_contour = ax.contour(x_axis,
                                 #cross_p['lon'], 
                                 cross_p['level'], cross_p['IWC'],
                                levels=iwc_lev, cmap=iwc_cmap, linewidths=3, alpha=0.7)


    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    #- - - - - - - - - - - - - - - - - - - Stippling for potential bcp fields - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def plot_bcp_prediction(bcp,
                            thr_vars,
                            thr_metrics = '50%',
                            bcp_thr_value = 'thr = -0.1',
                            ):
        """
        bcp:            One of the following entries accepted: 'tsubsi, 'tmeltsi' or 'tevr'

        thr_vars:       Varaibles that are used as an criterion, can be either of: 
                        'RWC', 'SIWC', 'RH_ifs', 'T', 'V_hor', 'OMEGA'

        thr_metrics:    Statistcal value out from df_stats that will be used as threshold
                        e.g., '50%'
                            Note: Improve this to allow for more freedom in choosing what threshold should be applied
        """
        #-------------------------------------------------------------------------------------
        ## Select the dataset that should be choosen
        if bcp == 'tsubsi':
            path = '/home/freimax/msc_thesis/data/case_study_ra19/edge_field_stats/tsubsi'
            df1, df2, df3 = 'df_dict_subsi_warm', 'df_dict_subsi_mix', 'df_dict_subsi_cold'
            # Define the color for the plot
            stip_color = 'red'

        elif bcp == 'tmeltsi':
            path = '/home/freimax/msc_thesis/data/case_study_ra19/edge_field_stats/tmeltsi'
            df1, df2, df3 = 'df_dict_meltsi_warm', 'df_dict_meltsi_mix', 'df_dict_meltsi_cold'
            # Define the color for the plot
            stip_color = 'blue'

        elif bcp == 'tevr':
            path = '/home/freimax/msc_thesis/data/case_study_ra19/edge_field_stats/tevr'
            df1, df2, df3 = 'df_dict_evr_warm', 'df_dict_evr_mix', 'df_dict_evr_cold'
            # Define the color for the plot
            stip_color = 'green'
        else:
            raise ValueError('bcp process mus tbe either: tsubsi, tmeltsi, tevr')
        
        # Open the list of dataframes
        with open(f'{path}/{df1}.pkl', 'rb') as f:
            df_warm = pickle.load(f)
        with open(f'{path}/{df2}.pkl', 'rb') as f:
            df_mix = pickle.load(f)
        with open(f'{path}/{df3}.pkl', 'rb') as f:
            df_cold = pickle.load(f)
        #-------------------------------------------------------------------------------------
        # Create masks for each temperature regime
        warm_mask = cross_p.T >= 0
        mix_mask = (cross_p.T > -23) & (cross_p.T < 0)
        cold_mask = cross_p.T <= -23
        #--------------------------------------------------------------------------------------------------------------------------
        # Select variables and corresponding thresholds
        thr_metrics = thr_metrics
        
        thr_warm_RWC, thr_warm_SIWC, thr_warm_RH, thr_warm_T, thr_warm_V, thr_warm_W = df_warm[0].loc[thr_metrics, bcp_thr_value], df_warm[1].loc[thr_metrics, bcp_thr_value], df_warm[2].loc[thr_metrics, bcp_thr_value],  df_warm[3].loc[thr_metrics, bcp_thr_value], df_warm[4].loc[thr_metrics, bcp_thr_value], df_warm[5].loc[thr_metrics, bcp_thr_value]
        thr_mix_RWC , thr_mix_SIWC , thr_mix_RH, thr_mix_T  , thr_mix_V , thr_mix_W  = df_mix[0].loc[thr_metrics, bcp_thr_value] , df_mix[1].loc[thr_metrics, bcp_thr_value] , df_mix[2].loc[thr_metrics, bcp_thr_value] ,  df_mix[3].loc[thr_metrics, bcp_thr_value] , df_mix[4].loc[thr_metrics, bcp_thr_value] , df_mix[5].loc[thr_metrics, bcp_thr_value]
        thr_cold_RWC, thr_cold_SIWC, thr_cold_RH, thr_cold_T, thr_cold_V, thr_cold_W = df_cold[0].loc[thr_metrics, bcp_thr_value], df_cold[1].loc[thr_metrics, bcp_thr_value], df_cold[2].loc[thr_metrics, bcp_thr_value],  df_cold[3].loc[thr_metrics, bcp_thr_value], df_cold[4].loc[thr_metrics, bcp_thr_value], df_cold[5].loc[thr_metrics, bcp_thr_value]
        #--------------------------------------------------------------------------------------------------------------------------
        # Calculate stippling masks for each temperature regime
        warmmask, mixmask, coldmask = [], [], []
        metrics_thr = {}
        if 'RWC' in thr_vars:
            mask_rwc_warm  = cross_p['RWC'] > thr_warm_RWC
            mask_rwc_mix   = cross_p['RWC'] > thr_mix_RWC
            mask_rwc_cold  = cross_p['RWC'] > thr_cold_RWC
            warmmask.append(mask_rwc_warm)
            mixmask.append(mask_rwc_mix)
            coldmask.append(mask_rwc_cold)
            metrics_thr['RWC'] = {'w': thr_warm_RWC, 'm': thr_mix_RWC, 'c': thr_cold_RWC}

        if 'SIWC' in thr_vars:
            mask_siwc_warm = cross_p['SIWC'] > thr_warm_SIWC
            mask_siwc_mix  = cross_p['SIWC'] > thr_mix_SIWC
            mask_siwc_cold = cross_p['SIWC'] > thr_cold_SIWC
            warmmask.append(mask_siwc_warm)
            mixmask.append(mask_siwc_mix)
            coldmask.append(mask_siwc_cold)
            metrics_thr['SIWC'] = {'w': thr_warm_SIWC, 'm': thr_mix_SIWC, 'c': thr_cold_SIWC}       
        
        if 'RH_ifs' in thr_vars:
            mask_rh_warm  = cross_p['RH_ifs'] < thr_warm_RH
            mask_rh_mix   = cross_p['RH_ifs'] < thr_mix_RH
            mask_rh_cold  = cross_p['RH_ifs'] < thr_cold_RH
            warmmask.append(mask_rh_warm)
            mixmask.append(mask_rh_mix)
            coldmask.append(mask_rh_cold)   
            metrics_thr['RH_ifs'] = {'w': thr_warm_RH, 'm': thr_mix_RH, 'c': thr_cold_RH} 

        if 'T' in thr_vars:
            mask_T_warm = cross_p['T'] > thr_warm_T
            mask_T_mix  = cross_p['T'] > thr_mix_T
            mask_T_cold = cross_p['T'] > thr_cold_T
            warmmask.append(mask_T_warm)
            mixmask.append(mask_T_mix)
            coldmask.append(mask_T_cold)   
            metrics_thr['T'] = {'w': thr_warm_T, 'm': thr_mix_T, 'c': thr_cold_T}       
        
        if 'V_hor' in thr_vars:
            mask_V_warm = cross_p['V_hor'] > thr_warm_V
            mask_V_mix  = cross_p['V_hor'] > thr_mix_V
            mask_V_cold = cross_p['V_hor'] > thr_cold_V
            warmmask.append(mask_V_warm)
            mixmask.append(mask_V_mix)
            coldmask.append(mask_V_cold)   
            metrics_thr['V_hor'] = {'w': thr_warm_V, 'm': thr_mix_V, 'c': thr_cold_V}       

        if 'OMEGA' in thr_vars:
            mask_W_warm = cross_p['OMEGA'] < thr_warm_W
            mask_W_mix  = cross_p['OMEGA'] < thr_mix_W
            mask_W_cold = cross_p['OMEGA'] < thr_cold_W
            warmmask.append(mask_W_warm)
            mixmask.append(mask_W_mix)
            coldmask.append(mask_W_cold)   
            metrics_thr['OMEGA'] = {'w': thr_warm_W, 'm': thr_mix_W, 'c': thr_cold_W}       

        mask_stippling_warm = np.logical_and.reduce(warmmask) & warm_mask
        mask_stippling_mix  = np.logical_and.reduce(mixmask)  & mix_mask
        mask_stippling_cold = np.logical_and.reduce(coldmask) & cold_mask

        # Plot the stippling for SWC and RWC conditions on top of the cross-section plot
        lons, levels = np.meshgrid(x_axis, cross_p['level'])
        ax.scatter(lons[mask_stippling_warm], levels[mask_stippling_warm], c=stip_color,  marker='.', s=2)
        ax.scatter(lons[mask_stippling_mix] , levels[mask_stippling_mix] , c=stip_color,  marker='.', s=2)
        ax.scatter(lons[mask_stippling_cold], levels[mask_stippling_cold], c=stip_color,  marker='.', s=2)

        ## Return the dictonary with threshold values for plot_info
        return metrics_thr
    

    def dict_to_text(dct):
        lines = []
        for key, values in dct.items():
            line = f"{key}: "
            formatted_values = []
            for k, v in values.items():
                if v > 0.1:
                    formatted_v = f"{v:.1f}"
                elif v == 0:
                    formatted_v = f"{v:.0f}"
                else:
                    formatted_v = f"{v:.1e}"
                formatted_values.append(f"{k}={formatted_v}")
            line += ', '.join(formatted_values)
            lines.append(line)
        lines.append('-' * 50)
        return '\n'.join(lines)


    ## Add stippling
    if baseline_bcp:
        thr_vars_sub  = ['SIWC', 'RH_ifs']
        thr_vars_melt = ['SIWC', 'T', ]
        thr_vars_ev   = ['RWC', 'RH_ifs', ]        
        
        metrics_thr_sub  = plot_bcp_prediction('tsubsi' , thr_vars_sub , )
        metrics_thr_melt = plot_bcp_prediction('tmeltsi', thr_vars_melt, )
        metrics_thr_ev   = plot_bcp_prediction('tevr'   , thr_vars_ev  , )
        
        metrics_text_sub = dict_to_text(metrics_thr_sub)
        metrics_text_melt = dict_to_text(metrics_thr_melt)
        metrics_text_ev = dict_to_text(metrics_thr_ev)

        fig = ax.get_figure()
        fig.text(0.65, 0.99, metrics_text_sub,  horizontalalignment='left', verticalalignment='top', c='red',   transform=fig.transFigure)
        fig.text(0.2, 0.99, metrics_text_melt, horizontalalignment='left', verticalalignment='top', c='blue',  transform=fig.transFigure)
        fig.text(0.835, 0.99, metrics_text_ev,   horizontalalignment='left', verticalalignment='top', c='green', transform=fig.transFigure)

    
    if rf_bcp:
        def get_stippling_mask(bcp_pred):
            prediction = bcp_pred.where((bcp_pred <= 1.5) | np.isnan(bcp_pred), 2)
            prediction = prediction.where((prediction < 0.5) | (prediction > 1.5) | np.isnan(prediction), 1)
            prediction = prediction.where((prediction >= 0.5) | np.isnan(prediction), 0)
            assert (prediction ==0).sum() + (prediction ==1).sum() + (prediction ==2).sum() + np.isnan(prediction).sum() == prediction.values.flatten().shape[0]
            mask_1 = prediction == 1
            mask_2 = prediction == 2
            return mask_1, mask_2


        mask_tsubsi_1, mask_tsubsi_2   = get_stippling_mask(cross_p['y_pred_tsubsi'])
        mask_tmeltsi_1, mask_tmeltsi_2 = get_stippling_mask(cross_p['y_pred_tmeltsi'])
        mask_tevr_1, mask_tevr_2       = get_stippling_mask(cross_p['y_pred_tevr'])
        
        x, y = np.meshgrid(x_axis, cross_p['level'])
        ax.scatter(x[mask_tsubsi_1], y[mask_tsubsi_1], color='red', s=2, alpha=0.75,zorder=10)
        ax.scatter(x[mask_tsubsi_2], y[mask_tsubsi_2], color='pink', marker='s', s=4, alpha=0.85 ,zorder=10)
    
        ax.scatter(x[mask_tmeltsi_1], y[mask_tmeltsi_1], color='mediumblue', s=2, alpha=0.75,zorder=10)
        ax.scatter(x[mask_tmeltsi_2], y[mask_tmeltsi_2], color='aqua', marker='s', s=4, alpha=0.85 ,zorder=10)

        ax.scatter(x[mask_tevr_1], y[mask_tevr_1], color='forestgreen', s=2, alpha=0.75,zorder=10)
        ax.scatter(x[mask_tevr_2], y[mask_tevr_2], color='springgreen', marker='s', s=4, alpha=0.85 ,zorder=10)



    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    #====================================================================================================================================
    #====================================================================================================================================
    ###       Plot Inset Map:
    #----------------------------------------------            
    if inset_contourf:
        # Retrieve the fig object from the ax object
        fig = ax.get_figure()
        # Add the inset plot only at the specified position
        inset_crs = ccrs.PlateCarree(central_longitude=180)
        ax_inset = fig.add_axes([0.002, 0.665, 0.25, 0.25], projection=inset_crs)

        if str(inset_contourf) == 'all_bcp_sums':
            inset_contourf = ['all_bcp_sums']
            #inset_contourf = [ 'tsubs', 'tevr', 'tsubi', 'tmelts']
        
        # Colorbar axis
        cax = fig.add_axes([ax_inset.get_position().x0, ax_inset.get_position().y1 + 0.01, ax_inset.get_position().width, 0.02])
        
        ax_inset = add_inset_plot(ax_inset, inset_crs, data_p, cross_p, start, end, inset_contourf=inset_contourf, cax=cax)
        #ax_inset.set_extent([-40, 10, 20, 70], crs=inset_crs)

        
        lonmin, lonmax = data_p.lon.min() +180, data_p.lon.max()+180
        latmin, latmax = data_p.lat.min(), data_p.lat.max()
        
        print(f'\n\nlonmin: {lonmin}, \t lonmax: {lonmax} \nlatmin:{latmin}, \t latmax: {latmax}\n\n')

        ax_inset.set_extent([lonmin, lonmax, latmin, latmax], crs=inset_crs)




    ax.set_ylim(cross_p['level'].max(), cross_p['level'].min())
    # Show the plot if no ax was provided
    if ax is None:
        plt.show()
#==================================================================================================================================================================================================================
#==================================================================================================================================================================================================================





















#==================================================================================================================================================================================================================
#### Auxillary functions:
#------------------------

### Draw colorbar-blocks (separate for every color)
#### Auxillary functions:
def draw_color_blocks(ax, cmap, levels, title='', width=4.8, height=0.05, spacing=0.03, orientation='vertical', fontsize=10):
    n = len(levels)
    for i, level in enumerate(levels):
        if orientation == 'vertical':
            block = plt.Rectangle((0, i * (height + spacing)), width, height, color=cmap(i / (n - 1)))
            ax.add_artist(block)
            ax.text(width + spacing, i * (height + spacing) + height / 2, f"{level:.3f}", fontsize=fontsize,
                    verticalalignment='center', horizontalalignment='left')
        else:
            block = plt.Rectangle((i * (width + spacing), 0), width, height, color=cmap(i / (n - 1)))
            ax.add_artist(block)
            ax.text(i * (width + spacing) + width / 2, height + spacing, f"{level:.2f}", fontsize=fontsize,
                    verticalalignment='bottom', horizontalalignment='center')
    ax.set_xlim(0, n * (width + spacing))
    ax.set_ylim(0, (height + spacing) * (n if orientation == 'vertical' else 1))
    ax.axis('off')
    ax.set_title(title, fontsize=fontsize+2)



def add_inset_plot(ax_inset, inset_crs, data_p, cross_p, start, end, cax=None, inset_contourf=None):
    # Define the CRS and inset axes
    data_crs = data_p['SLP'].metpy.cartopy_crs
    
    # Plot geopotential height at 500 hPa using contour
    ax_inset.contour(data_p['lon'], data_p['lat'], data_p['SLP'], levels=np.arange(900, 1100, 5), linestyles='-', colors='k', alpha=0.4, transform=data_crs)

    contourf_properties = {
        #'tevr'   :  {'data': data_p.tevr.sum(dim='level'),   'levels': np.linspace(np.floor(data_p.tevr.sum(dim='level').min().values)/10, 0.05, 10, endpoint=False)   , 'cmap': 'YlGn_r',  },
        #'tsubs'  :  {'data': data_p.tsubs.sum(dim='level'),  'levels': np.linspace(data_p.tsubs.sum(dim='level').min().values/10, 0, 10, endpoint=False)  , 'cmap': 'inferno', },
        #'tsubi'  :  {'data': data_p.tsubi.sum(dim='level'),  'levels': np.linspace(np.floor(data_p.tsubi.sum(dim='level').min().values)/10, 0.05, 10, endpoint=False)  , 'cmap': 'YlGnBu_r',},
        #'tmelts' :  {'data': data_p.tmelts.sum(dim='level'), 'levels': np.linspace(np.floor(data_p.tmelts.sum(dim='level').min().values)/10, 0.05, 10, endpoint=False) , 'cmap': 'RdPu_r' , },
        #'tsubs'  :  {'data': data_p.tsubs.sum(dim='level'),  'levels': np.linspace(np.floor(data_p.tsubs.sum(dim='level').min().values)/10, 0.0, 10, endpoint=False)  , 'cmap': 'inferno', },
        'all_bcp_sums' : {'data': (data_p.tevr + data_p.tsubsi + data_p.tmeltsi).sum(dim='level'), 
                          'levels': np.linspace(-10,0, 5, endpoint=False), 
                          'cmap' : 'Blues_r'}
        }
    # Plot tbcp field using contourf
    if inset_contourf is not None:
        for field in inset_contourf:
            data = contourf_properties[field]['data']
            cmap = contourf_properties[field]['cmap']
            levs = contourf_properties[field]['levels']
            #levs = np.linspace(-1.5,1.50,10)
            img = data.plot(ax=ax_inset, add_colorbar=False, add_labels=False, transform=data_crs, alpha=0.8, cmap=cmap, levels=levs)
        #inset_contourf.plot(ax=ax_inset, add_colorbar=False, add_labels=False, transform=data_crs, levels=np.linspace(-3.5,3.5)) 



        # Create colorbar
        cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
        # Move ticks and labels above the colorbar
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        # Add colorbar label above the ticks
        cbar.ax.set_title('Vertical integrated BCP [K/hr]', fontsize=10, pad=5)


    # Plot the path of the cross section
    endpoints = inset_crs.transform_points(ccrs.Geodetic(), *np.vstack([start, end]).transpose()[::-1])
    ax_inset.scatter(endpoints[:, 0], endpoints[:, 1], c='k', zorder=2)
    ax_inset.plot(cross_p['lon'], cross_p['lat'], c='k', zorder=2, transform=ccrs.Geodetic())

    # Calculate the indices corresponding to the 1/3 and 2/3 distances along the cross-section
    num_points = len(cross_p['lon'])
    one_third_index = int(num_points * 1/3)
    two_thirds_index = int(num_points * 2/3)
    # Extract the latitudes and longitudes at the 1/3 and 2/3 indices
    one_third_lat, one_third_lon = cross_p['lat'][one_third_index].item(), cross_p['lon'][one_third_index].item()
    two_thirds_lat, two_thirds_lon = cross_p['lat'][two_thirds_index].item(), cross_p['lon'][two_thirds_index].item()

    # Plot the black dots on the inset map
    ax_inset.plot(one_third_lon, one_third_lat, c='k', marker='v', markersize=5, transform=ccrs.PlateCarree(), zorder=3)
    ax_inset.plot(two_thirds_lon, two_thirds_lat, c='k', marker='d', markersize=5, transform=ccrs.PlateCarree(), zorder=3)

    # Add geographic features
    ax_inset.coastlines()
    ax_inset.gridlines(ylocs=np.arange(30, 91, 15), xlocs=np.arange(-180, 181, 15), alpha=0.5)
    ax_inset.set_xticks(np.arange(-180, 181, 15), crs=ccrs.PlateCarree())
    ax_inset.set_yticks(np.arange(30, 91, 15), crs=ccrs.PlateCarree())
    ax_inset.xaxis.set_major_formatter(LongitudeFormatter())
    ax_inset.yaxis.set_major_formatter(LatitudeFormatter())

    return ax_inset








