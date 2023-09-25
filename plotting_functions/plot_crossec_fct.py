## Import modules
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import metpy.calc as mpcalc

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
                       flag_1cb=False
                       ):
    
    ## Define what happens if fiels is not defined
    if ax is None:
        fig, ax = plt.subplots()
    if plot_contourf is None:
        plot_contourf = []
    if plot_contour is None:
        plot_contour = []
    
    ## Keep track of fields and colorbars that are already plotted
    fig = ax.get_figure()
    if not hasattr(fig, 'plotted_fields'):          # Keeps track of plotted fields and colorbars, sucht that they are not plotted multiple times
        fig.plotted_fields = []
    
    #---------------------------------------------------------------
    # Determine if cross-section x-axis is shown in lon or lat
    lon_difference = abs(max(cross_p['lon']) - min(cross_p['lon']))
    lat_difference = abs(max(cross_p['lat']) - min(cross_p['lat']))
    
    x_axis_name = 'lon'
    if lon_difference >= lat_difference:
        x_axis = cross_p['lon']
    else:
        #ax.invert_xaxis()
        x_axis = cross_p['lat'] # safed with 2 in the end
        x_axis_name = 'lat'
    #---------------------------------------------------------------        

    #====================================================================================================================================
    ###       Properties of plotted fields:
    #----------------------------------------------

    #########################################################################################################################
    #########                                        MAKE CUSTOM COLORMAPS                                          #########
    #-----------------------------------------------------------------------------------------------------------------------#
    ## COLORBAR FOR RH                                                                                                      #
    plasma_r = plt.cm.get_cmap('plasma_r', 100)                                                                             #
    skyblue = plt.cm.get_cmap('Blues_r', 40)                                                                                #
    #                                                                                                                       #
    colors0 = np.linspace([1, 1, 1, 1], [1, 1, 1, 1], 10)                                                                   #
    white_to_yellow = np.linspace([1, 1, 1, 1], plasma_r(0), 30)                                                            #
    colors1 = white_to_yellow                                                                                               #
    colors2 = plasma_r(np.linspace(0, 1, 60))                                                                               #
    colors3 = skyblue(np.linspace(0.1, 1, 40))                                                                              #
    #                                                                                                                       #
    colors = np.vstack((colors0, colors1, colors2, colors3))                                                                #
    rh_cust_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)                                                 #
    #-----------------------------------------------------------------------------------------------------------------------#                                                     
    ## COLORBAR FOR RESIDUAL FIELDS                                                                                         #
    #                                                                                                                       #
    # Define the colors for our new colormap                                                                                #
    cmap_seismic = plt.cm.get_cmap('seismic')                                                                               #
    # Define the colors for our new colormap                                                                                #
    colors = [(cmap_seismic(i/256)) for i in range(128)] + [(1,1,1,1)]*50 + [(cmap_seismic(i/256)) for i in range(128, 256)]#
    tsubsi_cmap = mcolors.LinearSegmentedColormap.from_list("new_PRGn", colors)                                             #
    #                                                                                                                       #
    # Define the RdYlGn colormap                                                                                            #
    cmap_RdYlGn = plt.cm.get_cmap('RdYlGn')                                                                                 #
    # Define the colors for our new colormap                                                                                #
    colors = [(cmap_RdYlGn(i/256)) for i in range(128)] + [(1,1,1,1)]*50 + [(cmap_RdYlGn(i/256)) for i in range(128, 256)]  #
    tevr_cmap = mcolors.LinearSegmentedColormap.from_list("new_RdYlGn", colors)                                             #
    #                                                                                                                       #
    # Define the RdYlGn colormap                                                                                            #
    cmap_PuOr = plt.cm.get_cmap('PuOr')                                                                                     #
    # Define the colors for our new colormap                                                                                #
    colors = [(cmap_PuOr(i/256)) for i in range(128)] + [(1,1,1,1)]*50 + [(cmap_PuOr(i/256)) for i in range(128, 256)]      #
    tmeltsi_cmap = mcolors.LinearSegmentedColormap.from_list("new_PuOr", colors)                                            #
    #########################################################################################################################

    ## Define properties contourf plots 
    contourf_properties = {
        'tevr'        :  {'levels': np.arange(-0.5, -0.049, 0.025) , 'cmap': 'YlGn_r'    , 'alpha': 0.75,  'cbar_title': 'T evR [K/h]'             , 'extend':'min'    , 'cbar_id':'cbar_tevr'   },
        'tsubsi'      :  {'levels': np.arange(-1.5, -0.074, 0.025) , 'cmap': 'inferno'   , 'alpha': 0.75,  'cbar_title': 'T sub S&I [K/h]'         , 'extend':'min'    , 'cbar_id':'cbar_tsubsi' },
        'tmeltsi'     :  {'levels': np.arange(-1.5, -0.074, 0.025) , 'cmap': 'Blues_r'   , 'alpha': 0.75,  'cbar_title': 'T melt S&I [K/h]'        , 'extend':'min'    , 'cbar_id':'cbar_tmeltsi'},
        'RH'          :  {'levels': np.linspace(0,140,15)          , 'cmap': rh_cust_cmap, 'alpha': 0.3 ,  'cbar_title': 'RH in %'                  , 'extend':'neither', 'cbar_id':'cbar_RH'     },
        'OMEGA'       :  {'levels': np.linspace(-1,1,10)           , 'cmap':'PuOr'       , 'alpha': 0.2 ,  'cbar_title': 'Omega [Pa/s]'             , 'extend':'both'   , 'cbar_id':'cbar_omega'  },
 
        'tsubsi_pred' :  {'levels': np.arange(-1.5, -0.074, 0.025) , 'cmap': 'inferno'   , 'alpha': 0.75,  'cbar_title': r'Pred Q$_{subsi}$ [K/h]' , 'extend':'min'    , 'cbar_id':'cbar_tsubsi' },
        'tmeltsi_pred':  {'levels': np.arange(-1.5, -0.074, 0.025) , 'cmap': 'Blues_r'   , 'alpha': 0.75,  'cbar_title': r'Pred Q$_{meltsi}$ [K/h]', 'extend':'min'    , 'cbar_id':'cbar_tmeltsi'},
        'tevr_pred'   :  {'levels': np.arange(-0.5, -0.049, 0.025) , 'cmap': 'YlGn_r'    , 'alpha': 0.75,  'cbar_title': r'Pred Q$_{evr}$ [K/h]'   , 'extend':'min'    , 'cbar_id':'cbar_tevr'   },
         
        'residual_tsubsi' :  {'levels': np.arange(-0.5, 0.51, 0.1) , 'cmap': tsubsi_cmap , 'alpha': 0.75 , 'cbar_title': r'Res Q$_{subsi}$ '        , 'extend':'both'   , 'cbar_id':'cbar_res_tsubsi' },
        'residual_tmeltsi':  {'levels': np.arange(-0.5, 0.51, 0.1) , 'cmap': tmeltsi_cmap, 'alpha': 0.75 , 'cbar_title': r'Res Q$_{meltsi}$ '       , 'extend':'both'   , 'cbar_id':'cbar_res_tmeltsi'},
        'residual_tevr'   :  {'levels': np.arange(-0.5, 0.51, 0.1) , 'cmap': tevr_cmap   , 'alpha': 0.75 , 'cbar_title': r'Res Q$_{evr}$ '          , 'extend':'both'   , 'cbar_id':'cbar_res_tevr'   },
    }

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    

    ## Define properties of contour plots 
    contour_properties = {
        'RH'        : { 'variable_name': 'RH'      ,  'levels': [80,100]                 ,  'colors': 'blue'  ,  'alpha': 0.75  ,  'linewidths': 1     ,  'linestyles':'-', },
        'TH'        : { 'variable_name': 'TH'      ,  'levels': np.arange(270, 365, 10)  ,  'colors': 'k'     ,  'alpha': 0.3   ,  'linewidths': 0.75  ,  'linestyles':'-', },
        'isotherms' : { 'variable_name': 'T'       ,  'levels': [-38, -23, 0]            ,  'colors': 'k'     ,  'alpha': 0.65  ,  'linewidths': 2     ,  'linestyles':'-', },
        'CC'        : { 'variable_name': 'CC'      ,  'levels': [0.8, 1]                 ,  'colors': 'k'     ,  'alpha': 0.99  ,  'linewidths': 0.75  ,  'linestyles':'-', },
  
        'tsubsi_thr_weak'    : {  'variable_name': 'tsubsi'   ,  'levels': [-0.075]  ,  'colors': 'gold'       ,  'alpha': 0.75  ,  'linewidths': 2.  ,  'linestyles':'--'  , },
        'tsubsi_thr_strong'  : {  'variable_name': 'tsubsi'   ,  'levels': [-0.75]   ,  'colors': 'indigo'     ,  'alpha': 0.75  ,  'linewidths': 2.  ,  'linestyles':'--'  , },
        'tmeltsi_thr_weak'   : {  'variable_name': 'tmeltsi'  ,  'levels': [-0.075]  ,  'colors': 'skyblue'    ,  'alpha': 0.75  ,  'linewidths': 2.  ,  'linestyles':'--'  , },
        'tmeltsi_thr_strong' : {  'variable_name': 'tmeltsi'  ,  'levels': [-0.75]   ,  'colors': 'navy'       ,  'alpha': 0.75  ,  'linewidths': 2.  ,  'linestyles':'--'  , },
        'tevr_thr_weak'      : {  'variable_name': 'tevr'     ,  'levels': [-0.05]   ,  'colors': 'palegreen'  ,  'alpha': 0.75  ,  'linewidths': 2.  ,  'linestyles':'--'  , },
        'tevr_thr_strong'    : {  'variable_name': 'tevr'     ,  'levels': [-0.5]    ,  'colors': 'green'      ,  'alpha': 0.75  ,  'linewidths': 2.  ,  'linestyles':'--'  , },
    }  


    #====================================================================================================================================
    ###       Properties of plotted fields:
    #----------------------------------------------
    ## Plot all plot_contourf-fields and store the returned contour plot objects
    contourf_plots = []
    for field in plot_contourf:
        if field in contourf_properties:
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
        colorbar_width = 0.15
        colorbar_space = 0.005
        colorbar_start_x = 0.01

        bbox = ax.get_position()
        if bbox.x0 > 0.35:
            print(f'Update cbar start position because bbox = {bbox}')
            # colorbar_start_x += 3 * (colorbar_width + colorbar_space)
            colorbar_start_x += 2 * (colorbar_width + colorbar_space)
        cbar_count=0

        if flag_1cb:
            colorbar_width = 0.2
            colorbar_space = 0.005
            colorbar_start_x = 0.4
     

        for i, (contourf_plot, field) in enumerate(zip(contourf_plots, plot_contourf)):
            if field in contourf_properties:
                # Increase cbar count
                cbar_count += 1
                cbar_ID = contourf_properties[field]['cbar_id']     ## Extract the cmap ID of the field
                print(f'Plot colorbar with iD: {cbar_ID}')
                
                if cbar_ID not in fig.plotted_fields:               # Check if the colorbar has been plotted
                    fig.plotted_fields.append(cbar_ID)              # Add colorbar to the list of already plotted

                    if cbar_count <= 3:
                        # fig = ax.get_figure()
                        # Create an axis for the colorbar at the specified position
                        cbar_ax = fig.add_axes([colorbar_start_x + i * (colorbar_width + colorbar_space),   #left
                                                0.05,                                                       # bottom
                                                colorbar_width,                                             # width
                                                0.025])                                                     # height
                        # Create the colorbar using the contourf plot
                        cbar = fig.colorbar(contourf_plot, cax=cbar_ax, orientation='horizontal')
                        
                        ## Adjust ticks of cbar
                        cbar_ticks = np.round(np.linspace(cbar.vmin, cbar.vmax, 5)*10)/10
                        cbar_ticks[0], cbar_ticks[-1] = np.round(cbar.vmin*1000)/1000, np.round(cbar.vmax*1000)/1000
                        cbar.set_ticks(cbar_ticks)                        
                        
                        cbar_label = contourf_properties[field]['cbar_title']
                        cbar.set_label(cbar_label)
                        cbar.ax.set_xticklabels(cbar.get_ticks(), rotation=0)
                    
                    else:
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
                    pass
                elif len(levels) < 4: #field == 'RH_ifs':
                    contour.clabel(contour.levels, fontsize=10, inline=1, inline_spacing=8)
                else:
                    contour.clabel(contour.levels[1::2], fontsize=10, inline=1, inline_spacing=8)
            
            elif 'cmap' in contour_properties[field]:
                contour = ax.contour(
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
    ## Check whether argument is a list, 
    #       If it is a list, first element specifies if contour should be shown at all, second if cbar should be plotted
    if isinstance(show_precip, list):
        show_precip_list = show_precip
        show_precip = show_precip_list[0]
        show_precip_cbar = show_precip_list[1]

    if show_precip:
        # Retrieve the fig object from the ax object
        fig = ax.get_figure()
        #-------------------------------------- Create colormaps --------------------------------------------#
        # Create custom colormaps
        rwc_cmap = LinearSegmentedColormap.from_list("rwc_cmap", ["#C5B358", "#574B1C"], N=8)
        swc_cmap = LinearSegmentedColormap.from_list("swc_cmap", ["lightcoral", "darkred"], N=8)
        # Create custom colormap
        rwc_lev = np.linspace(0.005, 0.6, 8)
        swc_lev = np.linspace(0.005, 0.6, 8)
        # Draw colorbars
        if show_precip_cbar:
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
    
    if isinstance(show_clouds, list):
        show_clouds_list = show_clouds
        show_clouds = show_clouds_list[0]
        show_clouds_cbar = show_clouds_list[1]
    
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
        if show_clouds_cbar:
            colorbar_ax = fig.add_axes([0.95, 0.125, 0.05, 0.15])  # Adjust these numbers for the position and size of the colorbar
            draw_color_blocks(colorbar_ax, lwc_cmap, lwc_lev, title='LWC g/kg', orientation='vertical')        
            colorbar_ax = fig.add_axes([0.95, 0.325, 0.05, 0.15])  # Adjust these numbers for the position and size of the colorbar
            draw_color_blocks(colorbar_ax, iwc_cmap, iwc_lev, title='IWC g/kg', orientation='vertical')

        #----------------------------------------- Create Plots -----------------------------------------------#        
        lwc_contour = ax.contour(x_axis,
                                 cross_p['level'], cross_p['LWC'],
                                levels=lwc_lev, cmap=lwc_cmap, linewidths=3, alpha=0.7)
        iwc_contour = ax.contour(x_axis,
                                 cross_p['level'], cross_p['IWC'],
                                levels=iwc_lev, cmap=iwc_cmap, linewidths=3, alpha=0.7)


    



    #==============================================================================================================================================================
    ####                 RF PREDCITONS SETTING HERE
    #----------------------------------------------
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
        # ax_inset = fig.add_axes([0.0, 0.75, 0.05, 0.05], projection=inset_crs)
        ax_inset = fig.add_axes([0.0, 0.75, 0.175, 0.175], projection=inset_crs)

        if str(inset_contourf) == 'all_bcp_sums':
            inset_contourf = ['all_bcp_sums']
        elif str(inset_contourf) == 'max_cc':
            inset_contourf = ['max_cc']
        
        # Colorbar axis
        cax = fig.add_axes([ax_inset.get_position().x0, ax_inset.get_position().y1 + 0.01, ax_inset.get_position().width, 0.02])
        
        ax_inset = add_inset_plot(ax_inset, inset_crs, data_p, cross_p, start, end, inset_contourf=inset_contourf, cax=cax)
        #ax_inset.set_extent([-40, 10, 20, 70], crs=inset_crs)


        lonmin, lonmax = data_p.lon.min() +180 +23, data_p.lon.max()+180 -22
        latmin, latmax = data_p.lat.min() +13, data_p.lat.max()-3
        # print(f'\nPlot figure with lonmin: {lonmin}, \t lonmax: {lonmax} \nlatmin:{latmin}, \t latmax: {latmax}\n')
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
# def draw_color_blocks(ax, cmap, levels, title='', width=4.8, height=0.05, spacing=0.03, orientation='vertical', fontsize=10):
# def draw_color_blocks(ax, cmap, levels, title='', width=5, height=0.075, spacing=0.03, orientation='vertical', fontsize=10):
def draw_color_blocks(ax, cmap, levels, title='', width=10, height=0.5, spacing=0.1, orientation='vertical', fontsize=10):

    n = len(levels)
    for i, level in enumerate(levels):
        if orientation == 'vertical':
            block = plt.Rectangle((0, i * (height + spacing)), width, height, color=cmap(i / (n - 1)))
            ax.add_artist(block)
            ax.text(width + spacing, i * (height + spacing) + height / 2, f" {level:.3f}", fontsize=fontsize,
                    verticalalignment='center', horizontalalignment='left')
        
        else:
            block = plt.Rectangle((i * (width + spacing), 0), width, height, color=cmap(i / (n - 1)))
            ax.add_artist(block)
            ax.text(i * (width + spacing) + width / 2, height + spacing, f" {level:.2f}", fontsize=fontsize,
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
        'all_bcp_sums' : {'data': (data_p.tevr + data_p.tsubsi + data_p.tmeltsi).sum(dim='level'), 'levels': np.linspace(-10,0, 5, endpoint=False), 'cmap' : 'Blues_r' , 'cbar_title' : 'Vertical integrated BCP [K/h]'},
        'max_cc'       : {'data': (data_p.CC).max(dim='level')                                   , 'levels': None                                 , 'cmap' : 'afmhot_r', 'cbar_title' : 'Column max. Cloud Cover'}
        }
    # Plot tbcp field using contourf
    if inset_contourf is not None:
        for field in inset_contourf:
            data = contourf_properties[field]['data']
            cmap = contourf_properties[field]['cmap']
            levs = contourf_properties[field]['levels']
            cbar_title = contourf_properties[field]['cbar_title']
            img = data.plot(ax=ax_inset, add_colorbar=False, add_labels=False, transform=data_crs, alpha=0.8, cmap=cmap, levels=levs)

            # Create colorbar
            cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
            # Move ticks and labels above the colorbar
            cbar.ax.xaxis.set_ticks_position('top')
            cbar.ax.xaxis.set_label_position('top')
            # Add colorbar label above the ticks
            cbar.ax.set_title(cbar_title, fontsize=10, pad=5)



    color_line = 'slategrey'
    # Plot the path of the cross section
    endpoints = inset_crs.transform_points(ccrs.Geodetic(), *np.vstack([start, end]).transpose()[::-1])
    ax_inset.scatter(endpoints[:, 0], endpoints[:, 1], c=color_line, zorder=2)
    ax_inset.plot(cross_p['lon'], cross_p['lat'], c=color_line, zorder=2, transform=ccrs.Geodetic())

    # Calculate the indices corresponding to the 1/3 and 2/3 distances along the cross-section
    num_points = len(cross_p['lon'])
    one_third_index = int(num_points * 1/3)
    two_thirds_index = int(num_points * 2/3)
    # Extract the latitudes and longitudes at the 1/3 and 2/3 indices
    one_third_lat, one_third_lon = cross_p['lat'][one_third_index].item(), cross_p['lon'][one_third_index].item()
    two_thirds_lat, two_thirds_lon = cross_p['lat'][two_thirds_index].item(), cross_p['lon'][two_thirds_index].item()
    


    # Plot the black dots on the inset map
    ax_inset.plot(one_third_lon, one_third_lat, c=color_line, marker='v', markersize=5, transform=ccrs.PlateCarree(), zorder=3)
    ax_inset.plot(two_thirds_lon, two_thirds_lat, c=color_line, marker='d', markersize=5, transform=ccrs.PlateCarree(), zorder=3)

    # Add geographic features
    ax_inset.coastlines()
    ax_inset.gridlines(ylocs=np.arange(30, 91, 15), xlocs=np.arange(-180, 181, 15), alpha=0.5)
    ax_inset.set_xticks(np.arange(-180, 181, 15), crs=ccrs.PlateCarree())
    ax_inset.set_yticks(np.arange(30, 91, 15), crs=ccrs.PlateCarree())
    ax_inset.xaxis.set_major_formatter(LongitudeFormatter())
    ax_inset.yaxis.set_major_formatter(LatitudeFormatter())

    return ax_inset








