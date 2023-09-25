## Import modules
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.colors as mcolors

#==================================================================================================================================================================================================================
#==================================================================================================================================================================================================================
#==================================================================================================================================================================================================================

def plot_crossec_rf_eval(cross_p, 
                         data_p, 
                         start, 
                         end,
                         plot_contourf,
                         plot_contour,
                         ax=None,
                         contourf_colorbar=True,
                         inset_contourf='all_bcp_sums',
                         ):
    
    name_vertical_coord = 'level'


    ## Keep track of fields and colorbars that are already plotted
    fig = ax.get_figure()
    if not hasattr(fig, 'plotted_fields'):          # Keeps track of plotted fields and colorbars, sucht that they are not plotted multiple times
        fig.plotted_fields = []
        
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
        x_axis = cross_p['lat'] # safed with 2 in the end
        x_axis_name = 'lat'
    #---------------------------------------------------------------        

    #====================================================================================================================================
    ###       Properties of plotted fields:
    #--------------------------------------
    
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
    colors = [(cmap_seismic(i/256)) for i in range(128)] + [(1,1,1,1)]*25 + [(cmap_seismic(i/256)) for i in range(128, 256)]#
    tsubsi_cmap = mcolors.LinearSegmentedColormap.from_list("new_PRGn", colors)                                             #
    #                                                                                                                       #
    # Define the RdYlGn colormap                                                                                            #
    cmap_RdYlGn = plt.cm.get_cmap('RdYlGn')                                                                                 #
    # Define the colors for our new colormap                                                                                #
    colors = [(cmap_RdYlGn(i/256)) for i in range(128)] + [(1,1,1,1)]*25 + [(cmap_RdYlGn(i/256)) for i in range(128, 256)]  #
    tevr_cmap = mcolors.LinearSegmentedColormap.from_list("new_RdYlGn", colors)                                             #
    #                                                                                                                       #
    # Define the RdYlGn colormap                                                                                            #
    cmap_PuOr = plt.cm.get_cmap('PuOr')                                                                                     #
    # Define the colors for our new colormap                                                                                #
    colors = [(cmap_PuOr(i/256)) for i in range(128)] + [(1,1,1,1)]*25 + [(cmap_PuOr(i/256)) for i in range(128, 256)]      #
    tmeltsi_cmap = mcolors.LinearSegmentedColormap.from_list("new_PuOr", colors)                                            #
    #########################################################################################################################



    ## Define properties contourf plots 
    contourf_properties = {
        'tsubsi'      :  {'levels': np.arange(-1.5, -0.049, 0.025) , 'cmap': 'inferno'   , 'alpha': 0.75,  'cbar_title': r'Q$_{subsi}$ [K/h]'      , 'extend':'min'    , 'cbar_id':'cbar_tsubsi' },
        'tmeltsi'     :  {'levels': np.arange(-1.0, -0.049, 0.025) , 'cmap': 'Blues_r'   , 'alpha': 0.75,  'cbar_title': r'Q$_{meltsi}$ [K/h]'     , 'extend':'min'    , 'cbar_id':'cbar_tmeltsi'},
        'tevr'        :  {'levels': np.arange(-0.5, -0.049, 0.01)  , 'cmap': 'YlGn_r'    , 'alpha': 0.75,  'cbar_title': r'Q$_{evr}$ [K/h]'        , 'extend':'min'    , 'cbar_id':'cbar_tevr'   },
        'RH'          :  {'levels': np.linspace(0,140,15)          , 'cmap': rh_cust_cmap, 'alpha': 0.3 ,  'cbar_title': 'RH in %'                  , 'extend':'neither', 'cbar_id':'cbar_RH'     },
        'OMEGA'       :  {'levels': np.linspace(-1,1,10)           , 'cmap':'PuOr'       , 'alpha': 0.2 ,  'cbar_title': 'Omega [Pa/s]'             , 'extend':'both'   , 'cbar_id':'cbar_omega'  },
 
        'tsubsi_pred' :  {'levels': np.arange(-1.5, -0.074, 0.025) , 'cmap': 'inferno'   , 'alpha': 0.75,  'cbar_title': r'Pred Q$_{subsi}$ [K/h]' , 'extend':'min'    , 'cbar_id':'cbar_tsubsi' },
        'tmeltsi_pred':  {'levels': np.arange(-1.5, -0.074, 0.025) , 'cmap': 'Blues_r'   , 'alpha': 0.75,  'cbar_title': r'Pred Q$_{meltsi}$ [K/h]', 'extend':'min'    , 'cbar_id':'cbar_tmeltsi'},
        'tevr_pred'   :  {'levels': np.arange(-0.5, -0.049, 0.025) , 'cmap': 'YlGn_r'    , 'alpha': 0.75,  'cbar_title': r'Pred Q$_{evr}$ [K/h]'   , 'extend':'min'    , 'cbar_id':'cbar_tevr'   },
         
        'residual_tsubsi' : {'levels': np.arange(-0.5, 0.505, 0.05) , 'cmap': tsubsi_cmap , 'alpha': 0.65   , 'cbar_title': r'Res Q$_{subsi}$ [K/h]'  , 'extend':'both'   , 'cbar_id':'cbar_res_tsubsi' },
        'residual_tmeltsi': {'levels': np.arange(-0.5, 0.505, 0.05) , 'cmap': tmeltsi_cmap, 'alpha': 0.65   , 'cbar_title': r'Res Q$_{meltsi}$ [K/h]' , 'extend':'both'   , 'cbar_id':'cbar_res_tmeltsi'},
        'residual_tevr'   : {'levels': np.arange(-0.25, 0.251, 0.025) , 'cmap': tevr_cmap   , 'alpha': 0.65 , 'cbar_title': r'Res Q$_{evr}$ [K/h]'    , 'extend':'both'   , 'cbar_id':'cbar_res_tevr'   },
    }

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    ## Define properties of contour plots 
    contour_properties = {
        'RH'        : { 'variable_name': 'RH'      ,  'levels': [80,100]                 ,  'colors': 'blue'  ,  'alpha': 0.75  ,  'linewidths': 1     ,  'linestyles':'-', },
        'TH'        : { 'variable_name': 'TH'      ,  'levels': np.arange(270, 365, 10)  ,  'colors': 'k'     ,  'alpha': 0.3   ,  'linewidths': 0.75  ,  'linestyles':'-', },
        'isotherms' : { 'variable_name': 'T'       ,  'levels': [-38, -23, 0]            ,  'colors': 'k'     ,  'alpha': 0.65  ,  'linewidths': 2     ,  'linestyles':'-', },
        'CC'        : { 'variable_name': 'CC'      ,  'levels': [0.8, 1]                 ,  'colors': 'k'     ,  'alpha': 0.99  ,  'linewidths': 0.75  ,  'linestyles':'-', },
  
        'tsubsi_thr_weak'    : {  'variable_name': 'tsubsi'   ,  'levels': [-0.05]  ,  'colors': 'gold'       ,  'alpha': 0.75  ,  'linewidths': 2.  ,  'linestyles':'--'  , },
        'tmeltsi_thr_weak'   : {  'variable_name': 'tmeltsi'  ,  'levels': [-0.05]  ,  'colors': 'skyblue'    ,  'alpha': 0.75  ,  'linewidths': 2.  ,  'linestyles':'--'  , },
        'tevr_thr_weak'      : {  'variable_name': 'tevr'     ,  'levels': [-0.01]  ,  'colors': 'palegreen'  ,  'alpha': 0.75  ,  'linewidths': 2.  ,  'linestyles':'--'  , },
        'tsubsi_thr_strong'  : {  'variable_name': 'tsubsi'   ,  'levels': [-0.75]  ,  'colors': 'indigo'     ,  'alpha': 0.75  ,  'linewidths': 2.  ,  'linestyles':'--'  , },
        'tmeltsi_thr_strong' : {  'variable_name': 'tmeltsi'  ,  'levels': [-0.75]  ,  'colors': 'navy'       ,  'alpha': 0.75  ,  'linewidths': 2.  ,  'linestyles':'--'  , },
        'tevr_thr_strong'    : {  'variable_name': 'tevr'     ,  'levels': [-0.03]  ,  'colors': 'green'      ,  'alpha': 0.75  ,  'linewidths': 2.  ,  'linestyles':'--'  , },
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
                ## Build in error-tolerance, such that contourf are not overlapping (covering each other)
                error_tolerance = 0.05
                res_field = res_field.where(np.abs(res_field) >= error_tolerance, np.nan)
                
                alpha = contourf_properties[field]['alpha']
                levels = contourf_properties[field]['levels']
                cmap = contourf_properties[field]['cmap']
                
                contourf = ax.contourf(
                    x_axis,
                    cross_p[name_vertical_coord],
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
                alpha = contourf_properties[field]['alpha']
                levels = contourf_properties[field]['levels']
                cmap = contourf_properties[field]['cmap']
                
                contourf = ax.contourf(
                    x_axis,
                    cross_p[name_vertical_coord],
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
        
        # Set position and size of colorbars
        colorbar_start_x = 0.12
        colorbar_start_y = 0.06
        colorbar_width   = 0.10
        colorbar_height  = 0.015
        colorbar_space = 0.033
        ## This condition increases the statlocation if we consider the second ax-subplot
        position_subplot = ax.get_position()
        if position_subplot.x0 > 0.25:
            nr_cbar_shift = 3 # nr_colorbars_previous_subplot
            colorbar_start_x += nr_cbar_shift * (colorbar_width + colorbar_space)
        

        cbar_count=0
        for i, (contourf_plot, field) in enumerate(zip(contourf_plots, plot_contourf)):
            if field in contourf_properties:
                # Increase cbar count by 1
                cbar_count += 1
                
                cbar_ID = contourf_properties[field]['cbar_id']  ## Extract the cmap ID of the field
                if cbar_ID not in fig.plotted_fields:  # Check if the colorbar has been plotted
                    fig.plotted_fields.append(cbar_ID)  # Add colorbar to the list of already plotted
                    if cbar_count <= 3:                 # Check it his colorbar still fits below the images  <--> only checks for same subplot
                        #fig = ax.get_figure()
                        # Create an axis for the colorbar at the specified position
                        cbar_ax = fig.add_axes([colorbar_start_x + i * (colorbar_width + colorbar_space),   # left
                                                colorbar_start_y,                                                       # bottom
                                                colorbar_width,                                             # width
                                                colorbar_height])                                                     # height
                        # Create the colorbar using the contourf plot
                        # ticks = contourf_properties[field]['levels']
                        # cbar_ticks = np.round(np.linspace(ticks.min(), ticks.max(),4,2)
                        cbar = fig.colorbar(contourf_plot, cax=cbar_ax, 
                                            orientation='horizontal') #ticks=cbar_ticks, 
                        
                        ## Adjust ticks of cbar
                        cbar_ticks = np.round(np.linspace(cbar.vmin, cbar.vmax, 4)*10)/10
                        cbar_ticks[0], cbar_ticks[-1] = np.round(cbar.vmin*1000)/1000, np.round(cbar.vmax*1000)/1000
                        cbar.set_ticks(cbar_ticks)                        
                        cbar_label = contourf_properties[field]['cbar_title']
                        cbar.set_label(cbar_label, fontsize=12)
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
                elif len(levels) < 4: #field == 'RH':
                    contour.clabel(contour.levels, fontsize=10, inline=1, inline_spacing=8)
                else:
                    contour.clabel(contour.levels[1::2], fontsize=10, inline=1, inline_spacing=8)
            
            elif 'cmap' in contour_properties[field]:
                contour = ax.contour(
                    x_axis,
                    cross_p[name_vertical_coord],
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

    ax.set_ylim(cross_p[name_vertical_coord].max(), cross_p[name_vertical_coord].min())
    
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
        ax_inset = fig.add_axes([0.0, 0.75, 0.175, 0.175], projection=inset_crs)  #xstart, ystart, long, height

        if str(inset_contourf) == 'all_bcp_sums':
            inset_contourf = ['all_bcp_sums']
            #inset_contourf = [ 'tsubs', 'tevr', 'tsubi', 'tmelts']
        
        # Colorbar axis
        cax = fig.add_axes([ax_inset.get_position().x0,             # Left
                            ax_inset.get_position().y1 + 0.005,     # Bottom
                            ax_inset.get_position().width,          # Width 
                            .01])                                   # Hiehgt
        
        ax_inset = add_inset_plot(ax_inset, inset_crs, data_p, cross_p, start, end, inset_contourf=inset_contourf, cax=cax)
        #ax_inset.set_extent([-40, 10, 20, 70], crs=inset_crs)

        
        lonmin, lonmax = data_p.lon.min() +180 +23, data_p.lon.max()+180 -22
        latmin, latmax = data_p.lat.min() +13, data_p.lat.max()-3
        
        # print(f'\nPlot figure with lonmin: {lonmin}, \t lonmax: {lonmax} \nlatmin:{latmin}, \t latmax: {latmax}\n')

        ax_inset.set_extent([lonmin, lonmax, latmin, latmax], crs=inset_crs)



    # if log_pres:
    ax.set_yscale('symlog')
    ax.set_yticklabels(np.arange(1000, 290, -100))
    ax.set_yticks(np.arange(1000, 290, -100))

    ax.set_ylim(cross_p[name_vertical_coord].max(), 300)
    # Show the plot if no ax was provided
    if ax is None:
        plt.show()
#==================================================================================================================================================================================================================
#==================================================================================================================================================================================================================




def add_inset_plot(ax_inset, inset_crs, data_p, cross_p, start, end, cax=None, inset_contourf=None):
    # Define the CRS and inset axes
    data_crs = data_p['SLP'].metpy.cartopy_crs
    
    # Plot geopotential height at 500 hPa using contour
    ax_inset.contour(data_p['lon'], data_p['lat'], data_p['SLP'], levels=np.arange(900, 1100, 5), linestyles='-', colors='k', alpha=0.4, transform=data_crs)

    contourf_properties = {
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

        # Create colorbar
        cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
        # Move ticks and labels above the colorbar
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        # Add colorbar label above the ticks
        cbar.ax.set_title('Vertical integrated BCP [K/h]', fontsize=10, pad=5)


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
    ax_inset.gridlines(ylocs=np.arange(30, 91, 10), xlocs=np.arange(-180, 181, 10), alpha=0.5)
    ax_inset.set_xticks(np.arange(-180, 181, 10), crs=ccrs.PlateCarree())
    ax_inset.set_yticks(np.arange(30, 91, 10), crs=ccrs.PlateCarree())
    ax_inset.xaxis.set_major_formatter(LongitudeFormatter())
    ax_inset.yaxis.set_major_formatter(LatitudeFormatter())

    return ax_inset








