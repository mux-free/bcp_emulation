## Import modules

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap


#==================================================================================================================================================================================================================
#==================================================================================================================================================================================================================
#==================================================================================================================================================================================================================

def plot_cross_section_test(cross_p, 
                            start, 
                            end,
                            plot_contourf,
                            ax=None,
                            contourf_colorbar=True,
                            ):
    fig = ax.get_figure()
    ## Define what happens if fiels is not defined
    if ax is None:
        fig, ax = plt.subplots()
    if plot_contourf is None:
        plot_contourf = []

    if not hasattr(fig, 'plotted_fields'):
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
        'tsubsi'          : {'levels': np.arange(-1.5, -0.074, 0.025), 'cmap': 'inferno' , 'alpha': 0.75, 'cbar_title': 'T sub S&I [K/hr]'       , 'extend':'min'  , 'cbar_id':'cbar_tsubsi'     },
        'tmeltsi'         : {'levels': np.arange(-1.5, -0.074, 0.025), 'cmap': 'Blues_r' , 'alpha': 0.75, 'cbar_title': 'T melt S&I [K/hr]'      , 'extend':'min'  , 'cbar_id':'cbar_tmeltsi'    },
        'tsubsi_pred'     : {'levels': np.arange(-1.5, -0.074, 0.025), 'cmap': 'inferno' , 'alpha': 0.75, 'cbar_title': 'Pred Tsub S&I [K/hr]'   , 'extend':'min'  , 'cbar_id':'cbar_tsubsi'     },
        'tmeltsi_pred'    : {'levels': np.arange(-1.5, -0.074, 0.025), 'cmap': 'Blues_r' , 'alpha': 0.75, 'cbar_title': 'Pred Tmelt S&I [K/hr]'  , 'extend':'min'  , 'cbar_id':'cbar_tmeltsi'    },
        'residual_tsubsi' : {'levels': np.arange(-1.0, 1.1, 0.1)     , 'cmap': 'seismic' , 'alpha': 0.75, 'cbar_title': 'Res Tsubsi (true-pred)' , 'extend':'both' , 'cbar_id':'residual_tsubsi' },
        'residual_tmeltsi': {'levels': np.arange(-1.0, 1.1, 0.1)     , 'cmap': 'PuOr'    , 'alpha': 0.75, 'cbar_title': 'Res Tmeltsi (true-pred)', 'extend':'both' , 'cbar_id':'residual_tmeltsi'},
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
                print(f'Plot Contourf {field}')
                
                alpha = contourf_properties[field]['alpha']
                levels = contourf_properties[field]['levels']
                cmap = contourf_properties[field]['cmap']
                
                contourf = ax.contourf( x_axis, cross_p['level'], res_field, levels=levels, cmap=cmap, alpha=alpha, extend=contourf_properties[field]['extend'],)
                contourf_plots.append(contourf)            
            #-----------------------------------------------------------------------
            # Non-residual fields
            else:
                print(f'Plot Contourf {field}')
                alpha = contourf_properties[field]['alpha']
                levels = contourf_properties[field]['levels']
                cmap = contourf_properties[field]['cmap']
                
                contourf = ax.contourf( x_axis, cross_p['level'], cross_p[field], levels=levels, cmap=cmap, alpha=alpha, extend=contourf_properties[field]['extend'], )
                contourf_plots.append(contourf)
        else:
            print(f'\n FIELD ({field}) DOES NOT HAVE DEFINED PROPERTIES AND WAS THEREFOR NOT PLOTTED! \n Modify "filed_properties" dictonary to add this field... \n')
    

    #### Handle ColorBars
    if contourf_colorbar:
        # Add the colorbars for each contourf plot
        colorbar_width = 0.125
        colorbar_space = 0.02
        colorbar_start_x = 0.02

        bbox = ax.get_position()
        if bbox.x0 > 0.35:
            colorbar_start_x += 3 * (colorbar_width + colorbar_space)
        cbar_count=0
        for i, (contourf_plot, field) in enumerate(zip(contourf_plots, plot_contourf)):
            if field in contourf_properties:
                # Increase cbar count by 1
                cbar_count += 1
                
                cbar_ID = contourf_properties[field]['cbar_id']  ## Extract the cmap ID of the field
                if cbar_ID not in fig.plotted_fields:  # Check if the colorbar has been plotted
                    fig.plotted_fields.append(cbar_ID)  # Add colorbar to the list of already plotted
                    if cbar_count <= 3:                 # Check it his colorbar still fits below the images
                        #fig = ax.get_figure()
                        # Create an axis for the colorbar at the specified position
                        cbar_ax = fig.add_axes([colorbar_start_x + i * (colorbar_width + colorbar_space),   #left
                                                0.05,                                                       # bottom
                                                colorbar_width,                                             # width
                                                0.015])                                                      # height
                        # Create the colorbar using the contourf plot
                        cbar = fig.colorbar(contourf_plot, cax=cbar_ax, orientation='horizontal')
                        
                        ## Adjust ticks of cbar
                        cbar_ticks = np.round(np.linspace(cbar.vmin, cbar.vmax, 5)*10)/10
                        cbar_ticks[0], cbar_ticks[-1] = np.round(cbar.vmin*1000)/1000, np.round(cbar.vmax*1000)/1000
                        cbar.set_ticks(cbar_ticks)                        
                        
                        cbar_label = contourf_properties[field]['cbar_title']
                        cbar.set_label(cbar_label)
                        cbar.ax.set_xticklabels(cbar.get_ticks(), rotation=45)


                    else:
                        print(f'\n\nATTENTION: MORE THAN 3 COLORBARS\n')
                        cbar_ax = fig.add_axes([0.05, 0.15, 0.0075, 0.4 ])                                                  
                        # Create the colorbar using the contourf plot
                        cbar = fig.colorbar(contourf_plot, cax=cbar_ax, orientation='vertical')
                        cbar_label = contourf_properties[field]['cbar_title']
                        cbar.set_label(cbar_label)
                    
        ax.set_ylim(cross_p['level'].max(), cross_p['level'].min())
    




 





