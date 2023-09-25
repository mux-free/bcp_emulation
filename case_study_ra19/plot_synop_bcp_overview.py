
# Plotting Function
def myplot_subplot(myax,
                   mycax,
                   data_contour,
                   data_contourf,
                   axes_titles,
                   uv_wind =None
                   ):
    
    """
    This function creates a subplot including a colorbar. 
    Plotted:    Contour-lines
                Contourf-colors ( + corresponding colorbar)
                Windvectors
    """

    cmap = plt.cm.nipy_spectral     # mpl.cm.viridis    # plt.cm.RdYlBu   
    levs =  np.linspace( data_contourf.min() , data_contourf.max() , 11 )                     #np.round(delta_levs/20,1)))
    norm = mpl.colors.BoundaryNorm(levs,cmap.N)



    # map
    myax.coastlines(linewidth=0.25)
    myax.gridlines(ylocs=np.arange(-90, 91, 10), xlocs=np.arange(-180, 181, 10))
    # set the ticks
    myax.set_xticks(np.arange(-180, 181, 10), crs=ccrs.PlateCarree());
    myax.set_yticks(np.arange(-90, 91, 10), crs=ccrs.PlateCarree());
    # format the ticks as e.g 60302260Wtemp_ceil_array
    myax.xaxis.set_major_formatter(LongitudeFormatter())
    myax.yaxis.set_major_formatter(LatitudeFormatter())   
    # Title
    myax.set_title(axes_titles.get('title'))

    # Plot pressure contours
    E=myax.contour(X,Y , data_contour , np.arange(950,1050,5), linewidths=0.8, linestyles ='-', colors='black', zorder=3, alpha=0.8 ,transform=ccrs.PlateCarree() )
    myax.clabel(E, E.levels  )#=np.arange[950,1050,10], fontsize=15)

    img1=myax.contourf(X,Y, data_contourf, cmap=cmap , levels=levs , norm=norm, extend='both'  , transform=ccrs.PlateCarree() )

    cbar = plt.colorbar(img1, use_gridspec=True , orientation="vertical" ,cax=mycax)      ### Took out formatter: ,format='%3i'
    cbar.set_label(axes_titles.get('cbar_label'))
    # Plot Wind vectors
    if uv_wind != None:
        vect=myax.quiver(X[skip[1]],Y[skip[1]] ,  uv_wind[0][skip] ,uv_wind[1][skip], zorder=2  , alpha=0.8)         #, scale=1e3,   headwidth=4, headlength=5)
    
    myax.set_extent([lonrange[0],lonrange[1],latrange[0],latrange[1]] , ccrs.PlateCarree() )




if __name__=='__main__':
    
    ## Import modules
    import numpy as np 
    import pandas as pd
    import matplotlib.pyplot as plt 
    import xarray as xr 
    from matplotlib.gridspec import GridSpec
    import cartopy.crs as ccrs
    import matplotlib as mpl
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    import sys
    sys.path.append('/home/freimax/msc_thesis/scripts/helper_scripts/')
    from data_preprocessing_helpers import PRES_3d


    #======================================================================================================================================

    ### Load data
    # Set Path
    file_path = '/net/thermo/atmosdyn2/atroman/PACaccu/cdf'
    # Open files 
    ds_p_casestudy = xr.open_mfdataset(f'{file_path}/P*')
    ds_s_casestudy = xr.open_mfdataset(f'{file_path}/TH*')


    #------------------------------ FIXING BUGS ----------------------------------------------------------------------------
    # TH-data have problems with dimensions (lon,lat,time)
    ## Change model timestep to datetime object <- in TH files, time is given as model timestep, starting at 1950-01-01-00:00
    time_array =[]
    base_time = pd.to_datetime('1950-01-01-00')                         # Model start time
    for idx,time in enumerate( ds_s_casestudy.time.values):             # Loop through all time-steps
        time_array.append( base_time + pd.DateOffset(hours=int(time)))  # Add model-time (given in hours) to model start time

    ##Change coordinates to lon/lat format
    ds_s_new=ds_s_casestudy.rename({'dimx_PS': 'lon' , 'dimy_PS': 'lat' , 'dimz_TH': 'lev' , 'dimz_PS':'lev2' , 'time':'time' })
    ds_s_casestudy=ds_s_new.assign_coords(lon=(ds_s_new.lon * 0.4 - 180), lat=(ds_s_new.lat * 0.4 - 90) , time=time_array)



    ds_p_casestudy = ds_p_casestudy.squeeze('lev2')
    ds_s_casestudy = ds_s_casestudy.squeeze('lev2')
    #------------------------------------------------------------------------------------------------------------------------


    da_p_pres = PRES_3d(ds_p_casestudy, 'T')
    da_s_pres = PRES_3d(ds_s_casestudy, 'PV')




    #======================================================================================================================================#
    #                                              DATA SELECTION PLOT1                                                                    #
    #======================================================================================================================================#

    ### Extract the heating rates for below-cloud processes: Evaporation, Sublimation, Snowmelt for a given lonlatbox
    ### Take the average over every vertical model-level

    # Define lonlatbox
    lonrange = [130,180]
    latrange = [30,70]

    # Apply lonlatbox
    ds_p_lonlatbox = ds_p_casestudy.sel(lon=slice(lonrange[0],lonrange[1]), lat=slice(latrange[0],latrange[1]))
    ds_s_lonlatbox = ds_s_casestudy.sel(lon = slice(lonrange[0],lonrange[1]) , lat=slice(latrange[0],latrange[1]))
    da_PS = ds_s_casestudy.PS.sel(lon = slice(lonrange[0],lonrange[1]) , lat=slice(latrange[0],latrange[1]) )
    da_SLP = ds_p_casestudy.SLP.sel(lon = slice(lonrange[0],lonrange[1]) , lat=slice(latrange[0],latrange[1]) )

    #======================================================================================================================================




    #======================================================================================================================================#
    #                                              DATA SELECTION PLOT2                                                                    #
    ## Settings for plot TIME AND LEVEL
    #----------------------------------
    plot_times = ['2017-04-08-15', '2017-04-10-03', '2017-04-10-17']
    level = 0

    #======================================================================================================================================#

    ds_s_lev_lonlatbox_0 = ds_s_lonlatbox.sel(lev=level, time=plot_times[0])
    ds_p_lev_lonlatbox_0 = ds_p_lonlatbox.sel(lev=level, time=plot_times[0])
    da_SLP_0 = da_SLP.sel(time=plot_times[0])

    ds_s_lev_lonlatbox_1 = ds_s_lonlatbox.sel(lev=level, time=plot_times[1])
    ds_p_lev_lonlatbox_1 = ds_p_lonlatbox.sel(lev=level, time=plot_times[1])
    da_SLP_1 = da_SLP.sel(time=plot_times[1])

    ds_s_lev_lonlatbox_2 = ds_s_lonlatbox.sel(lev=level, time=plot_times[2])
    ds_p_lev_lonlatbox_2 = ds_p_lonlatbox.sel(lev=level, time=plot_times[2])
    da_SLP_2 = da_SLP.sel(time=plot_times[2])

    #------------------------------------------------------------------------------
    ## U and V components of wind gust for drawing vectors
    ugs_array_0 = ds_p_lonlatbox.sel(time=plot_times[0], lev=level).U.data
    vgs_array_0 = ds_p_lonlatbox.sel(time=plot_times[0], lev=level).V.data

    ugs_array_1 = ds_p_lonlatbox.sel(time=plot_times[1], lev=level).U.data
    vgs_array_1 = ds_p_lonlatbox.sel(time=plot_times[1], lev=level).V.data

    ugs_array_2 = ds_p_lonlatbox.sel(time=plot_times[2], lev=level).U.data
    vgs_array_2 = ds_p_lonlatbox.sel(time=plot_times[2], lev=level).V.data

    skip = (slice(None,None,5), slice(None,None,5))

    #-------------------------------------------------------------------------------------------------------
    # Calculate below-cloud processes
    bc_cooling_mean = []
    for idx,dt in enumerate(plot_times):
        foo1 = ds_p_lonlatbox.tsubi.sel(time=plot_times[idx]).mean(dim=('lev')) + \
                        ds_p_lonlatbox.tevr.sel(time=plot_times[idx]).mean(dim=('lev')) + \
                            ds_p_lonlatbox.tsubs.sel(time=plot_times[idx]).mean(dim=('lev')) + \
                                ds_p_lonlatbox.tmelti.sel(time=plot_times[idx]).mean(dim=('lev')) + \
                                    ds_p_lonlatbox.tmelts.sel(time=plot_times[idx]).mean(dim=('lev'))
        bc_cooling_mean.append(foo1)

    bc_cooling_min = []
    for idx,dt in enumerate(plot_times):
        foo1 = ds_p_lonlatbox.tsubi.sel(time=plot_times[idx]).min(dim=('lev')) + \
                        ds_p_lonlatbox.tevr.sel(time=plot_times[idx]).min(dim=('lev')) + \
                            ds_p_lonlatbox.tsubs.sel(time=plot_times[idx]).min(dim=('lev')) + \
                                ds_p_lonlatbox.tmelti.sel(time=plot_times[idx]).min(dim=('lev')) + \
                                    ds_p_lonlatbox.tmelts.sel(time=plot_times[idx]).min(dim=('lev'))
        bc_cooling_min.append(foo1)

    bc_cooling_sum = []
    for idx,dt in enumerate(plot_times):
        foo1 = ds_p_lonlatbox.tsubi.sel(time=plot_times[idx]).sum(dim=('lev')) + \
                        ds_p_lonlatbox.tevr.sel(time=plot_times[idx]).sum(dim=('lev')) + \
                            ds_p_lonlatbox.tsubs.sel(time=plot_times[idx]).sum(dim=('lev')) + \
                                ds_p_lonlatbox.tmelti.sel(time=plot_times[idx]).sum(dim=('lev')) + \
                                    ds_p_lonlatbox.tmelts.sel(time=plot_times[idx]).sum(dim=('lev'))
        bc_cooling_sum.append(foo1)
    #-------------------------------------------------------------------------------------------------------



    ############################################################################################################################################
    # START PLOTTING
    ############################################################################################################################################
    ### Open Plot ###
    fig = plt.subplots(figsize=(12,16))

    # Define grid for plots
    # gs  = GridSpec(nrows=6, ncols=2, height_ratios=[20, -2.4, 1.2, 20, -2.4, 1.2])  # arange plot windows
    gs = GridSpec(nrows=5,ncols=7,  height_ratios=[1 , 0.1 , 1 , 0.1 , 1  ],width_ratios=[1 , -0.02, 0.04, 0.2 , 1 , -0.02,  0.04])

    ## DEFINE MESHGRID
    X,Y = ds_s_lonlatbox.lon, ds_s_lonlatbox.lat

    # Define coloar abr axes
    cax1 = plt.subplot(gs[0,2])
    cax2 = plt.subplot(gs[0,6])
    cax3 = plt.subplot(gs[2,2])
    cax4 = plt.subplot(gs[2,6])
    cax5 = plt.subplot(gs[4,2])
    cax6 = plt.subplot(gs[4,6])
    # cax4 = plt.subplot(gs[5,1])    

    ## General layout things (title and space between figures)


    ax1=plt.subplot(gs[0,0], projection=ccrs.PlateCarree())
    ax2=plt.subplot(gs[0,4], projection=ccrs.PlateCarree())
    ax3=plt.subplot(gs[2,0], projection=ccrs.PlateCarree())
    ax4=plt.subplot(gs[2,4], projection=ccrs.PlateCarree())
    ax5=plt.subplot(gs[4,0], projection=ccrs.PlateCarree())
    ax6=plt.subplot(gs[4,4], projection=ccrs.PlateCarree())


    ############################################# UPPER LEFT #########################################################################
    #---------- data ----------------------------------#
    contour_data = da_SLP_0.values

    myplot_subplot(ax1, cax1, da_SLP_0.values, bc_cooling_sum[0], axes_titles={'title':plot_times[0], 'cbar_label':'Sum of B.C. cooling rates'}, uv_wind=[ugs_array_0 ,vgs_array_0])
    myplot_subplot(ax2, cax2, da_SLP_0.values, ds_p_lev_lonlatbox_0.Q.values, axes_titles={'title':plot_times[0], 'cbar_label':'Specific Humidity'}, uv_wind=[ugs_array_0 ,vgs_array_0])
    myplot_subplot(ax3, cax3, da_SLP_0.values, bc_cooling_sum[1], axes_titles={'title':plot_times[1], 'cbar_label':'Sum of B.C. cooling rates'}, uv_wind=[ugs_array_1 ,vgs_array_1])
    myplot_subplot(ax4, cax4, da_SLP_0.values, ds_p_lev_lonlatbox_1.Q.values, axes_titles={'title':plot_times[1], 'cbar_label':'Specific Humidity'}, uv_wind=[ugs_array_1 ,vgs_array_1])
    myplot_subplot(ax5, cax5, da_SLP_0.values, bc_cooling_sum[2], axes_titles={'title':plot_times[2], 'cbar_label':'Sum of B.C. cooling rates'}, uv_wind=[ugs_array_2 ,vgs_array_2])
    myplot_subplot(ax6, cax6, da_SLP_0.values, ds_p_lev_lonlatbox_2.Q.values, axes_titles={'title':plot_times[2], 'cbar_label':'Specific Humidity'}, uv_wind=[ugs_array_2 ,vgs_array_2])

    plt.savefig('/home/freimax/msc_thesis/figures/case_study_RA19/synoptic_overview_BC_Q.png', dpi=400) 