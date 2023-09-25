## Load modules
import xarray as xr
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

import seaborn as sns
import matplotlib.ticker as ticker
import pickle

import sys
sys.path.append('/home/freimax/msc_thesis/scripts/helpers/')
from data_preprocessing_helpers import get_lonlatbox, calculate_rh_ifs, PRES_3d, interpolate_pres, get_cross_section_data
sys.path.append('/home/freimax/msc_thesis/scripts/plotting_functions/')
from plot_functions import plot_cross_section
sys.path.append('/home/freimax/msc_thesis/scripts/random_forest/')
import rf_functions


path = '/net/helium/atmosdyn/IFS-1Y/JAN18/cdf'
ds_p_all = xr.open_mfdataset(f'{path}/P2018011*')
ds_p_all = ds_p_all.squeeze('lev2')
ds_s_all = xr.open_mfdataset(f'{path}/S2018011*')
ds_s_all = ds_s_all.squeeze('lev_PS')


lonrange = [-180,-50]
latrange = [20,80]
ds_p = get_lonlatbox(ds_p_all, lon_range=lonrange, lat_range=latrange)
ds_s = get_lonlatbox(ds_s_all, lon_range=lonrange, lat_range=latrange)

ds_p['tsubsi']  = ds_p['tsubs'] + ds_p['tsubi']
ds_p['tmeltsi'] = ds_p['tmelts']+ ds_p['tmelti']
ds_p['SIWC']    = ds_p['SWC']+ ds_p['IWC']
ds_p['TH'] = ds_s['TH']
ds_p['THE']= ds_s['THE']

ds_subset   = ds_p.sel(time=slice('20180113-06','20180113-08'), lon=slice(-170,-130), lat=slice(25,55))
## Convert predicitons to pressure-level data:
ds_subset['pres_3d'] = PRES_3d(ds_subset, ds_subset['tsubsi'])

## Calculate RH_ifs
pres_4d = PRES_3d(ds_subset, ds_subset.T) 
ds_subset['RH_ifs'] = calculate_rh_ifs(pres_4d, ds_subset.Q, ds_subset.T)


## Interpolate data and predicitons to pressure levels
ds_subset_p = interpolate_pres(ds_subset, ds_subset['pres_3d'])
# Add SLP
ds_subset_p['SLP'] = ds_subset.SLP




#--------------------------------------------------------------------------------------------------------------------
## Load classifier
path = '/home/freimax/msc_thesis/scripts/random_forest/models'
with open(f'{path}/rf_classifier_tsubsi.pickle', 'rb') as f:
    rf_classifier_tsubsi = pickle.load(f)

path = '/home/freimax/msc_thesis/scripts/random_forest/models'
with open(f'{path}/rf_classifier_tmeltsi.pickle', 'rb') as f:
    rf_classifier_tmeltsi = pickle.load(f)

path = '/home/freimax/msc_thesis/scripts/random_forest/models'
with open(f'{path}/rf_classifier_tevr.pickle', 'rb') as f:
    rf_classifier_tevr = pickle.load(f)

## MAKE PREDICTIONS -----------------------------------------------------------------------------------------
da_y_pred_tsubsi = rf_functions.predict_bcp_labels(ds=ds_subset, 
                                                   rf_classifier=rf_classifier_tsubsi,
                                                   water_type='SIWC',
                                                   feature_names=['SIWC','RH_ifs','T','CC','OMEGA'], 
                                                   type_filteredvalues='zero',
                                                   add_temp_filter=False)

da_y_pred_tmeltsi = rf_functions.predict_bcp_labels(ds=ds_subset, 
                                                    rf_classifier=rf_classifier_tmeltsi,
                                                    water_type='SIWC',
                                                    feature_names=['SIWC','RH_ifs','T','CC','OMEGA'], 
                                                    type_filteredvalues='zero',
                                                    add_temp_filter=True)

da_y_pred_tevr = rf_functions.predict_bcp_labels(ds=ds_subset, 
                                                 rf_classifier=rf_classifier_tevr,
                                                 water_type='RWC',
                                                 feature_names=['RWC','RH_ifs','T','CC','OMEGA'], 
                                                 type_filteredvalues='zero',
                                                 add_temp_filter=False)

da_y_tsubsi_pres  = interpolate_pres(da_y_pred_tsubsi, ds_subset['pres_3d'])
da_y_tmeltsi_pres = interpolate_pres(da_y_pred_tmeltsi, ds_subset['pres_3d'])
da_y_tevr_pres    = interpolate_pres(da_y_pred_tevr, ds_subset['pres_3d'])


## Set values to labels 0,1,2
def set_values_to_labels(da): 
    da = da.where((da <= 1.5) | np.isnan(da), 2)
    da = da.where((da < 0.5) | (da > 1.5) | np.isnan(da), 1)
    da = da.where((da >= 0.5) | np.isnan(da), 0)
    return da

da_y_tsubsi_pres  = set_values_to_labels(da_y_tsubsi_pres)
da_y_tmeltsi_pres = set_values_to_labels(da_y_tmeltsi_pres)
da_y_tevr_pres    = set_values_to_labels(da_y_tevr_pres)

ds_subset_p['y_pred_tsubsi']  = da_y_tsubsi_pres
ds_subset_p['y_pred_tmeltsi'] = da_y_tmeltsi_pres
ds_subset_p['y_pred_tevr']    = da_y_tevr_pres



#=====================================#
# Plot cross-sections                 #
#=====================================#

special_note = []
# Define time-step                    #
timestep = '20180113-07'

## Choose contourf and contour
plot_cf1 = ['tevr', 'tsubsi', 'tmeltsi', 'OMEGA', ]
plot_c1  = [ 'RH_ifs', 'isotherms', 'TH', 'tsubsi_thr_weak', 'tsubsi_thr_strong', 'tmeltsi_thr_weak', 'tmeltsi_thr_strong', 'tevr_thr_weak', 'tevr_thr_strong']

plot_cf2 = ['RH_ifs']
plot_c2  = ['isotherms', 'CC', 'TH']
########################################################################################################

lon_start1, lon_end1 = -165.0, -140.0  
lat_start1, lat_end1 =  50.0 ,  30.0    

lon_start2, lon_end2 = -165.0, -135.0  
lat_start2, lat_end2 =  45.0 ,  45.0    

lon_start3, lon_end3 = -160.0, -140.0  
lat_start3, lat_end3 =  35.0 ,  50.0    

lon_start4, lon_end4 = -155.0, -155.0  
lat_start4, lat_end4 =  30.0 ,  55.0    

lon_start5, lon_end5 = -150.0, -140.0  
lat_start5, lat_end5 =  40.0 ,  35.0    

lon_start6, lon_end6 = -160.0, -145.0  
lat_start6, lat_end6 =  50.0 ,  50.0    

lon_start7, lon_end7 = -165.0, -135.0  
lat_start7, lat_end7 =  35.0 ,  35.0    


lon_starts = [lon_start1, lon_start2, lon_start3, lon_start4, lon_start5, lon_start6, lon_start7]
lon_ends   = [lon_end1, lon_end2, lon_end3, lon_end4, lon_end5, lon_end6, lon_end7]
lat_starts = [lat_start1, lat_start2, lat_start3, lat_start4, lat_start5, lat_start6, lat_start7]
lat_ends   = [lat_end1, lat_end2, lat_end3, lat_end4, lat_end5, lat_end6, lat_end7]







for lons, lats, lone, late in zip(lon_starts, lat_starts, lon_ends, lat_ends):
    start = (lats, lons)
    end = (late, lone)
    ##Prepare data
    print('Prepare Data for plotting')
    data_p  = get_cross_section_data(data=ds_subset_p, start=start, end=end, timestep=timestep).get('data_p')
    cross_p = get_cross_section_data(data=ds_subset_p, start=start, end=end, timestep=timestep).get('cross_p')
    tbcp_all_sum = get_cross_section_data(data=ds_subset_p, start=start, end=end, timestep=timestep).get('tbcp_all')
    

    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    # Create Figure
    print(f'\nPlot with lon/lat: {start[1]}/{start[0]} to {end[1]}/{end[0]}')
    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(1, 2)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    # Set the suptitle
    fig.suptitle(f'Cross-Section \u2013 {start} to {end} \u2013 \n'
                f'Time: {cross_p["time"].dt.strftime("%Y-%m-%d %H:%M").item()}', fontsize=16, y=1.0)

    #---------------------------------------------------------------------------------------------#

    img1 = plot_cross_section(ax=ax1, cross_p=cross_p, data_p=data_p, start=start, end=end, plot_contourf=plot_cf1, plot_contour=plot_c1,    
                                show_wind_barbs=False, show_clouds=False, show_precip=False, inset_contourf='all_bcp_sums', baseline_bcp=False, rf_bcp=True)
    
    img2 = plot_cross_section(ax=ax2, cross_p=cross_p, data_p=data_p, start=start, end=end, plot_contourf=plot_cf2, plot_contour=plot_c2,    
                                show_wind_barbs=True, show_clouds=True,  show_precip=True, inset_contourf=None, baseline_bcp=None)

    plt.savefig(f'/home/freimax/msc_thesis/figures/IFS_sim/RF_tests/Cyc_Jan2018/crosec_from{start}_to{end}_{special_note}_t{cross_p["time"].dt.strftime("%Y%m%d-%H").item()}_rfnew.png', dpi=250)