
import pandas as pd
import numpy as np
import xarray as xr
import seaborn as sns
import joblib
import sys
from glob import glob 
import wrf

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
# from matplotlib.gridspec import GridSpec
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors

sys.path.append('/home/freimax/msc_thesis/scripts/helpers/')
import helper

from sklearn.metrics import r2_score, mean_squared_error
from collections import Counter

sys.path.append('/home/freimax/msc_thesis/scripts/helpers/')
import data_preprocessing_helpers
import logging
#=======================================================================================================================================================================================================================================
#=======================================================================================================================================================================================================================================




## This functions loads and combines different trajectories from different cyclones using the load and predict along trajecotry functions

def load_trajs_predict_bcp(traj_file_path, 
                           model_tmeltsi, model_tsubsi, model_tevr, 
                           era5_data=False, 
                           folder_branch_list=None, parent_branch='NA', 
                           save_path=None,
                           debug=None):

    if era5_data:
        df_big = pd.DataFrame(
            columns=['time', 'lon', 'lat', 'p', 'tmeltsi_pred', 'tsubsi_pred', 'tevr_pred', 'Atmeltsi_pred', 'Atsubsi_pred', 'Atevr_pred', 'id'])
    else:
        df_big = pd.DataFrame(
            columns=['time','lon','lat','p','tmeltsi_pred','tsubsi_pred','tevr_pred','Atmeltsi_pred','Atsubsi_pred','Atevr_pred','id','tmeltsi','tsubsi','tevr','Atsubsi','Atmeltsi','Atevr'])


    id_CYC = 0
    counter=0

    ## Folder branch means if the trajectory fiels are saved inidfferent direcetories, e.g. in different months (IFS-18) or different location (ERA5). 
    ## For the case-study there is only one trajectory file (1 directory), therefore folder_branch = None
    if folder_branch_list is None:
        # Set ID to 1, since here we only have 1 cyclone
        id_CYC = 1
        file = traj_file_path 
        date = file[-25:-14]
        # Check that date has correct form
        if not (date[0:2]=='20' or date[0:2]=='19'):
            print(date[0:2])
            raise ValueError(f'\nDate has weird form: {date}\n') 
        
        dftraj    = load_clean_trajectories(file, era5_data=era5_data)
        dftraj_rf = predict_accumulated_cooling_trajectories(df_traj=dftraj, model_tsubsi=model_tsubsi, model_tmeltsi=model_tmeltsi, model_tevr=model_tevr)

        ## Add date, cyclone ID and month 
        dftraj_rf['date'] = date        
        dftraj_rf['month'] = 'APR17'        
        dftraj_rf['id_CYC'] = id_CYC
        
        
        
        # If df_big is not empty, add the maximum id in df_big to the id's of dftraj_rf
        if not df_big.empty:
            dftraj_rf['id'] = dftraj_rf['id'] + df_big['id'].max()
        ## Merge newly concatenated DataFrame with the previously created big DataFrame
        df_big = pd.concat([df_big, dftraj_rf])

    else:
        for folder_branch in folder_branch_list:

            ## Set file path
            if era5_data:
                files = glob(f"{traj_file_path}/{parent_branch}/{folder_branch}/Max-tracing*.txt")
            else:
                files = glob(f"{traj_file_path}/{folder_branch}/*.txt")
        

            # Configure the logging module
            log_path='/home/freimax/msc_thesis/scripts/ERA5'
            logging.basicConfig(filename=f'{log_path}/ERA5_traj_{parent_branch}_{folder_branch}_log.log',    # Log file name
                                level=logging.INFO,         # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                                format='%(asctime)s - %(levelname)s - %(message)s')  # Format for log messages
            
            
            print(f'----------------------------------------------- {folder_branch} ------ Nr. of files: {len(files)} -----------------------------------------------')
            logging.info(f'\n----------- Start with region {parent_branch} {folder_branch} ---- Nr. of files: {len(files)} --------')
                         
            path_processed_cyc=f'{save_path}/{folder_branch}'
            print(f'Trajectories files are saved in:   {path_processed_cyc}')
            
            ## Loop through all files
            for file in files:
                id_CYC += 1
                
                date = file[-25:-14]
                print(f'  Cyclone ID:\t{id_CYC}\t\tDate:\t{date}')
                
                # Check that date has correct form
                if not (date[0:2]=='20' or date[0:2]=='19'):
                    print(date[0:2])
                    raise ValueError(f'\nDate has weird form: {date}\n') 

                ## Call functions to load trajectories and predict bcp along them
                dftraj    = load_clean_trajectories(file, era5_data=era5_data)
                if dftraj is None:
                    print(f'\n\nWEIRD FILE: {file}\n\n')
                    continue 
                dftraj_rf = predict_accumulated_cooling_trajectories(df_traj=dftraj, model_tsubsi=model_tsubsi, model_tmeltsi=model_tmeltsi, model_tevr=model_tevr)


                ## Add date, cyclone ID and month 
                dftraj_rf['date'] = date        
                dftraj_rf['id_CYC'] = id_CYC

                if era5_data:
                    dftraj_rf['region'] = folder_branch
                else:
                    dftraj_rf['month'] = folder_branch      # for IFS, this is sometimes different than what is indicates by "date", becuase 35day simulations
                

                ## Save processed trajectories file                
                cols = dftraj_rf.columns.drop(['date', 'region'])
                dftraj_rf[cols] = dftraj_rf[cols].astype(float)  
                dftraj_rf.to_hdf(f'{path_processed_cyc}/cyc_{date}_id_{id_CYC}.h5', key='df', mode='w')
                
                
                ## Merge newly concatenated DataFrame with the previously created big DataFrame
                # df_big = pd.concat([df_big, dftraj_rf])

                if (id_CYC) % 10 == 0:
                    logging.info(f'Iteration {id_CYC} completed.')
            
        return dftraj_rf

            # df_big.to_hdf(f'{save_path}/df_big_{folder_branch}.h5', key='df', mode='w')
            # del df_big


    # ## Change dtype of columns to float (except for dtypes of 'date' and 'month')
    # if era5_data and folder_branch_list:
    #     cols = df_big.columns.drop(['date', 'region'])
    #     df_big[cols] = df_big[cols].astype(float)     
    # else: 
    #     try:  
    #         cols = df_big.columns.drop(['date', 'month'])
    #         df_big[cols] = df_big[cols].astype(float)
    #     except KeyError:
    #         cols = df_big.columns.drop(['date'])
    #         df_big[cols] = df_big[cols].astype(float)

    # return df_big






## This function loads the trajecortories (txt files), adds an ID to every taj. and inspects for NaN (-999) values  --> Output is a dataframe with all trajectories

def load_clean_trajectories(file, era5_data=False):
    # Load data (skip first 5 rows) and columns are seperated by spaces (maybe a tab)
    df_traj = pd.read_csv(file, delim_whitespace=True, skiprows=[0, 1, 3, 4])
    nrows=df_traj.shape[0]
    # Drop trajectories older than 48 hours
    #============================================================================================================================
    ## CLEAN DATA -- Drop invalid values
    #-----------------------------------
    ## Add a ID column
    # df_traj['id'] = np.zeros(df_traj.shape[0])
    df_traj.loc[:, 'id'] = np.zeros(df_traj.shape[0])
    id_counter1 = 0
    for idx, time in enumerate(df_traj.loc[:]['time']):
        if time == 0.:
            id_counter1 += 1
        df_traj.loc[idx, 'id'] = id_counter1
    # Drop trajectories where any point is invlaid (=-999.00) in 'p', 'lon', or 'lat'
    df_traj = df_traj.groupby('id').filter(lambda group: not ((group['p'] == -999.00).any() or (group['lon'] == -999.00).any() or (group['lat'] == -999.00).any()))
    df_traj.reset_index(drop=True, inplace=True)
    

    # Create a mask for invalid trajectories
    invalid_p = (df_traj.groupby('id')['p'].transform(lambda x: (x == -999.00).any()))
    # invalid_lon = (df_traj.groupby('id')['lon'].transform(lambda x: (x == -999.00).any()))
    # invalid_lat = (df_traj.groupby('id')['lat'].transform(lambda x: (x == -999.00).any()))
    # Combine the masks and filter out invalid trajectories
    mask_invalid = invalid_p #clear| invalid_lon | invalid_lat
    df_traj = df_traj[~mask_invalid]

    if df_traj.shape[0] - nrows > 0:
        id_counter = 0
        for idx, time in enumerate(df_traj.loc[:]['time']):
            ## ID is always the floored division of 48 against time
            if time == 0.:
                id_counter += 1
            df_traj.loc[idx, 'id'] = id_counter
        if (id_counter1-id_counter)>0:
            print(f'  Invalid values encountered, dropping them... (trajectories dropped: {int(id_counter1-id_counter)} of {id_counter1})')

    df_traj = df_traj[df_traj['time'] >= -48]

    #------------------------------------------------ ATTENTION, DIFFERENT OUTPUT FILES FOR IFS AND ERA5 TRAJS ------------------------------------------------
    
    if era5_data: 
        ## Subselect columns from LAGRANTO output based on inspection
        try:
            cols = ['time', 'lon', 'lat', 'p', 'PS', 'PV', 'TH', 'THE', 'OL', 'ZB','CP','LSP'] + list(df_traj.columns[-9:])
            # Select the desired columns from the DataFrame
            df_traj = df_traj[cols] 
        except KeyError:
            cols = ['time', 'lon', 'lat', 'p', 'PS', 'PV'] + list(df_traj.columns[-9:])
            df_traj = df_traj[cols] 

        # Remove ".1" or ".2" from column names
        df_traj.columns = df_traj.columns.str.replace("\.1$", "", regex=True).str.replace("\.2$", "", regex=True)
        
        # Check if all necessary columns are in the df_traj
        necessary_cols = ['LWC', 'RWC', 'IWC', 'SWC', 'Q', 'RH', 'T', 'OMEGA']
        assert set(necessary_cols).issubset(set(df_traj.columns)), 'Not all necessary columns are in the df_traj.'

        # Create a new 'scaled-' column 'SIWC' that is the sum of 'SWC' and 'IWC'
        df_traj['SIWC'] = df_traj['SWC'] / 1000 + df_traj['IWC'] / 1000
        # Rescale other columns (RWC, LWC, Q, T)
        df_traj['RWC'] = df_traj['RWC'] / 1000  
        df_traj['LWC'] = df_traj['LWC'] / 1000  
        df_traj['Q'] = df_traj['Q'] / 1000        ## Note, the random forest uses Q in units kg/kg (this is inconsistent with the other units, but Max didn't get around to change that)
        df_traj['T'] = df_traj['T'] - 273.15 


        assert_text1 = f"Water contents are not given in correct units! \n\tSIWC: {df_traj['SIWC'].max()} \n\tRWC: {df_traj['RWC'].max()} \t\nLWC: {df_traj['LWC'].max()} \n\tQ: {df_traj['Q'].max()}"
        if not (df_traj['SIWC'].max()<10) | (df_traj['LWC'].max()<10) | (df_traj['RWC'].max()<10) | (df_traj['Q'].max()<1):
            logging.warning(assert_text1)
            return None

        df_traj.drop(columns=['SWC', 'IWC'], inplace=True)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
    ###########     For IFS-18 DATA     ################ 
    else:
        df_traj['SIWC'] = df_traj['SWC'] + df_traj['IWC']
        df_traj['tmeltsi'] = df_traj['tmelts'] + df_traj['tmelti']
        df_traj['tsubsi'] = df_traj['tsubs'] + df_traj['tsubi']
        df_traj.drop(columns=['tmelts', 'tmelti', 'tsubs', 'tsubi', 'SWC', 'IWC'], inplace=True)

        ## Check that values have correct units
        if df_traj['SIWC'].max() > 100:
            df_traj['SIWC'] = df_traj['SIWC']/1000
        if df_traj['RWC'].max() > 100:
            df_traj['RWC'] = df_traj['RWC']/1000
        if df_traj['LWC'].max() > 100:
            df_traj['LWC'] = df_traj['LWC']/1000
        if df_traj['Q'].max() > 1:
            df_traj['Q'] = df_traj['Q']/1000

        if df_traj['tmeltsi'].min() < -100:
            df_traj['tmeltsi'] = df_traj['tmeltsi']/1000
        if df_traj['tsubsi'].min() < -100:
            df_traj['tsubsi'] = df_traj['tsubsi']/1000
        # if df_traj['tevr'].min() < -50:
        #     df_traj['tevr'] = df_traj['tevr']/1000

        assert_text2 = f"Cooling rates are wrong o.o.m.! \n\t tmeltsi: {df_traj['tmeltsi'].min()} \n\ttsubsi: {df_traj['tmeltsi'].min()}"
        assert (df_traj['tmeltsi']>-100).any() |  (df_traj['tmeltsi']>-100).any(),  assert_text2

        assert_text1 = f"Water contents are not given in correct units! \n\tSIWC: {df_traj['SIWC'].max()} \n\tRWC: {df_traj['RWC'].max()} \t\nLWC: {df_traj['LWC'].max()} \n\tQ: {df_traj['Q'].max()}"
        assert (df_traj['SIWC'].max()<10) | (df_traj['LWC'].max()<10) | (df_traj['RWC'].max()<10) | (df_traj['Q'].max()<1), assert_text1

        ## Accumulate cooling values (true values) along trajectories
        def accumulate_columns(df):
            df = df.sort_index(ascending=False)
            df['Atsubsi']  = df['tsubsi'].cumsum()
            df['Atmeltsi'] = df['tmeltsi'].cumsum()
            df['Atevr']    = df['tevr'].cumsum()
            return df
        ## Add a ID column
        id_counter = 0
        # for idx, time in enumerate(df_traj.loc[:]['time']):
        # Apply the function to each id-group
        df_traj = df_traj.groupby('id').apply(accumulate_columns)

        ## Reset the Multi-index to a single index
        df_traj = df_traj.reset_index('id', drop=True)
        #============================================================================================================================
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    ## Check that all trajs have same lenght
    t_end = np.min(df_traj['time'])
    mod_result = int(df_traj.shape[0] % (t_end-1))
    if mod_result != 0:
        logging.warning(f'Modulo of {t_end}-hr={int(df_traj.shape[0]%(t_end-1))}\tFile: {file}')
        return None
    # assert_string = f'Modulo of {t_end}-hours backward trajectories is not 0. Modulo={df_traj.shape[0]%(t_end-1)}'
    # assert int(df_traj.shape[0] % (t_end-1))==0, assert_string
    return df_traj





## This function returns a new dataframe with 
def predict_accumulated_cooling_trajectories(df_traj, model_tsubsi, model_tmeltsi, model_tevr):
    
    # print('  Make Predictions ')

    ## Exclude domain filtered values (SIWC=0, T=0)
    df_traj_tsubsi_filtered  = df_traj[df_traj.SIWC > 0]
    df_traj_tmeltsi_filtered = df_traj[(df_traj.SIWC > 0) & (df_traj['T'] > 0)]
    df_traj_tevr_filtered    = df_traj[df_traj.RWC > 0]

    ## create DataFrame of same shape of trajectory DataFrame and initialize columns for subsi and meltsi with zeros
    df_preds = pd.DataFrame(index=df_traj.index)
    df_preds['tsubsi_pred'] = 0 
    df_preds['tmeltsi_pred'] = 0 
    df_preds['tevr_pred'] = 0 

    ## Check if there are some any gridpoints that full-fill these conditions
    if not df_traj_tsubsi_filtered.empty: 
        # no_samples_flag = True
        y_pred_tsubsi  = model_tsubsi.predict(df_traj_tsubsi_filtered[['SIWC', 'LWC', 'RWC', 'RH', 'Q', 'OMEGA', 'T']])
        df_preds.loc[df_traj_tsubsi_filtered.index,  'tsubsi_pred'] = y_pred_tsubsi
    else:
        print('  No samples detected that pass domain filter (SIWC > 0)\n')
    
    if not df_traj_tmeltsi_filtered.empty: 
        # no_samples_flag = True
        y_pred_tmeltsi = model_tmeltsi.predict(df_traj_tmeltsi_filtered[['SIWC', 'LWC', 'RWC', 'RH', 'Q', 'OMEGA', 'T']])
        df_preds.loc[df_traj_tmeltsi_filtered.index, 'tmeltsi_pred'] = y_pred_tmeltsi
    else:
        print('  No samples detected that pass domain filter (T > 0)\n')
    
    if not df_traj_tevr_filtered.empty: 
        # no_samples_flag = True
        y_pred_tevr    = model_tevr.predict(df_traj_tevr_filtered[['SIWC', 'LWC', 'RWC', 'RH', 'Q', 'OMEGA', 'T']]) 
        df_preds.loc[df_traj_tevr_filtered.index, 'tevr_pred'] = y_pred_tevr
    else:
        print('  No samples detected that pass domain filter (RWC > 0)\n')


    df_traj['tsubsi_pred']  = df_preds['tsubsi_pred']
    df_traj['tmeltsi_pred'] = df_preds['tmeltsi_pred']    
    df_traj['tevr_pred']    = df_preds['tevr_pred']    
    
    def accumulate_columns_rf(df):
        df = df.sort_index(ascending=False)
        df['Atsubsi_pred']  = df['tsubsi_pred'].cumsum()
        df['Atmeltsi_pred'] = df['tmeltsi_pred'].cumsum()
        df['Atevr_pred']    = df['tevr_pred'].cumsum()
        return df

    # Apply the function to each id-group
    df_traj = df_traj.groupby('id').apply(accumulate_columns_rf)
    df_traj = df_traj.reset_index(drop=True)                # reset double-index created by .apply()
    # df_traj['NOBCP_FLAG'] = no_samples_flag

    return df_traj


#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -




































#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## This function scales all longitudes and latitudes, s.t. they have same dimensions (almost completely true, some inaccuracies wit scaling)

def get_composite(df):
    ## get cyclone IDs 
    cyc_id_list = np.unique(df['id_CYC'].values)
    
    df_composite = df.copy()
    
    for i in cyc_id_list:
        df_cyc = df_composite[df_composite['id_CYC']==i].copy()
        
        ## Get lon and lats for every cyclone id as arrays
        lons = df_cyc['lon'].copy()
        lats = df_cyc['lat'].copy()

        lons_unique = np.unique(lons)
        lats_unique = np.unique(lats)

        center_lon = lons_unique[int(np.floor(lons_unique.shape[0]/2))]
        center_lat = lats_unique[int(np.floor(lats_unique.shape[0]/2))]

        
        ## Multiply to longitude with the cosine of it's correpsonding latitude (to make them all evenly iszed)
        lat_in_rads = np.deg2rad( lats.values)
        lons_scaled = np.cos(lat_in_rads) * (lons - center_lon)  


        lats_shifted = lats - center_lat

        ## Round the longitudes and latitudes to 0.4 degrees
        df_composite.loc[df_composite['id_CYC']==i, 'lon'] = np.round(lons_scaled * 2) / 2
        df_composite.loc[df_composite['id_CYC']==i, 'lat'] = np.round(lats_shifted * 2) / 2
        
        
        #------------------------------------------------------------------
        ## Test where the new lon/lat center are
        cyc_idi_new = df_composite[df_composite['id_CYC']==i].copy()
        lons_new = np.unique(cyc_idi_new.lon)
        lats_new = np.unique(cyc_idi_new.lat)

        c_lon = lons_new[int(np.floor(lons_new.shape[0]/2))]
        c_lat = lats_new[int(np.floor(lats_new.shape[0]/2))]
        #------------------------------------------------------------------

        print(f'Center lon (before): {center_lon:4.1f}\t(after): {c_lon:2.1f} \t\t Center_lat: {center_lat:2.1f}\t(after): {c_lat:2.1f}\t\tLon_difference: {cyc_idi_new["lon"].max() - cyc_idi_new["lon"].min():.1f}')

    return df_composite




#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------







def get_avg_geostrophic_wind(df, cyc_id, radius, pres_level, show_plot=False):
    df_1cyc = df[df['id_CYC']==cyc_id]
    date = str(np.unique(df_1cyc['date'])[0])
    month = str(np.unique(df_1cyc['month'])[0])

    ifs_path = f'/net/helium/atmosdyn/IFS-1Y/{month}/cdf/P{date}'
    ds_p = xr.open_dataset(ifs_path)
    ds_p = ds_p.squeeze()


    ## Initialise longitudes and latitudes
    LAT = np.round(np.linspace(0,90,226),1)
    LON = np.round(np.linspace(-180,180,901),1)

    ## Retrieve the center longitutde and latitude of the cyclone
    center_lat = np.mean(np.unique(df_1cyc[df_1cyc['time']==0]['lat']))
    center_lon = np.mean(np.unique(df_1cyc[df_1cyc['time']==0]['lon']))

    ## Cal function that return concatenated array of all longitudes and another array with the correspondign latitudes
    CLONIDS, CLATIDS = helper.IFS_radial_ids_correct(radius,center_lat)

    addlon = CLONIDS + np.where(LON==np.round(center_lon,1))[0][0]



    ## Check if cyclone center mask is located across date-boundary (edge betweem 180e and -180e)
    crossing_dateboundary = False 
    if np.any(addlon > 900):
        # addlon[np.where((addlon-900)>0)] = addlon[np.where((addlon-900)>0)]-900
        clon = addlon - 450
        # Set flag to true that cyclone corsses date_boubdary
        crossing_dateboundary = True
    else:
        clon = addlon

    clat = CLATIDS.astype(int) + np.where(LAT==np.round(center_lat,1))[0][0]

    # Create a rectangular box around cyclone-center-engulfing cuircle with specified radius
    all_lons = np.arange(clon.min()-1, clon.max()+2, 1)
    all_lats = np.arange(clat.min()-1, clat.max()+2, 1)
    lon_grid, lat_grid = np.meshgrid(all_lons, all_lats)

    # Initialize mask that represents circle around cyclone center with specified radius 
    mask = np.full(lon_grid.shape, np.nan)


    # Check whether each pair is in the list that contains all coordiantes of circle around cyclone center
    for i in range(lon_grid.shape[0]):
        for j in range(lon_grid.shape[1]):
            if np.any((clon == lon_grid[i, j]) & (clat == lat_grid[i, j])):
                mask[i, j] = 1  # If point is inside the circle

    # Check if the longitude range crosses the date boundary
    if crossing_dateboundary:
        # Concatenate the data across the date boundary
        ds_region = xr.concat([ds_p.isel(lon=slice(450, 901)), ds_p.isel(lon=slice(0, 451))], dim='lon')
        # ds_region['lon'] = np.mod(ds_region['lon'], 360)  # wrap longitudes to 0-360 degrees
        ds_p_cyc = ds_region.isel(lon=slice(all_lons.min(),all_lons.max()+1), lat=slice(all_lats.min(), all_lats.max()+1))
        print('Here')
    else:
        ds_p_cyc = ds_p.isel(lon=slice(all_lons.min(),all_lons.max()+1), lat=slice(all_lats.min(), all_lats.max()+1))

    ## Cretae a 3d pressure field (pressure at every model level), this  is needed for interpolation
    PRES_cyc = data_preprocessing_helpers.PRES_3d(ds_p_cyc, ds_p_cyc['U'])
    ## Interpolate pressure field onto specified pressure-level 
    da_u_850 = wrf.interplevel(ds_p_cyc['U'], PRES_cyc, pres_level)
    da_v_850 = wrf.interplevel(ds_p_cyc['V'], PRES_cyc, pres_level)
    da_gs_850 = np.sqrt(da_u_850**2 + da_v_850**2)

    ## Only keep circular mask
    da_gs_850 = da_gs_850.where(mask == 1)

    ## Calculate mean wind on that pressure level
    mean_wind = np.nanmean(da_gs_850.values)
    print(f'Mean horizontal windspeed at 850hPa:\t{mean_wind:.2f} m/s')

    ## Create a oplot of wind at 850hPa
    if show_plot:
        if crossing_dateboundary:
            # Adjust longitude
            longitude = ds_p_cyc.lon.values
            longitude = (longitude + 360) % 360

            da_gs_850.coords['lon'] = longitude
            # Sort the DataArray
            da_gs_850 = da_gs_850.sortby('lon')
            da_gs_850.plot();
        else:
            da_gs_850.plot();

    return mean_wind





















### Same as function above, but for ERA5 data
def get_avg_geostrophic_wind_era5(df, cyc_id, radius=200, show_plot=False, verbose=0):
    
    
    df_1cyc = df[df['id_CYC']==cyc_id]
    date = df_1cyc['date'].unique()[0]
    year  = date[0:4]
    num_month = date[4:6]

    ## Load the ERA5 file
    era5_path = f'/net/thermo/atmosdyn/era5/cdf/{year}/{num_month}'
    ds_p = xr.open_dataset(f'{era5_path}/Z{date}')
    ds_p = ds_p.sel(lat=slice(0,90)).squeeze()

    gridpoints_lon = ds_p.lon.shape[0]
    gridpoints_lat = ds_p.lat.shape[0]
    ## Initialise longitudes and latitudes
    LAT = np.linspace(0,90,gridpoints_lat)
    LON = np.linspace(-180,179.5,gridpoints_lon)

    ## Retrieve the center longitutde and latitude of the cyclone as the median of the points
    center_lon = np.mean(np.unique(df_1cyc[df_1cyc['time']==0]['lon']))
    center_lat = np.mean(np.unique(df_1cyc[df_1cyc['time']==0]['lat']))


    ## Cal function that return concatenated array of all longitudes and another array with the correspondign latitudes
    CLONIDS, CLATIDS = helper.ERA5_radial_ids_correct(radius,center_lat)

    center_lon_rounded = np.round(center_lon * 2) / 2
    center_lat_rounded = np.round(center_lat * 2) / 2

    addlon = CLONIDS + np.where(LON==center_lon_rounded)[0][0]

    ## Check if any point (integer of grid) touches the dateboundary
    crossing_dateboundary = False 
    if np.any(addlon == (gridpoints_lon-1)):    # Check if any of the mask points has lon-index value of 719 (correpsonds to left-most point)
        clon = addlon - (gridpoints_lon)/2      # Shift grid by 180 degrees to avoid date-boundary
        crossing_dateboundary = True            # Set flag, since it will be needed later

    elif np.any(addlon < 1):                    # Check if left-most point of mask corses dateboundary
        clon = addlon + (gridpoints_lon)/2
        crossing_dateboundary = True

    else:
        clon = addlon
    clat = CLATIDS.astype(int) + np.where(LAT==center_lat_rounded)[0][0]

    # Create a rectangular box around cyclone-center-engulfing cuircle with specified radius
    all_lons = np.arange(clon.min()-1, clon.max()+2, 1)
    all_lats = np.arange(clat.min()-1, clat.max()+2, 1)
    lon_grid, lat_grid = np.meshgrid(all_lons, all_lats)

    # Initialize mask that represents circle around cyclone center with specified radius 
    mask = np.full(lon_grid.shape, np.nan)
    # Check whether each pair is in the list that contains all coordiantes of circle around cyclone center
    for i in range(lon_grid.shape[0]):
        for j in range(lon_grid.shape[1]):
            if np.any((clon == lon_grid[i, j]) & (clat == lat_grid[i, j])):
                mask[i, j] = 1  # If point is inside the circle

    # If date_boundary crossing flag is set, concatenate the era5 fields around the date-boundary
    if crossing_dateboundary:
        # Concatenate the data across the date boundary
        ds_region = xr.concat([ds_p.sel(lon=slice(0, 179.5)), ds_p.sel(lon=slice(-180, -0.5))], dim='lon')
        # ds_region['lon'] = np.mod(ds_region['lon'], 360)  # wrap longitudes to 0-360 degrees
        all_lons = all_lons.astype(int)
        all_lats = all_lats.astype(int)
        ds_p_cyc = ds_region.isel(lon=slice(all_lons.min(),all_lons.max()+1), lat=slice(all_lats.min(), all_lats.max()+1))
    else:
        ds_p_cyc = ds_p.isel(lon=slice(all_lons.min(),all_lons.max()+1), lat=slice(all_lats.min(), all_lats.max()+1))


    ## Interpolate pressure field onto specified pressure-level 
    da_u_850 = ds_p_cyc['U'].sel(plev=85000)
    da_v_850 = ds_p_cyc['V'].sel(plev=85000)
    da_gs_850 = np.sqrt(da_u_850**2 + da_v_850**2)

    ## Only keep circular mask
    da_gs_850 = da_gs_850.where(mask == 1)

    ## Calculate mean wind on that pressure level
    mean_wind = np.nanmean(da_gs_850.values)
    if verbose > 0:
        print(f'Mean horizontal windspeed at 850hPa:\t{mean_wind:.2f} m/s')

    ## Create a oplot of wind at 850hPa
    if show_plot:
        if crossing_dateboundary:
            # Adjust longitude
            longitude = ds_p_cyc.lon.values
            longitude = (longitude + 360) % 360
            da_gs_850.coords['lon'] = longitude
            # Sort the DataArray
            da_gs_850 = da_gs_850.sortby('lon')
            da_gs_850.plot();
        else:
            da_gs_850.plot();

    return mean_wind
















