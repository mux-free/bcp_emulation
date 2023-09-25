import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import sys

import joblib
import glob
import warnings
sys.path.append('/home/freimax/msc_thesis/scripts/helpers/')
import helper
import data_preprocessing_helpers
sys.path.append('/home/freimax/msc_thesis/scripts/random_forest/')
import rf_functions


def subsampling_by_time_spacing(file_list, day_spacing=2, hour_spacing=6):
    """
    Select files from a list with given day and hour spacings.
    
    Parameters:
    file_list: list of files, sorted chronologically and with 24 files per day
    day_spacing: integer, number of days to skip
    hour_spacing: integer, number of hours to skip, should not be greater than 24

    Returns:
    A list of files selected with the specified day and hour spacings
    """

    # Check if day_spacing and hour_spacing are integers and hour_spacing is not more than 24
    if int(hour_spacing) > 24:
        raise ValueError('hour_spacing should not be greater than 24')

    # Split the list into chunks of 24 (one chunk for each day)
    files_daily_nested_list = [file_list[i:i+24] for i in range(0, len(file_list), 24)]

    # Take every second day
    files_sparse_dayly_nested_list = files_daily_nested_list[::int(day_spacing)]

    # Flatten the nested list back to a list
    files_every_n_day_flat = []
    for daily_list in files_sparse_dayly_nested_list:
        for day in daily_list:
            files_every_n_day_flat.append(day)

    # Select what hourly spacing we should keep
    files_i_hour_every_n_day = files_every_n_day_flat[::int(hour_spacing)]

    return files_i_hour_every_n_day



def compute_season_climatology(start_year, end_year, season, model_tsubsi, model_tmeltsi, model_tevr):
    """
    Compute the seasonal climatology over a range of years for a given season.
    """
    # Create a list of years
    years = [str(year) for year in range(start_year-1, end_year+1)]

    for idx_yr in range(1,len(years)):
        ## Select the season to compute
        if season == 'DJF':
            # Treat case that December is from the previous year
            months = [years[idx_yr-1]+'12', years[idx_yr]+'01', years[idx_yr]+'02']
        elif season == 'MAM':
            months = [years[idx_yr]+'03', years[idx_yr]+'04', years[idx_yr]+'05']
        elif season == 'JJA':
            months = [years[idx_yr]+'06', years[idx_yr]+'07', years[idx_yr]+'08']
        elif season == 'SON':
            months = [years[idx_yr]+'09', years[idx_yr]+'10', years[idx_yr]+'11']
        else:
            raise ValueError(f"Invalid season '{season}'. Accepted values are 'DJF', 'MAM', 'JJA', 'SON'.")        
        
        pfile_list, sfile_list = [], []
        ## Get a list wit all file-directories
        number_of_hours = 0
        for year_month in months:
            # Prepare the file pattern
            era5_path = f'/net/thermo/atmosdyn/era5/cdf/{year_month[0:4]}/{year_month[4:6]}'
            pfile_path = f'{era5_path}/P{year_month}*'
            sfile_path = f'{era5_path}/S{year_month}*'

            number_of_hours += (len(glob.glob(sfile_path)))

            # Use glob to get the list of files
            pfile_list.extend(glob.glob(pfile_path))
            sfile_list.extend(glob.glob(sfile_path))

            # Remove any tmeporary files in directory
            pfile_list = [file for file in pfile_list if not file.endswith('.tmp')]
            sfile_list = [file for file in sfile_list if not file.endswith('.tmp')]


        ## Subsample directory list to keep only every second day and every 6th hour
        pfile_list_subsamp = subsampling_by_time_spacing(pfile_list, day_spacing=2, hour_spacing=6)
        sfile_list_subsamp = subsampling_by_time_spacing(sfile_list, day_spacing=2, hour_spacing=6)

        nr_files = len(pfile_list_subsamp)

        print(f'Processing year: {years[idx_yr]}\twith months:',  *months, f'\tTotal hours contained in month: {number_of_hours}\tFiles after subsampling: {nr_files} ')

        ## Initialize the dataset new for a new year
        datasets = []
        for (p_path, s_path) in zip(pfile_list_subsamp, sfile_list_subsamp):
            # print('Load Data')
            ds_era5 = xr.open_mfdataset(p_path, drop_variables=['LSP','CP','SF','SSHF','SLHF','BLH','U','V'])
            ds_sera = xr.open_mfdataset(s_path, drop_variables=['THE','PV','TH','hyai','hybi','hyam','hybm'])
            ds_era5['RH'] = ds_sera['RH']
            ds_era5['SIWC'] = (ds_era5['SWC'] +  ds_era5['IWC']) * 1000         # Convert from kg/kg to g/kg
            ds_era5['RWC'] = ds_era5['RWC'] * 1000                              # Convert from kg/kg to g/kg
            ds_era5['LWC'] = ds_era5['LWC'] * 1000                              # Convert from kg/kg to g/kg
            # ds_era5['Q'] = ds_era5['Q'] * 1000                                # NOTE: in my RF-model Q is given in kg/kg 
            ds_era5['T'] = ds_era5['T'] - 273.15                                # Convert to celsius
            ds_era5 = ds_era5.drop_vars(['SWC', 'IWC'])
            del ds_sera

            ## Define highest model-level
            top_level=60
            ds_era5 = ds_era5.sel(lat=slice(0,90)).isel(lev=slice(-top_level,None)).squeeze()


            ## Apply Random Forest to whole dataset
            variables = ['SIWC', 'LWC', 'RWC', 'RH', 'Q', 'OMEGA', 'T']
            
            
            
            
            # input_array = np.stack([ds_era5[var].values for var in variables], axis=-1)
            # # Reshape the input array to 2D
            # input_2d = input_array.reshape(-1, len(variables))
            # # df_X = pd.DataFrame(input_2d, columns=['SIWC', 'LWC', 'RWC', 'RH', 'Q', 'OMEGA', 'T'])


            # # print('Make Predictions')
            # with warnings.catch_warnings():
            #     # warnings.filterwarnings("ignore", message="X does not have valid feature names, but RandomForestRegressor was fitted with feature names")
            #     warnings.filterwarnings("ignore", category=UserWarning)
            #     y_pred_tsubsi = model_tsubsi.predict(input_2d  )
            #     y_pred_tmetlsi= model_tmeltsi.predict(input_2d )
            #     y_pred_tevr   = model_tevr.predict(input_2d   )                
            # # Reshape the predictions back to 3D
            # y_pred_3d_sub = y_pred_tsubsi.reshape(input_array.shape[:-1])
            # y_pred_3d_melt= y_pred_tmetlsi.reshape(input_array.shape[:-1])
            # y_pred_3d_evr = y_pred_tevr.reshape(input_array.shape[:-1])
            
            
            ## Make predictions for corss-section data
            features = ['SIWC','LWC','RWC','RH','Q','OMEGA','T']
            y_pred_tsubsi = rf_functions.make_predictions( ds_era5, model_tsubsi , water_type='SIWC', feature_names=variables, type_filteredvalues='nan', add_temp_filter=False, verbose=0)
            y_pred_tmetlsi = rf_functions.make_predictions(ds_era5, model_tmeltsi, water_type='SIWC', feature_names=variables, type_filteredvalues='nan', add_temp_filter=True , verbose=0)
            y_pred_tevr = rf_functions.make_predictions(   ds_era5, model_tevr   , water_type='RWC' , feature_names=variables, type_filteredvalues='nan', add_temp_filter=False, verbose=0)

            ds_bcp = xr.Dataset({'tsubsi_pred': y_pred_tsubsi, 'tmeltsi_pred': y_pred_tmetlsi, 'tevr_pred': y_pred_tevr, 'PS': ds_era5['PS'], 'hyam': ds_era5['hyam'][-top_level:], 'hybm': ds_era5['hybm'][-top_level:]})


            
            # # Extract the relevant dimensions and coordinates
            # relevant_dims = ['lev', 'lat', 'lon']
            # relevant_coords = {dim: ds_era5.coords[dim] for dim in relevant_dims}


            # # Create the new DataArray
            # tsubsi_pred = xr.DataArray(y_pred_3d_sub, coords=relevant_coords, dims=relevant_dims)
            # tmeltsi_pred= xr.DataArray(y_pred_3d_melt, coords=relevant_coords, dims=relevant_dims)
            # tevr_pred   = xr.DataArray(y_pred_3d_evr, coords=relevant_coords, dims=relevant_dims)

            # # Add it to a new xarray.Dataset or do something else with it
            # ds_bcp = xr.Dataset({'tsubsi_pred': tsubsi_pred, 'tmeltsi_pred': tmeltsi_pred, 'tevr_pred': tevr_pred, 'PS': ds_era5['PS'], 'hyam': ds_era5['hyam'][-top_level:], 'hybm': ds_era5['hybm'][-top_level:]})

            ## Interpolate result onto pressure levels
            pres3d = data_preprocessing_helpers.PRES_3d_era5(data_set_PS=ds_bcp, shape_var=ds_bcp['tsubsi_pred'], hya=ds_bcp.hyam.values, hyb=ds_bcp.hybm.values)
            ds_era_pres = data_preprocessing_helpers.interpolate_pres(data=ds_bcp, pres_field=pres3d, pressure_levels=np.arange(1000,300,-25))
            
            # print(f'Finished with timestep {p_path[-12:]}')
            # Append ds_bcp to datasets
            datasets.append(ds_era_pres)

        ## Concat all 3 datasets to get the dataset for the season
        ds_season = xr.concat(datasets, dim='time')
        ds_season_avg = ds_season.mean(dim='time')

        filename = f'/net/helium/atmosdyn/freimax/data_msc/ERA5/bcp_climatology/{years[idx_yr]}_{season}.nc'
        ds_season_avg.to_netcdf(filename)
    



if __name__ == '__main__':
    ## Load rf models
    model_tsubsi, model_tmeltsi, model_tevr = helper.load_rf_models()

    ## Compute 10yr-climatology for JJA and DJF
    compute_season_climatology(start_year=2010, end_year=2020, season='JJA', model_tmeltsi=model_tmeltsi, model_tsubsi=model_tsubsi, model_tevr=model_tevr)
    # compute_season_climatology(start_year=2010, end_year=2020, season='DJF', model_tmeltsi=model_tmeltsi, model_tsubsi=model_tsubsi, model_tevr=model_tevr)

