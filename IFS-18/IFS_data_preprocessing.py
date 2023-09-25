import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import dask
import tempfile
from dask.distributed import Client, LocalCluster

import pickle
import sys
sys.path.append('/home/freimax/msc_thesis/scripts/helpers/')
from data_preprocessing_helpers import calculate_rh_ifs
import shutil



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## Function to retireve the numerical value of every month: DEC -> 12
def get_numeric_month(month):
    # Create a dictionary to map month strings to numerical representation
    month_mapping = {
        'JAN': '01',
        'FEB': '02',
        'MAR': '03',
        'APR': '04',
        'MAY': '05',
        'JUN': '06',
        'JUL': '07',
        'AUG': '08',
        'SEP': '09',
        'OCT': '10',
        'NOV': '11',
        'DEC': '12' }
    # Map the month string to numerical representation using the dictionary
    num_month = month_mapping.get(month[0:3])
    return num_month



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def load_data(month):
    # define path names
    ifs_path = f'/net/helium/atmosdyn/IFS-1Y/{month}/cdf'    
    
    if month == 'MAR18':
        cyclone_mask = f'/net/helium/atmosdyn/IFS-1Y/{month}/features/tracking/CYCLONES_MAR13.nc'
    elif month == "OCT18":
        cyclone_mask = f'/net/helium/atmosdyn/IFS-1Y/{month}/features/tracking/CYCLONES.nc'
    else:
        cyclone_mask = f'/net/helium/atmosdyn/IFS-1Y/{month}/features/tracking/CYCLONES_{month}.nc'
    
    # LOAD DATA
    num_month = get_numeric_month(month)
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        ds_p_ifs = xr.open_mfdataset(f'{ifs_path}/P20{month[-2:]}{num_month}*')
        ds_s_ifs = xr.open_mfdataset(f'{ifs_path}/S20{month[-2:]}{num_month}*', drop_variables=['PS','PVRCONVT','PVRCONVM','PVRTURBT','PVRTURBM','PVRLS','PVRCOND','PVRSW','PVRLWH','PVRLWC','PVRDEP','PVREVC','PVREVR', 'PVRSUBI', 'PVRSUBS', 'PVRMELTI', 'PVRMELTS', 'PVRFRZ', 'PVRRIME','PVRBF','LABEL'])
        ds_cycmask = xr.open_dataset(cyclone_mask)


    ## Select varaibles of interest in ifs_data_p
    var_list_p = ['RWC', 'tevr', 'CC', 'T', 'OMEGA', 'Q', 'PS', 'SLP', 'LWC']   # SIWC, tsubsi and tmeltsi are latter added by combing snow adn ice immediately
    ds_ifs = ds_p_ifs[var_list_p]

    ## Assert that bot S and P datasts have same structure
    def assert_coords_equal(ds1, ds2):
        assert set(ds1.coords.keys()) == set(ds2.coords.keys()), "Coordinate names do not match"
        for coord in ds1.coords:
            assert np.all(ds1[coord] == ds2[coord]), f"Coordinate values for {coord} do not match"
    assert_coords_equal(ds_ifs, ds_s_ifs)
    
    
    var_list_s = ['RH', 'TH', 'THE', 'PV', 'VORT', 'P']
    ds_s_ifs_selected = ds_s_ifs[var_list_s]

    # Merge datasets
    ds_ifs = ds_ifs.merge(ds_s_ifs_selected)    

    print('DataSet S and P merged succesfully')

    # Make field combiantion (SNOW & Ice are viewd together)
    ds_ifs['SIWC'] = ds_p_ifs['SWC'] + ds_p_ifs['IWC']
    ds_ifs['tsubsi'] = ds_p_ifs['tsubi'] + ds_p_ifs['tsubs']
    ds_ifs['tmeltsi'] = ds_p_ifs['tmelti'] + ds_p_ifs['tmelts']
    # Get rid of empty dimensions
    ds_ifs = ds_ifs.squeeze()
    

    # Delte unnecessary datasets to free up memory
    del ds_p_ifs
    del ds_s_ifs


    ## Make sure the time dimesnion of both arrays are the same
    # cyc_mask sometimes starts at 01:00 
    if str(ds_cycmask.time[0].values) == str(ds_ifs.time[0].values):
        print('Start-time of ds_ifs and ds_cycmask are equal.')
    else:
        if ds_cycmask.time[0].values > ds_ifs.time[0].values:
            print(f'ds_cymask starts later:\t{str(ds_cycmask.time[0].values)}')
            ds_ifs = ds_ifs.sel(time=slice(str(ds_cycmask.time[0].values), str(ds_ifs.time[-1].values)))
        elif ds_cycmask.time[0].values < ds_ifs.time[0].values:
            print(f'ds_ifs starts later:\t{str(ds_ifs.time[0].values)}')
            ds_cycmask = ds_cycmask.sel(time=slice(str(ds_ifs.time[0].values), str(ds_cycmask.time[-1].values)))
        else:
            raise ValueError('Time-horizonts of ds_cycmask and ds_ifs couldnt be resolved')
        
    # Make sure endtime is the same
    if str(ds_cycmask.time[-1].values) == str(ds_ifs.time[-1].values):
        print('End-time of ds_ifs and ds_cycmask are equal.')
    else:
        if ds_cycmask.time[-1].values > ds_ifs.time[-1].values:
            print(f'ds_cymask ends later:\t{str(ds_cycmask.time[-1].values)}')
            ds_cycmask = ds_cycmask.sel(time=slice(str(ds_cycmask.time[0].values), str(ds_ifs.time[-1].values)))
        elif ds_cycmask.time[-1].values < ds_ifs.time[-1].values:
            print(f'ds_ifs ends later:\t{str(ds_ifs.time[-1].values)}')
            ds_ifs = ds_ifs.sel(time=slice(str(ds_ifs.time[0].values), str(ds_cycmask.time[-1].values)))
        else:
            raise ValueError('Time-horizonts of ds_cycmask and ds_ifs couldnt be resolved')
    

    assert str(ds_cycmask.time[0].values) == str(ds_ifs.time[0].values)
    assert str(ds_cycmask.time[-1].values) == str(ds_ifs.time[-1].values)
    print('Data Loaded\n')

    return ds_ifs, ds_cycmask




#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





def cyclone_mask_train_validation_slpit(ds_cycmask, path):
    ## Set a random Seed 
    random.seed(42)
    
    ## Get a list with all cyclone IDs
    cyc_id_list = set(ds_cycmask.FLAG.values.flatten())
    id_list = list(cyc_id_list)
    ## Get number of cyclones that go into validation set
    val_length = round(len(id_list) * 0.1)    
    ## Randomly select 10% of entries from the list without replacement
    val_ids = random.sample(id_list, val_length)
    print(f'Randomlly (10%) selected validation IDs are: {val_ids}')
    
    #-----------------------------------------------------------------------------------
    ## Write information about cyclones in txt file that is store in same directroy
    def save_txt(text, file):
        file.write(text + '\n')
    # Specify the path of the output file
    output_file_path = f'{path}/cyclone_split_info.txt'
    with open(output_file_path, 'w') as file:
        save_txt(f'Number of cyclones: {len(id_list)}', file)
        save_txt(f'ID list: {id_list}', file)
        save_txt(f'ID of cyclones in validation set: {val_ids}', file)
    #-----------------------------------------------------------------------------------
    
    print('Create cyclone mask (train and validation)')
    ## Split trainign and validation masks based on val_ids
    def split_train_val_cyclones(mask_field, validation_IDs):
        mask_train, mask_val = mask_field.copy(), ds_cycmask.FLAG.copy()
        # Create a boolean mask where True indicates the value is in val_ids
        val_mask = xr.apply_ufunc(np.isin, mask_field, validation_IDs)
        # For mask_train, replace all instances where val_mask is True with 0
        mask_train = mask_train.where(~val_mask, 0)
        # For mask_val, replace all instances where val_mask is False with 0
        mask_val = mask_val.where(val_mask, 0)
        return mask_train, mask_val
    
    FLmask_train, FLmask_val = split_train_val_cyclones(ds_cycmask['FLAG'], val_ids)
    CFmask_train, CFmask_val = split_train_val_cyclones(ds_cycmask['CFRONTS'], val_ids)
    WFmask_train, WFmask_val = split_train_val_cyclones(ds_cycmask['WFRONTS'], val_ids)
    BBmask_train, BBmask_val = split_train_val_cyclones(ds_cycmask['BBFRONTS'], val_ids)

    # Combine all train masks
    cyclone_mask_train = FLmask_train + CFmask_train + WFmask_train + BBmask_train
    cyclone_mask_train = cyclone_mask_train.where(cyclone_mask_train == 0, 1)
    # Combine all validation masks
    cyclone_mask_val = FLmask_val + CFmask_val + WFmask_val + BBmask_val
    cyclone_mask_val = cyclone_mask_val.where(cyclone_mask_val == 0, 1)
    
    print('Save train and validation cyclone mask.\n')
    ## Safe the netcdf files in the data_directory under <MONTH> val_set and train_set
    ds_cyclone_mask_train = cyclone_mask_train.to_dataset(name='cyclone_mask')
    ds_cyclone_mask_val = cyclone_mask_val.to_dataset(name='cyclone_mask')
    ds_cyclone_mask_train.to_netcdf(f'{path}/cyclone_mask_train.nc')
    ds_cyclone_mask_val.to_netcdf(f'{path}/cyclone_mask_validation.nc')
    #--------------------------------------------------------------------------------
    return cyclone_mask_train, cyclone_mask_val





#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------







def apply_mask_to_ifs(ds_ifs, mask):
    # Create two empty dictionaries to store the 4D and 3D variables
    data_vars_4d = {}
    data_vars_3d = {}

    # Loop over all data variables in the dataset
    for var_name, da in ds_ifs.data_vars.items():
        # Check if the variable has a 'lev' dimension
        if 'lev' in da.dims:
            # Add the variable to the 4D dictionary
            data_vars_4d[var_name] = da
        else:
            # Add the variable to the 3D dictionary
            data_vars_3d[var_name] = da

    # Create datasets from the dictionaries
    ds_4d = xr.Dataset(data_vars_4d)
    ds_3d = xr.Dataset(data_vars_3d)

    # Expand and apply the mask to the 4D dataset
    nlevels = ds_4d.dims['lev']  # get number of levels from 4D dataset
    mask4d = mask.expand_dims(lev=nlevels).transpose(*ds_4d.dims.keys())
    ds_4d_masked = ds_4d.where(mask4d)

    # Apply the mask to the 3D dataset
    ds_3d_masked = ds_3d.where(mask)
    # Combine the masked datasets back together
    ds_ifs_masked = xr.merge([ds_4d_masked, ds_3d_masked])
    return ds_ifs_masked



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def PRES_3d_new(data_set_PS):
    """
    This function creates a 3d pressure filed based on surface pressure (model-level 0).
    """   
    # Ensure aklay and bklay are Dask arrays
    aklay = np.array([0, 0.01878906, 0.1329688, 0.4280859, 0.924414, 1.62293, 2.524805, 3.634453, 4.962383, 6.515274, 8.3075, 10.34879, 12.65398, 15.23512,    18.10488, 21.27871, 24.76691, 28.58203, 32.7325, 37.22598, 42.06668,    47.25586, 52.7909, 58.66457, 64.86477, 71.37383, 78.16859, 85.21914,    92.48985, 99.93845, 107.5174, 115.1732, 122.848, 130.4801, 138.0055,    145.3589, 152.4757, 159.2937, 165.7537, 171.8026, 177.3938, 182.4832,    187.0358, 191.0384, 194.494, 197.413, 199.8055, 201.683, 203.0566,    203.9377, 204.339, 204.2719, 203.7509, 202.7876, 201.398, 199.5966,    197.3972, 194.8178, 191.874, 188.585, 184.9708, 181.0503, 176.8462,    172.382, 167.6805, 162.7672, 157.6719, 152.4194, 147.0388, 141.5674,    136.03, 130.4577, 124.8921, 119.3581, 113.8837, 108.5065, 103.253,    98.1433, 93.19541, 88.42463, 83.83939, 79.43383, 75.1964])
    bklay = np.array([0.9988151, 0.9963163, 0.9934933, 0.9902418, 0.9865207, 0.9823067,    0.977575, 0.9722959, 0.9664326, 0.9599506, 0.9528069, 0.944962,    0.9363701, 0.9269882, 0.9167719, 0.9056743, 0.893654, 0.8806684,    0.8666805, 0.8516564, 0.8355686, 0.8183961, 0.8001264, 0.7807572,    0.7602971, 0.7387676, 0.7162039, 0.692656, 0.6681895, 0.6428859,    0.6168419, 0.5901701, 0.5629966, 0.5354602, 0.5077097, 0.4799018,    0.4521973, 0.424758, 0.3977441, 0.3713087, 0.3455966, 0.3207688,    0.2969762, 0.274298, 0.2527429, 0.2322884, 0.212912, 0.1945903,    0.1772999, 0.1610177, 0.145719, 0.1313805, 0.1179764, 0.1054832,    0.0938737, 0.08312202, 0.07320328, 0.06408833, 0.05575071, 0.04816049,    0.04128718, 0.03510125, 0.02956981, 0.02465918, 0.02033665, 0.01656704,    0.01331083, 0.01053374, 0.008197418, 0.006255596, 0.004674384,    0.003414039, 0.002424481, 0.001672322, 0.001121252, 0.0007256266,    0.0004509675, 0.0002694785, 0.0001552459, 8.541815e-05, 4.1635e-05,   1.555435e-05, 3.39945e-06])
    
    interp_pres = []
    cur_lev=0
    for i in range(len(aklay)):
        pres_slice = data_set_PS.PS * bklay[i] + aklay[i]
        interp_pres.append(pres_slice)
        cur_lev+=1
        if cur_lev%40==0:
            print(f'Computed {cur_lev} levels from total {len(aklay)} levels')

    interp_pres = xr.concat(interp_pres, dim='lev')
        
    return interp_pres.transpose(*data_set_PS.dims.keys())



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def process_dataset_to_dataframe(ds, calc_rh=False):
    """
    This function processes a Dataset by flattening each 4-dimensional 
    DataArray and converting the Dataset to a DataFrame. All rows with NaN 
    values are dropped.

    Parameters:
    -----------
    ds : xarray.Dataset
        The input Dataset.

    Returns:
    --------
    pandas.DataFrame
        The processed DataFrame.
    """

    df_dict = {}
    
    # Iterate over each DataArray in the Dataset
    for var in ds.data_vars:
        # Check if the DataArray is 4D
        if ds[var].ndim == 4:
            print(f'Flatten {var}')
            # Flatten the DataArray and add to dictionary
            df_dict[var] = ds[var].values.flatten()
        else:
            print(f'Not flatten {var}, because it is a surface field')

    print('Form pd.DatFrame')
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(df_dict)

    shape_of_df = df.shape
    print(f'Shape of df before dropnan: {shape_of_df[0]:4.4e}')
    # Drop NaN values
    df = df.dropna()

    return df, shape_of_df







#################################################################################  Function that combines all in one and parallelizes: #####################################################################################

def get_train_validation_data_IFS18(month, calc_rh=False):
    
    path = f'/net/helium/atmosdyn/freimax/data_msc/IFS-18/cyclones/data_random_forest/{month}'
    # Check if the directory exists
    if not os.path.exists(path):
        # If not, create the directory
        os.makedirs(path)
    
    ds_ifs, ds_cycmask = load_data(month)

    ## Split train and validation cyclones
    cyc_mask_train, cyc_mask_val = cyclone_mask_train_validation_slpit(ds_cycmask, path)

    print(f'Used volume of ds_ifs:\t{(ds_ifs.nbytes / 1e9):.2f} GB\n')


    #=========================================== TRAIN DATA ===========================================================#

    ## Apply the cyclone Mask to the IFS data
    print('Apply cyclone mask to IFS train data')
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        ds_ifs_masked_train = apply_mask_to_ifs(ds_ifs=ds_ifs, mask=cyc_mask_train)

    #-------------------------------------------------------------------------






    if calc_rh:
        # Calculate a 4d pressure filed
        print('Get 4d pressure field for RH_ifs calcualtion')
        pres_4d_train = PRES_3d_new(ds_ifs_masked_train)

        #-------------------------------------------------------------------------
        # Set up a Dask LocalCluster and Client
        local_directory = tempfile.mkdtemp()
        cluster = LocalCluster(local_directory=local_directory, n_workers=24, threads_per_worker=2, memory_limit='96GB')
        client = Client(cluster)

        print('Calculate RH_ifs for train data')
        ds_ifs_masked_train['RH_ifs'] = calculate_rh_ifs(pres_4d_train, ds_ifs_masked_train.Q, ds_ifs_masked_train.T)

        # Close the Dask Client and LocalCluster when finished
        client.close()
        cluster.close()
        shutil.rmtree(local_directory) # Remove the temporary directory when done
        #-------------------------------------------------------------------------

        print('Drop varaibles (Q and PS (surface pressure)) that are superfluous')
        ds_ifs_masked_train = ds_ifs_masked_train.drop_vars(['Q','PS'])


    else:
        print('RH_ifs is not calculated, but pressure at surface is dropped')
        ds_ifs_masked_train = ds_ifs_masked_train.drop_vars(['PS'])
    #-------------------------------------------------------------------------


    # Set up a Dask LocalCluster and Client
    local_directory = tempfile.mkdtemp()
    cluster = LocalCluster(local_directory=local_directory, n_workers=24, threads_per_worker=2, memory_limit='96GB')
    client = Client(cluster)
    
    print('Flatten dataset and convert to pd.Dataframe')
    df_ifs_masked_train, df_train_shape_before_dropnan = process_dataset_to_dataframe(ds_ifs_masked_train)

    print('Safe the dataframe with the training data')
    df_ifs_masked_train.to_pickle(f'{path}/df_ifs_masked_train.pkl')

    # Close the Dask Client and LocalCluster when finished
    client.close()
    cluster.close()
    # Remove the temporary directory when done
    shutil.rmtree(local_directory)


    #=============================================== VALIDATION DATA ==================================================#

    ## Apply the cyclone Mask to the IFS data
    print('Apply cyclone mask to IFS validation data')
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        ds_ifs_masked_val = apply_mask_to_ifs(ds_ifs=ds_ifs, mask=cyc_mask_val)

    #-------------------------------------------------------------------------

    if calc_rh:
        # Calculate a 4d pressure filed
        print('Get 4d pressure field')
        pres_4d_val = PRES_3d_new(ds_ifs_masked_val)

        #-------------------------------------------------------------------------

        print('Calculate RH_ifs for train data')
        ds_ifs_masked_val['RH_ifs'] = calculate_rh_ifs(pres_4d_val, ds_ifs_masked_val.Q, ds_ifs_masked_val.T)

        # Set up a Dask LocalCluster and Client
        local_directory = tempfile.mkdtemp()
        cluster = LocalCluster(local_directory=local_directory, n_workers=24, threads_per_worker=2, memory_limit='96GB')
        client = Client(cluster)

        print('Calculate RH_ifs for train data')
        ds_ifs_masked_val['RH_ifs'] = calculate_rh_ifs(pres_4d_val, ds_ifs_masked_val.Q, ds_ifs_masked_val.T)
        
        # Close the Dask Client and LocalCluster when finished
        client.close()
        cluster.close()
        shutil.rmtree(local_directory) # Remove the temporary directory when done

        #-------------------------------------------------------------------------

        print('Drop varaibles (Q and PS (surface pressure)) that are superfluous')
        ds_ifs_masked_val = ds_ifs_masked_val.drop_vars(['Q', 'PS'])

    else:
        print('RH_ifs is not calculated, but pressure at surface is dropped')
        ds_ifs_masked_val = ds_ifs_masked_val.drop_vars(['PS'])
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------

    # Set up a Dask LocalCluster and Client
    local_directory = tempfile.mkdtemp()
    cluster = LocalCluster(local_directory=local_directory, n_workers=24, threads_per_worker=2, memory_limit='96GB')
    client = Client(cluster)
    print('Flatten dataset and convert to pd.Dataframe')
    df_ifs_masked_val, shape_before_dropnan = process_dataset_to_dataframe(ds_ifs_masked_val)

    print('Safe the dataframe with the validation data')
    df_ifs_masked_val.to_pickle(f'{path}/df_ifs_masked_val.pkl')

    # Close the Dask Client and LocalCluster when finished
    client.close()
    cluster.close()
    # Remove the temporary directory when done
    shutil.rmtree(local_directory)

    #-------------------------------------------------------------------------


    ## Return big dataset so it can be delted
    #return ds_ifs_masked_val, ds_ifs

###########################################################################################################################################################################################################################################








import gc
if __name__ == '__main__':
    # monthlist = ['DEC17', 'JAN18', 'MAR18', 'APR18', 'MAY18', 'JUN18', 'JUL18', 'AUG18', 'SEP18', 'OCT18', 'NOV18']     # FEB18 is missing and calculated in seperate ipynb script
    monthlist = ['AUG18', 'SEP18', 'OCT18', 'NOV18']     # FEB18 is missing and calculated in seperate ipynb script

    for month in monthlist:
        print(f'\n\tStart with {month}:\n\n')
        get_train_validation_data_IFS18(month, calc_rh=False)
        print('\n\tDONE\n')

        # Call the garbage collector to make sure we do not run out of RAM
        gc.collect()


    print('################################################################### FINISHED WITH SCRIPT ###################################################################')