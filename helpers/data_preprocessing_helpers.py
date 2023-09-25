#!/usr/bin/env python


import numpy as np
import wrf
import xarray as xr
from dypy.small_tools import interpolate
import metpy.calc as mpcalc
from metpy.interpolate import cross_section
from tqdm import tqdm #
#--------------------#
##    FUNCTIONS:    ##
"""
The functions in this script are:

    - get_crosec_data:              Load the necessary datasets and prepare the cross-section for them given a date and start, end point 
                                    (Further merge certain datavariables if encessary)

    - get_lon_latbox:               Get a subset of data (allows for taking subset over dateline)
    
    - calc_RH_w:                    Calculate the Relative Humidity and output fields same shape as input field (Temperatue, specific hum. Q and pressure p)

    - calc_RH_i:


    - calc_RH_ifs:                  Calculates RH that represents RH over ice and liquid for whole temperature range


    
    - inter2level:                  Intepolates pressure of one pressure level on specified pressure-level. Input field of desired variable and the pressure grid parr3D 

    - PRES_3d:                      Calculate 3D pressure field (on every model level) based on surface pressure                                                                (4D if time is included)

    - interpolate_pres:             Interpolate 3D pressure -> create pressure coordinates                                                                                      (4D if time is included)

    - get_cloudbase_pres:           Calculate the cloud base pressure in a given xarray Dataset with pressure levels and relative humidity.

    """
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def get_cross_section_data(data, start, end, timestep, era5_data=False):

    if 'time' in data.dims:
        data_p = data.sel(time=timestep).metpy.parse_cf()
    else:
        data_p = data.metpy.parse_cf()

    if data_p['RH'].max() < 80:
        print('Change unit of RH to % (*100)')
        data_p['RH'] = (data_p['RH']*100)
    
    
    ## Get cross section and convert lat/lon to supplementary coordinates
    cross_p = cross_section(data_p, start, end)
    # Get tangential and normal wind-field to cross-section
    cross_p['t_wind'], cross_p['n_wind'] = mpcalc.cross_section_components(cross_p['U'], cross_p['V'])
    
    if era5_data:

        # Create a dictonary to return
        data_dict = dict(data_p = data_p,
                        cross_p = cross_p)
        return data_dict 

    #---------------------------------------------------------------------------------------------------------------
    else:
        ## Get vertical sum of bc-porcesses (for inset plot)
        if 'tsubsi' in data_p:
            tbcp_all = data_p['tevr'] + data_p['tsubsi'] + data_p['tmeltsi']
        else:
            tbcp_all = data_p['tevr'] + data_p['tsubs'] + data_p['tsubi'] + data_p['tmelti'] + data_p['tmelts']

        try:
            tbcp_all = tbcp_all.sum(dim='level').compute()
        except:
            tbcp_all = tbcp_all.sum(dim='lev').compute()

        # Create a dictonary to return
        data_dict = dict(data_p = data_p,
                        cross_p = cross_p,
                        tbcp_all = tbcp_all)
        return data_dict 

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




def calc_RH_w(Q, T, p, temp_unit='celsius'): #Q,T,p can be 3D fields and T is provided in K, I think p should be in Pa rather than hPa, but find out by trying both
    if temp_unit == 'celsius':
        T_k = T + 273.16
        return 100 * 0.263 * p * Q / (np.exp(17.67 * (T_k-273.16)/(T_k-29.65)))
    elif temp_unit == 'kelvin':
        return 0.263 * p * Q / (np.exp(17.67 * (T-273.16)/(T-29.65)))

## Calculate RH_i in vectorized form

def calculate_e_sat(T, phase):
    a1 = 611.21  #Pa
    T0 = 273.16  #K
    # Assign differetn parameter depending on surface of ice or water
    if phase == 'w':
        a3 = 17.502
        a4 = 32.19  #K
    elif phase == 'i':
        a3 = 22.587
        a4 = -0.7 #K
    else:
        raise ValueError("Invalid input: phase must be either 'i' or 'w'")
    ## Calculate the saturation vapor pressure over ice or water
    return a1 * np.exp( a3 * ( (T-T0) / (T-a4) ))


def calculate_e_sat_T(T):
    # Define threhsold values of temperrture
    T_ice = 250.16
    T_0    = 273.16


    # Define the conditions
    cond_liq = T >= T_0
    cond_ice = T <= T_ice
    cond_mix = (T > T_ice) & (T < T_0)
    
    # Calculate intermediate alpha values
    alpha_intermediate = ((T - T_ice) / (T_0 - T_ice))**2
    # Define the values for each condition
    values_liq = np.ones_like(T)
    values_ice = np.zeros_like(T)
    values_mix = np.where(cond_mix, alpha_intermediate, 0)

    # Define Alpha by appling conditions and values using np.select
    alpha = np.select([cond_liq, cond_ice, cond_mix], [values_liq, values_ice, values_mix])
    
    e_sat_w = calculate_e_sat(T=T, phase='w') 
    e_sat_i = calculate_e_sat(T=T, phase='i')
    # Calculate e_sat(T) using the previously computed alpha
    return alpha * e_sat_w + (1-alpha) * e_sat_i
        

def calculate_rh_ifs(pres, q, T):
    """
    pres:       Pressure field (4d)  -> create using PRES_3d(data_set_PS, like_var4d)

    q:          Specific humidity field (4d)

    T:          Tempoerature fields (4d)
    
    """
    ## Change units
    print('Converting [hPa] to [Pa]')
    pres = pres * 100               # Convert hPa to Pa
    print('Converting [Celsius] to [Kelvin]')
    T_ndarray = T.values + 273.16   # Convert temperature to Kelvin

    # Get the Ratio of vapor / dry-air (molar masses)
    epsilon = 0.621981 

    # Calculate the vapor pressure
    print('Start Calculating Vapor pressure')
    vap_pres = pres * q * (1/epsilon) / (1 + q * (1/epsilon-1))
    
    print('Start Calculating Saturation Vapor pressure')
    # Calculate the saturation vapor pressure
    e_sat_T = calculate_e_sat_T(T_ndarray) 

    # Convert e_sat_T numpy array to a DataArray with matching dimensions to T
    e_sat_T_da = xr.DataArray(e_sat_T, coords=T.coords, dims=T.dims)

    print('Start Calculating RH')
    # Return the ration of vap_pres and sat_vap_pres (=RH)
    return vap_pres / e_sat_T_da

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_lonlatbox(ds, 
                  lon_range=(135,-165), 
                  lat_range=(20,70)):
    """
    Subset the xarray.DataSet ds to the given longitude and latitude range and shift longitudes if field extends over date-boundary.

    Args:
        ds (xarray.Dataset): The original dataset.
        
        lon_range (tuple): A tuple of two values specifying the minimum and maximum longitude values.
                           If the range crosses the dateline, the values should be given in degrees
                           east. For example, to select data between 150 degrees west and 170
                           degrees east, use (-210, 170).
        
        lat_range (tuple): A tuple of two values specifying the minimum and maximum latitude values,
                           both given in degrees north.
    
    Returns:
        xarray.Dataset: The subset of the original dataset within the given longitude and latitude
                        range.
    """
    # Check if the longitude range crosses the date boundary
    if lon_range[0] > lon_range[1]:
        # Concatenate the data across the date boundary
        ds_region = xr.concat([ds.sel(lon=slice(lon_range[0], 180)), ds.sel(lon=slice(-179.99, lon_range[1]))], dim='lon')
        ds_region['lon'] = np.mod(ds_region['lon'], 360)  # wrap longitudes to 0-360 degrees
    else:
        ds_region = ds.sel(lon=slice(lon_range[0], lon_range[1]))
    
    # Select the latitude range
    ds_region = ds_region.sel(lat=slice(lat_range[0], lat_range[1]))
    
    return ds_region


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Pressure field things:
#------------------------

## First: Get pressure on model level (one level or whole 4d cube):
def inter2level(varr3D, parr3D , plevel):
    """
    Interpolates 3-D (level, lat, lon) over level for variable array varr with
    associated pressure grid parr to the scalar pressure level plevel
    """ 
    v_i = interpolate(varr3D[::1,:, :], parr3D[:, :], plevel)
    return(v_i)



def PRES_3d(data_set_PS, shape_var):
    """
    This function creates a 3d pressure filed based on surface pressure (model-level 0).

    Variables:
    ----------

    data_set_ PS:    A xarray dataset that contains PS, as well as a varaibel that has the 
                    desired shape of the pressure field (i.e., either 3D or 4D if time is included)

    shape_var 4d:     This argument is the specific variable CONTAINED IN data_set_PS with the desired shape

    Output:
    -------
    A 3d or 4d data array (depending on the shape of like_var4d) with the pressure field on every model level.
    """
    
    aklay = np.array([0, 0.01878906, 0.1329688, 0.4280859, 0.924414, 1.62293, 2.524805, 3.634453, 4.962383, 6.515274, 8.3075, 10.34879, 12.65398, 15.23512,  \
                        18.10488, 21.27871, 24.76691, 28.58203, 32.7325, 37.22598, 42.06668, 47.25586, 52.7909, 58.66457, 64.86477, 71.37383, 78.16859, 85.21914,  \
                                92.48985, 99.93845, 107.5174, 115.1732, 122.848, 130.4801, 138.0055, 145.3589, 152.4757, 159.2937, 165.7537, 171.8026, 177.3938, 182.4832,  \
                                        187.0358, 191.0384, 194.494, 197.413, 199.8055, 201.683, 203.0566, 203.9377, 204.339, 204.2719, 203.7509, 202.7876, 201.398, 199.5966,  \
                                                197.3972, 194.8178, 191.874, 188.585, 184.9708, 181.0503, 176.8462, 172.382, 167.6805, 162.7672, 157.6719, 152.4194, 147.0388, 141.5674, \
                                                          136.03, 130.4577, 124.8921, 119.3581, 113.8837, 108.5065, 103.253, 98.1433, 93.19541, 88.42463, 83.83939, 79.43383, 75.1964 ])
    bklay = np.array([0.9988151, 0.9963163, 0.9934933, 0.9902418, 0.9865207, 0.9823067, 0.977575, 0.9722959, 0.9664326, 0.9599506, 0.9528069, 0.944962,  \
                        0.9363701, 0.9269882, 0.9167719, 0.9056743, 0.893654, 0.8806684, 0.8666805, 0.8516564, 0.8355686, 0.8183961, 0.8001264, 0.7807572,  \
                                0.7602971, 0.7387676, 0.7162039, 0.692656, 0.6681895, 0.6428859, 0.6168419, 0.5901701, 0.5629966, 0.5354602, 0.5077097, 0.4799018,  \
                                        0.4521973, 0.424758, 0.3977441, 0.3713087, 0.3455966, 0.3207688, 0.2969762, 0.274298, 0.2527429, 0.2322884, 0.212912, 0.1945903,  \
                                                0.1772999, 0.1610177, 0.145719, 0.1313805, 0.1179764, 0.1054832, 0.0938737, 0.08312202, 0.07320328, 0.06408833, 0.05575071, 0.04816049,  \
                                                        0.04128718, 0.03510125, 0.02956981, 0.02465918, 0.02033665, 0.01656704, 0.01331083, 0.01053374, 0.008197418, 0.006255596, 0.004674384,  \
                                                                0.003414039, 0.002424481, 0.001672322, 0.001121252, 0.0007256266, 0.0004509675, 0.0002694785, 0.0001552459, 8.541815e-05, 4.1635e-05, 1.555435e-05, 3.39945e-06])

    # Create new DataArray with appropriate dimensions and coordinates
    interp_pres = xr.DataArray(
        np.zeros(shape_var.shape), 
        dims=shape_var.dims, 
        coords=shape_var.coords
    )

    # Check if time dimension exists in shape_var
    if 'time' in shape_var.dims:
        # Time dimension exists, use 4D indexing
        for i in range(len(aklay)):
            interp_pres[:, i, :, :] = data_set_PS.PS[:, :, :] * bklay[i] + aklay[i]
    else:
        for i in range(len(aklay)):
            interp_pres[i, :, :] = data_set_PS.PS[:, :] * bklay[i] + aklay[i]

    return interp_pres


def PRES_3d_era5(data_set_PS, shape_var, hya,hyb):
    """
    This function creates a 3d pressure filed based on surface pressure (model-level 0).

    Variables:
    ----------

    data_set_ PS:    A xarray dataset that contains PS, as well as a varaibel that has the 
                    desired shape of the pressure field (i.e., either 3D or 4D if time is included)

    shape_var 4d:     This argument is the specific variable CONTAINED IN data_set_PS with the desired shape

    Output:
    -------
    A 3d or 4d data array (depending on the shape of like_var4d) with the pressure field on every model level.
    """


    hya_array = 0.01*hya#[np.arange(39,137)]
    hyb_array = hyb#[np.arange(39,137)]

    # Create new DataArray with appropriate dimensions and coordinates
    interp_pres = xr.DataArray(
        np.zeros(shape_var.shape), 
        dims=shape_var.dims, 
        coords=shape_var.coords)

    # Check if time dimension exists in shape_var
    if 'time' in shape_var.dims:
        # Time dimension exists, use 4D indexing
        for i in range(len(hya_array)):
            interp_pres[:, i, :, :] = data_set_PS.PS[:, :, :] * hyb_array[i] + hya_array[i]
    else:
        for i in range(len(hya_array)):
            interp_pres[i, :, :] = data_set_PS.PS[:, :] * hyb_array[i] + hya_array[i]

    return interp_pres
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def interpolate_pres(data, 
                     pres_field, 
                     pressure_levels=np.arange(1030,190,-10), 
                     varlist=None,
                     ):
    """
    Interpolates data variables from a given xarray Dataset or DataArray to specified pressure levels.
    
    The function supports 3D or 4D data, and it can handle cases with one or multiple timesteps.

    Parameters
    ----------
    data : xr.DataSet or xr.DataArray
        The input dataset or data array to interpolate. It can have 3D (with an empty time dimension)
        or 4D structure. For 3D data, the function will automatically handle it as a single time step.
        If a Dataset, the variables specified in `varlist` will be interpolated. If a DataArray, the
        entire array will be interpolated.

    pres_field : xr.DataArray
        A 3D (4D) pressure field variable corresponding to the pressure levels in the `data`. It has to have
        the same spatial and time dimensions as `data`.

    pressure_levels : array-like, optional
        The pressure levels to interpolate to. Default is an array ranging from 1030 to 190 with a step
        of -10.

    varlist : list of str, optional
        List of variable names to be interpolated. If None (default) and `data` is a Dataset, all variables
        in the Dataset will be interpolated.

    Returns
    -------
    ds_s_interp : xarray.Dataset or xarray.DataArray
        The interpolated dataset or data array, with the same structure as the input `data`, but with
        the variables interpolated to the specified pressure levels.

    Notes
    -----
    This function relies on the `wrf` library for the actual interpolation and requires that the input data
        be in a compatible format.
    
    """

    def interpolate_single_timestep(data_t, pres_3d_t):
        data_vars = {}
        for varname in varlist:
            if len(data_t[varname].shape) == 3:    # Ignore surface fields
                data_vars[varname] = wrf.interplevel(data_t[varname], pres_3d_t, pressure_levels)
        return xr.Dataset(data_vars) if varlist else xr.DataArray(data_vars)


    if varlist is None and isinstance(data, xr.Dataset):
        varlist = list(data.data_vars)

    # Check for empty time dimension
    if data['time'].shape == ():
        return interpolate_single_timestep(data, pres_field).squeeze()

    # Iterate over time, interpolating for each step
    data_arrays = []
    for t in tqdm(data.time, desc='Interpolating to pressure levels'): 
        data_t = data.sel(time=t)
        pres_3d_t = pres_field.sel(time=t)
        
        interpolated = interpolate_single_timestep(data_t, pres_3d_t)
        data_arrays.append(interpolated)

    return xr.concat(data_arrays, dim='time')


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"""

## DEPRECATED FUNCTION --> ONLY HERE IF OTHER FUNCTION FAILS

# def interpolate_pres(data, 
#                      pres_3d, 
#                      pressure_levels=np.arange(1030,190,-10), 
#                      varlist=None):
    

    ts, total_ts = 0, 0
# Create an empty list to store the interpolated DataArrays.
    data_arrays = []

    
    # Check if the input is a Dataset or DataArray
    if isinstance(data, xr.Dataset):
        print('data is instance xr.DataSet')
        if varlist == None:
            varlist = list(data.data_vars)
        print(f'Variable list: {varlist}')
        
        
        #### Add this part if there is only one timestep
        if data['time'].shape == ():
            print("Time dimension is empty.")
            data_vars = {}
            for varname in varlist:
                if (len(data[varname].shape) == 3):    ## Ignore surface fields (with only nx,ny and dt)
                    data_vars[varname] = wrf.interplevel(data[varname], pres_3d, pressure_levels)
            data_arrays.append(xr.Dataset(data_vars))       
            ds_s_interp = xr.concat(data_arrays, dim='time')                       #.transpose('lon', 'lat', 'pressure', 'time')
            return ds_s_interp.squeeze()



        else:
            # Loop over every timestep and interpolate the data variables to pressure levels.
            for t in data.time:
                # Select the data for this timestep.
                ds_t = data.sel(time=t)
                pres_3d_t = pres_3d.sel(time=t)
                # Interpolate the data variables to the pressure levels.
                
                data_vars = {}
                for varname in varlist:
                    if (len(ds_t[varname].shape) == 3):    ## Ignore surface fields (with only nx,ny and dt)
                        data_vars[varname] = wrf.interplevel(ds_t[varname], pres_3d_t, pressure_levels)
                data_arrays.append(xr.Dataset(data_vars))       
        
                # Logging
                ts += 1
                if ts == 1:
                    print('Start interpolating all variables to pressure-levels...')
                if ts % 6 == 0:
                    total_ts += 6
                    print(f'Done with {total_ts} timesteps out of {data.time.shape[0]+1}')


    elif isinstance(data, xr.DataArray):
        data = data.astype('float64')        
        print('data is instance xr.DataArray')
        for t in data.time:
            # Select the data for this timestep.
            da_t = data.sel(time=t)
            pres_3d_t = pres_3d.sel(time=t)
            data_vars = wrf.interplevel(da_t, pres_3d_t, pressure_levels)
            data_arrays.append(xr.DataArray(data_vars))       
            
            # Logging
            ts += 1
            if ts == 1:
                print('Start interpolating all variables to pressure-levels...')
            if ts % 6 == 0:
                total_ts += 6
                print(f'Done with {total_ts} timesteps out of {data.time.shape[0]+1}')

    else:
        raise TypeError("Input should be an xarray.Dataset or xarray.DataArray")
    
    
    #---------------------------------------------------------------------------------------------------
    # Combine the interpolated DataArrays into a single 4D dataset.
    ds_s_interp = xr.concat(data_arrays, dim='time')                       #.transpose('lon', 'lat', 'pressure', 'time')
    print('Function interpolate_pres DONE')
    return ds_s_interp

"""

























#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Next we generate new fileds:
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 
 # First the base of the clouds (in hPa)

def get_cloudbase_pres(ds,
                       rh_threshold = 95,
                       cc_threshold = None
                       )-> xr.Dataset:
    
    """
    Calculate the cloud base pressure in a given xarray Dataset with pressure levels and relative humidity.

    Parameters
    ----------
    ds : xarray.Dataset
        The input Dataset containing relative humidity (named 'RH') and pressure levels (named 'level').
        Dataset must have dimensions (time, level, lat, lon).
    rh_threshold : float, optional
        The threshold for relative humidity (%) above which the cloud base is defined.
        Default value is 90.

    Returns
    -------
    ds : xarray.Dataset
        The input Dataset with an additional variable 'cloud_base', which contains
        the cloud base pressure levels for each (time, lat, lon) grid point. If the cloud base
        is at the lowest pressure level, its value will be set to NaN.
    """

    # Find the levels where the relative humidity exceeds the cloud base threshold
    if rh_threshold is not None: 
        exceeds_threshold = ds['RH'] > rh_threshold

    if cc_threshold is not None:
        exceeds_threshold = ds['CC'] > cc_threshold

    # Find the first level that exceeds the threshold along the vertical coordinate (level)
    first_exceeds_threshold = exceeds_threshold.argmax(dim='level').compute()

    # Get the corresponding pressure levels
    cloud_base_pressure = ds['level'].isel(level=first_exceeds_threshold)


    # Check if the cloud base is predicted to be at the lowest pressure level (200 hPa)
    min_pres = ds['level'].min().compute()
    max_pres = ds['level'].max().compute()
    # Set the cloud base pressure to NaN if it is predicted to be at the lowest pressure level
    cloud_base_pressure_filtered = xr.where(cloud_base_pressure==min_pres, np.nan, cloud_base_pressure)
    cloud_base_pressure_filtered = xr.where(cloud_base_pressure_filtered==max_pres, np.nan, cloud_base_pressure_filtered)

    # Create a new DataArray with dimensions (time, lat, lon) to store the cloud base pressure levels
    cloud_base = xr.DataArray(
        cloud_base_pressure_filtered,
        coords={'time': ds['time'], 'lat': ds['lat'], 'lon': ds['lon']},
        dims=['time', 'lat', 'lon'],
        name='cloud_base'
    )

    # Add the cloud_base DataArray to the existing Dataset
    return cloud_base


def interpolate_lwc_to_cloud_base(ds, cloud_base_pressure):
    lwc = ds['LWC']
    cloud_base_lwc = lwc.interp(level=cloud_base_pressure)
    return cloud_base_lwc

def interpolate_cc_at_cloud_base(da_CC, cloud_base_pressure):
    cloud_base_cc = da_CC.interp(level=cloud_base_pressure)
    return cloud_base_cc
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------







