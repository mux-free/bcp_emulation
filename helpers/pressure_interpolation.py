#!/usr/bin/env python

if __name__=='__main__':
    ## Import modules
    import numpy as np 
    import pandas as pd
    import xarray as xr 
    import sys
    from numbers import Number
    sys.path.append('/home/freimax/msc_thesis/scripts/helpers/')
    from data_preprocessing_helpers import PRES_3d, calc_RH, get_lonlatbox, inter2level, interpolate_pres

    ## Load data
    file_path = '/net/thermo/atmosdyn2/atroman/PACaccu/cdf'

    # P-files:
    #--------------------------------------------------------------------------------------
    ds_p_modlev = xr.open_mfdataset(f'{file_path}/P*')
    ds_p_modlev = ds_p_modlev.squeeze('lev2')
    #--------------------------------------------------------------------------------------
    
    # TH-files:
    #--------------------------------------------------------------------------------------
    ds_s_modlev = xr.open_mfdataset(f'{file_path}/TH*')
    ## Change model timestep to datetime object
    time_array =[]
    base_time = pd.to_datetime('1950-01-01-00')
    for idx,time in enumerate( ds_s_modlev.time.values):
        time_array.append( base_time + pd.DateOffset(hours=int(time)))
    ##Change coordinates to lon/lat format
    ds_s_new=ds_s_modlev.rename({'dimx_PS': 'lon' , 'dimy_PS': 'lat' , 'dimz_TH': 'lev' , 'dimz_PS':'lev2' , 'time':'time' })
    ds_s_modlev=ds_s_new.assign_coords(lon=(ds_s_new.lon * 0.4 - 180), lat=(ds_s_new.lat * 0.4 - 90) , time=time_array)
    ds_s_modlev = ds_s_modlev.squeeze('lev2')
    #--------------------------------------------------------------------------------------


    #--------------------------------------------------------------------------------------
    ## Lonlatbox  -- apply to model-level data
    lonrange = [135,-165]
    latrange = [20,70]

    ds_s_modlev_lonlatbox = get_lonlatbox(ds_s_modlev, lon_range=lonrange, lat_range=latrange)
    ds_p_modlev_lonlatbox = get_lonlatbox(ds_p_modlev, lon_range=lonrange, lat_range=latrange)
    #--------------------------------------------------------------------------------------
    ################################################################
    # Apply function to interopolate variables onto pressure level #
    ################################################################
    # Define the pressure levels to interpolate to.
    pressure_levels = list(range(1030, 190, -10))

    # Specify all variables that should be interpolated from Primary or TH set
    varlist_s = list(ds_s_modlev_lonlatbox.data_vars)[1:-4]
    varlist_p = ['Q','RH','RWC','LWC','IWC','SWC','T','OMEGA','U','V',\
                'tcond','tdep','tbf','tevc','tsubi','tevr','tsubs','tmelti','tmelts',]



    def interpolate_lev_to_pres(ds,
                                pres_levels,
                                varlist,
                                path,
                                slp_data = ds_p_modlev_lonlatbox.SLP
                                ):
        # Create a 3d pressure file
        try:
            da_pres_p = PRES_3d(ds, 'Q')
            print(f'\nDimension of 3dPres_field (with Q): \n {da_pres_p.coords}')
        except:
            da_pres_p = PRES_3d(ds, 'PV')
            print(f'\nDimension of 3dPres_field (with PV): \n {da_pres_p.coords}')

        ## Add RH to the dataset
        ds = ds.assign(RH = calc_RH(Q=ds.Q, T=ds.T, p=PRES_3d(ds, 'T') ))
        assert 'RH' in ds.data_vars 
        print('RH added')
        
        # Interpolate the dataset and save the output to a netCDF file.
        ds_interpp = interpolate_pres(ds, da_pres_p , pres_levels, varlist)
        print(f'Dimension of function output: {ds_interpp.coords}')

        ds_interpp = ds_interpp.assign(SLP = slp_data)
        print('SLP added')

        ##############################################################################
        # Safe dataset
        valid_types = (str, Number, np.ndarray, np.number, list, tuple)
        try:
            ds_interpp.to_netcdf(path=path)
            # Fails with TypeError: Invalid value for attr: ...
        except TypeError as e:
            print(e.__class__.__name__, e)
            for variable in ds_interpp.variables.values():
                for k, v in variable.attrs.items():
                    if not isinstance(v, valid_types) or isinstance(v, bool):
                        variable.attrs[k] = str(v)

            ds_interpp.to_netcdf(path=path)  # Works as expected
            print(f'NetCDF saved in {path}')
        ###############################################################################
        return ds_interpp


    '''        
        ## Add RH to the dataset
        ds_p_modlev_lonlatbox = ds_p_modlev_lonlatbox.assign(RH = calc_RH(Q=ds_p_modlev_lonlatbox.Q, T=ds_p_modlev_lonlatbox.T, p=PRES_3d(ds_p_modlev_lonlatbox, 'T') ))
        assert 'RH' in ds_p_modlev_lonlatbox.data_vars 
        print('RH added')
    '''



    ## Convert files in two parts, to avoid OOM-killer
    path_p_p1 = '/home/freimax/msc_thesis/data/case_study_ra19/ifs_17/P_p1_lonlatbox.nc'
    #path_s_p1 = '/home/freimax/msc_thesis/data/case_study_ra19/ifs_17/S_p1_lonlatbox.nc'

    path_p_p2 = '/home/freimax/msc_thesis/data/case_study_ra19/ifs_17/P_p2_lonlatbox.nc'
    #path_s_p2 = '/home/freimax/msc_thesis/data/case_study_ra19/ifs_17/S_p2_lonlatbox.nc'
    
    path_p_p3 = '/home/freimax/msc_thesis/data/case_study_ra19/ifs_17/P_p3_lonlatbox.nc'
    #path_s_p3 = '/home/freimax/msc_thesis/data/case_study_ra19/ifs_17/S_p3_lonlatbox.nc'

    path_p_p4 = '/home/freimax/msc_thesis/data/case_study_ra19/ifs_17/P_p4_lonlatbox.nc'
    #path_s_p4 = '/home/freimax/msc_thesis/data/case_study_ra19/ifs_17/S_p4_lonlatbox.nc'
   

    ds_p_p1 = ds_p_modlev_lonlatbox.isel(time=slice(0,18))
    ds_p_p2 = ds_p_modlev_lonlatbox.isel(time=slice(18,36))
    ds_p_p3 = ds_p_modlev_lonlatbox.isel(time=slice(36,54))
    ds_p_p4 = ds_p_modlev_lonlatbox.isel(time=slice(54,-1))
    
    #ds_s_p1 = ds_s_modlev_lonlatbox.isel(time=slice(0,18))
    #ds_s_p2 = ds_s_modlev_lonlatbox.isel(time=slice(18,36))
    #ds_s_p3 = ds_s_modlev_lonlatbox.isel(time=slice(36,54))
    #ds_s_p4 = ds_s_modlev_lonlatbox.isel(time=slice(54,-1))


    # Interpolate fileds:
    #--------------------
    #ds_p_interpp1 = interpolate_lev_to_pres(ds=ds_p_p1, pres_levels=pressure_levels, varlist=varlist_p, path=path_p_p1)
    #ds_p_interpp2 = interpolate_lev_to_pres(ds=ds_p_p2, pres_levels=pressure_levels, varlist=varlist_p, path=path_p_p2)
    #ds_p_interpp3 = interpolate_lev_to_pres(ds=ds_p_p3, pres_levels=pressure_levels, varlist=varlist_p, path=path_p_p3)
    #ds_p_interpp4 = interpolate_lev_to_pres(ds=ds_p_p4, pres_levels=pressure_levels, varlist=varlist_p, path=path_p_p4)

    #ds_s_interpp1 = interpolate_lev_to_pres(ds=ds_s_p1, pres_levels=pressure_levels, varlist=varlist_s, path=path_s_p1)
    #ds_s_interpp2 = interpolate_lev_to_pres(ds=ds_s_p2, pres_levels=pressure_levels, varlist=varlist_s, path=path_s_p2)
    #ds_s_interpp3 = interpolate_lev_to_pres(ds=ds_s_p3, pres_levels=pressure_levels, varlist=varlist_s, path=path_s_p3)
    #ds_s_interpp4 = interpolate_lev_to_pres(ds=ds_s_p4, pres_levels=pressure_levels, varlist=varlist_s, path=path_s_p4)

    path_p_p1 = '/home/freimax/msc_thesis/data/case_study_ra19/ifs_17/additional_fields/CC_p1_lonlatbox.nc'
    ds_p_CC_interp1 = interpolate_lev_to_pres(ds=ds_p_p1, pres_levels=pressure_levels, varlist=['CC'], path=path_p_p1)
    
    path_p_p2 = '/home/freimax/msc_thesis/data/case_study_ra19/ifs_17/additional_fields/CC_p2_lonlatbox.nc'
    ds_p_CC_interp2 = interpolate_lev_to_pres(ds=ds_p_p2, pres_levels=pressure_levels, varlist=['CC'], path=path_p_p2)

    path_p_p3 = '/home/freimax/msc_thesis/data/case_study_ra19/ifs_17/additional_fields/CC_p3_lonlatbox.nc'
    ds_p_CC_interp3 = interpolate_lev_to_pres(ds=ds_p_p3, pres_levels=pressure_levels, varlist=['CC'], path=path_p_p3)

    path_p_p4 = '/home/freimax/msc_thesis/data/case_study_ra19/ifs_17/additional_fields/CC_p4_lonlatbox.nc'
    ds_p_CC_interp4 = interpolate_lev_to_pres(ds=ds_p_p4, pres_levels=pressure_levels, varlist=['CC'], path=path_p_p4)



