import numpy as np
import xarray as xr
import sys
import os

sys.path.append('/home/freimax/msc_thesis/scripts/helpers/')
import data_preprocessing_helpers, helper


## Define functions to calculate e_sat, e and finally rh
def calc_e_sat(da_T):
    # Define Temperatures boundaries
    T_0   = 273.16
    T_ice = 250.16
    # alpha values (1 -> warm, 0 -> cold and mix-term for mix-temperature) 
    alpha_inter = ((da_T - T_ice) / (T_0-T_ice))**2
    alpha = xr.where(da_T > T_0, 1, xr.where(da_T < T_ice, 0, alpha_inter ))
    a1 = 611.21 # Pa
    a3w = 17.502
    a3i = 22.587
    a4w = 32.19 # K
    a4i = -0.7  # K
    # Tetens formula for saturation vapor pressure (c.f. IFS Cy43r1 p.115)
    e_sat_w = a1 * np.exp(a3w*( (da_T-T_0) / (da_T-a4w) ))
    e_sat_i = a1 * np.exp(a3i*( (da_T-T_0) / (da_T-a4i) ))
    # Calc total vapor pressure using eq. 7.92 (c.f. IFS Cy43r1 p.115)
    e_sat = alpha * e_sat_w + (1-alpha)*e_sat_i
    return e_sat

def calc_vap_pres(ds):
    ## Get 3d pressure fields
    pres_3d = data_preprocessing_helpers.PRES_3d(ds, shape_var=ds['Q'])
    ## Calculate vapor pressure
    epsilon_inverse = 1/0.621981      # Molar mass ration of dry air and vapor
    vap_pres = pres_3d * ds['Q'] * epsilon_inverse  / (1 + ds['Q']*(epsilon_inverse-1)  )
    ## Return RH in %: e / e_sat
    return vap_pres



if __name__ == '__main__':
    ## Define data path
    inpath='/net/thermo/atmosdyn2/atroman/PACaccu/cdf'
    outpath='/net/helium/atmosdyn/freimax/data_msc/IFS-17/cdf'

    # Create a list with all dates 48 hours before max intensity
    date_maxintensity = '20170410_17'
    filelist = [helper.change_date_by_hours(date_maxintensity, i) for i in range (-48,1)] # List is in chronological order



    filename = filelist[-1]
    for filename in filelist:
        print(f'Process file:\t{filename}')
        ## Load dataset
        ds_p = xr.load_dataset(f'{inpath}/P{filename}')
        ds_p['T'] += 273.16     # Convert temperature to Kelvin
        ds_p['PS'] *= 100       # Convert hPa to Pa 
        ds_p = ds_p.squeeze()

        ## Apply functions to calculate rh
        da_e_sat = calc_e_sat(ds_p['T'])
        da_e = calc_vap_pres(ds_p, da_e_sat)
        da_rh = da_e / da_e_sat * 100

        ## Only retain necessary variables and store dataset
        ds_pnew = ds_p.drop_vars(['CC','ttot','tdyn','tsw','tlw','tmix','tconv','tls','tcond','tdep','tbf','tevc','tfrz','trime','udot','vdot','tce'])
        ds_pnew['RH'] = da_rh.assign_coords(ds_p.coords)

        ## Save file
        ds_pnew.to_netcdf(f'{outpath}/P{filename}')
