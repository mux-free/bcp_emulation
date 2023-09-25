import pandas as pd
import joblib
import numpy as np
import xarray as xr
import seaborn as sns
import sys
from glob import glob 
import os

sys.path.append('/home/freimax/msc_thesis/scripts/IFS-18/')
import traj_processing_functions

sys.path.append('/home/freimax/msc_thesis/scripts/helpers/')
import helper


############################################
## Process trajectories from ERA5 dataset ##
############################################

model_tmeltsi, model_tsubsi, model_tevr = helper.load_rf_models()

region_list = ['ge-30-l-45', 'ge-45-l-60', 'ge-60-l-90']

save_path_NP  = '/net/helium/atmosdyn/freimax/data_msc/ERA5/trajs/climatology/NP'
ocean_basin   = 'NP'
df_big_NP  = traj_processing_functions.load_trajs_predict_bcp(
    model_tmeltsi=model_tmeltsi,
    model_tsubsi=model_tsubsi,
    model_tevr=model_tevr,
    folder_branch_list=region_list, 
    era5_data=True, 
    parent_branch=ocean_basin, 
    save_path=save_path_NP,
    traj_file_path=f'/net/thermo/atmosdyn2/ascherrmann/MA/Max')


ocean_basin = 'NA'
save_path_NA = '/net/helium/atmosdyn/freimax/data_msc/ERA5/trajs/climatology/NA'
df_big_NA  = traj_processing_functions.load_trajs_predict_bcp(
    model_tmeltsi=model_tmeltsi,
    model_tsubsi=model_tsubsi,
    model_tevr=model_tevr,
    folder_branch_list=region_list, 
    era5_data=True, 
    parent_branch=ocean_basin, 
    save_path=save_path_NA,
    traj_file_path=f'/net/thermo/atmosdyn2/ascherrmann/MA/Max')




