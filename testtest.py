
import pandas as pd
import numpy as np
import joblib
import sys


sys.path.append('/home/ascherrmann/scripts/')
import helper





sys.path.append('/home/freimax/msc_thesis/scripts/IFS-18/')
import traj_processing_functions


sys.path.append('/home/freimax/msc_thesis/scripts/helpers/')
import helper




def load_rf_models():
    print(' =========================================================')
    print('                   *** Load Models ***                    ')
    ## Load models:
    model_tmeltsi = joblib.load("/net/helium/atmosdyn/freimax/data_msc/IFS-18/rf_models/tmeltsi/full_data_girdsearch_tmeltsi_f1.joblib")
    print('  Model_tmeltsi loaded')
    model_name = 'rf_fulldata_gridsearch_f1'
    filepath = f"/net/helium/atmosdyn/freimax/data_msc/IFS-18/rf_models/tsubsi/{model_name}.joblib"
    model_tsubsi = joblib.load(filepath)
    print('  Model_tsubsi loaded')
    print(' =========================================================')
    return model_tmeltsi, model_tsubsi
model_tmeltsi, model_tsubsi = load_rf_models()


df_traj_debug, df_big_debug = traj_processing_functions.load_trajs_predict_bcp(traj_file_path=f'/net/helium/atmosdyn/freimax/data_msc/IFS-17/traj-20170410_16-ID-000019.txt', 
                                                                               model_tmeltsi=model_tmeltsi,
                                                                               model_tsubsi=model_tsubsi,
                                                                               folder_branch_list=None, era5_data=False, debug=False)