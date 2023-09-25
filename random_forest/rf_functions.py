## Load libaries
import pickle
import xarray as xr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from functools import reduce
import operator   



#=========================================================================================================================================================================================================================
#=========================================================================================================================================================================================================================
#=========================================================================================================================================================================================================================
#=========================================================================================================================================================================================================================
#======================================     Functions for retrieving labels (ground truth and baseline)     ===================================================#
## functions

def get_y_true(data_array, thresholds={'weak':-0.075,'medium':-0.75}):
    """
    This function categorises the bcp-cooling tendency array based on 2 thesholds and outputs an xr.Data.Array with the labels 0, 1 and 2.
    
    0 stands for no (or too weak) bcp-cooling occuring,
    1 medium bcp cooling occurs,
    2 strong bcp cooling occurs.

    Args:
        input_data (xr.Data.Array): The data array with the corresponding cooling tendency (either tevr, tsubsi or tmeltsi).
        thresholds (dict, optional): This dictonary contains the two thresholds for setting the labels. Defaults to {'weak':-0.075,'medium':-0.75}'.

    Returns:
        xr.DataArrat: DataArray with 0, 1 and 2 in the same shape as the input array.

    """    
    occur = (data_array <= thresholds['weak']) & (data_array > thresholds['medium'])
    strong_occur = data_array <= thresholds['medium']
    
    # Create new xr.DataArray where label 1 is given when tbcp exceeds weak threshold & 2 if medium threshold
    data_array_labeled = data_array.where(~occur, 1)                    # Occur
    data_array_labeled = data_array_labeled.where(~strong_occur, 2)     # Strong Occur
    # Set values that not satifsy the weak threshold to 0
    data_array_labeled = data_array_labeled.where(data_array <= thresholds['weak'], 0)
    
    return data_array_labeled


#==============================================================================================================================================================#
#==============================================================================================================================================================#





#==============================================================================================================================================================#
#==============================================================================================================================================================#


def make_predictions(ds, rf_model, water_type='SIWC', 
                     feature_names=['SIWC','LWC','RWC','RH','Q','OMEGA','T'], 
                     type_filteredvalues='nan', 
                     add_temp_filter=False,
                     verbose=1):
    
    if verbose>0:
        print('===================================================     Make RF Predictions     ===================================================')
        print('Create flattened features datframe')
    ds_features = ds[feature_names]
    # Flatten the features
    flat_features = {feature: ds_features[feature].values.flatten() for feature in feature_names}
    # Create a pandas DataFrame from the flattened features
    df_features = pd.DataFrame(flat_features)

    if add_temp_filter == True:
        ## Apply the domain filter and drop rows with NaN values
        df_features_filtered, domain_filter = apply_domainfilter(df=df_features, water_type=water_type, verbose=verbose, temp_filter=True, output_filter=True)
    else:
        ## Apply the domain filter and drop rows with NaN values
        df_features_filtered, domain_filter = apply_domainfilter(df=df_features, water_type=water_type, verbose=verbose, temp_filter=False, output_filter=True)

    #---------------------------------------------------------------------------------------------
    # print('Make predictions...')
    y_pred = rf_model.predict(df_features_filtered)
    #---------------------------------------------------------------------------------------------

    # Initialize the output array with NaN/0's
    output_shape = ds[water_type].shape
    if type_filteredvalues == 'nan':
        if verbose > 0:
            print(f'Reshape predictions to 4d data-array with shape {output_shape}.\t\tFiltered values are set to NaN')
        output_array = np.full(output_shape, np.nan)
    else:
        if verbose > 0:
            print(f'Reshape predictions to 4d data-array with shape {output_shape}.\t\tFiltered values are set to 0')
        output_array = np.full(output_shape, 0)

    # Flatten output_array and fill it with predicitons, where is is not NaN
    flat_output_array = output_array.flatten()
    flat_output_array[domain_filter] = y_pred

    # Reshape the flattened output array to the original 4D shape and convert it to a xr.DataArray
    reshaped_output_array = flat_output_array.reshape(output_shape)
    output_data = xr.DataArray(reshaped_output_array, coords=ds[water_type].coords, dims=ds[water_type].dims)    

    if verbose>0:
        print('===================================================================================================================================')
    return output_data















#=========================================================================================================================================================================================================================
#=========================================================================================================================================================================================================================
#=========================================================================================================================================================================================================================
#=========================================================================================================================================================================================================================









###        PREPARE DATA FOR TRAINING FO MODEL
def get_train_test_split(ds_p, 
                         bcp, 
                         save = True,
                         path = '/home/freimax/msc_thesis/data/case_study_ra19/rf_data', 
                         ):
    #------------------------------------------------------------------------------------------------------------------------------------------------------------
    ## Get flattened dataframe with all features and class-labels
    tsubsi_thresholds  = {'weak': -0.075, 'medium': -0.75}
    tmeltsi_thresholds = {'weak': -0.075, 'medium': -0.75}
    tevr_thresholds    = {'weak': -0.05, 'medium': -0.5}

    # Load and preprocess your data
    if bcp == 'tsubsi':
        print(f'\nCreate feature dataframe for tsubsi')
        features = preprocess_data(ds_p, 
                                   bcp = 'tsubsi' , 
                                   predictors = ['SIWC','RH_ifs','T','CC', 'W'], 
                                   bcp_thresholds = tsubsi_thresholds, )  
    elif bcp == 'tmeltsi':
        print(f'\nCreate feature dataframe for tmeltsi')
        features = preprocess_data(ds_p, 
                                   bcp='tmeltsi', 
                                   predictors = ['SIWC', 'RH_ifs', 'T', 'CC', 'W'],  
                                   bcp_thresholds = tmeltsi_thresholds)  
    elif bcp == 'tevr':
        print(f'\nCreate feature dataframe for tevr')
        features = preprocess_data(ds_p, 
                                   bcp='tevr', 
                                   predictors = ['RWC', 'RH_ifs', 'T', 'CC', 'W'], 
                                   bcp_thresholds=tevr_thresholds)  
    else:
        raise ValueError(f"Invalid variable name. Expected one of ['tsubsi', 'tmeltsi', 'tevr'], but got {bcp}.")
        #-----------------------------------------------------------------------------------------------------------------

    if bcp == 'tsubsi':
        print('\nApply domain filter to tsubsi (SIWC > 0)')
        data = apply_domainfilter(df=features, water_type='SIWC', verbose=2, temp_filter=False)

    elif bcp == 'tmeltsi':
        print('\nApply domain filter to tmeltsi (SIWC > 0 & Temp > 0')
        data = apply_domainfilter(df=features, water_type='SIWC', verbose=2, temp_filter=True)

    elif bcp == 'tevr':
        print('\nApply domain filter (RWC > 0)')
        data = apply_domainfilter(df=features, water_type='RWC', verbose=2, temp_filter=False)
    #----------------------------------------------------------------------------------------------------------------------

    # Split the data into training and testing sets for the tsubsi process
    X = data.drop('bcp_label', axis=1)
    y = data['bcp_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    ## Check training and testset
    print(f'Shape X_train: {X_train.shape}\t\tShape y_train: {y_train.shape}')
    print(f'Shape X_test:  {X_test.shape}\t\tShape y_test:  {y_test.shape}')
    assert X_train.shape[0] + X_test.shape[0] == data.shape[0]

    #-------------------------------------------------------------------------------------------------------------------------
    
    if save:
        ## Save dataframe
        print('\nSave not undersampled data\n')
        path = '/home/freimax/msc_thesis/data/case_study_ra19/rf_data'
        with open(f'{path}/X_train_{bcp}.pkl', 'wb') as f:
            pickle.dump(X_train, f)
        with open(f'{path}/y_train_{bcp}.pkl', 'wb') as f:
            pickle.dump(y_train, f)
        with open(f'{path}/X_test_{bcp}.pkl', 'wb') as f:
            pickle.dump(X_test, f)
        with open(f'{path}/y_test_{bcp}.pkl', 'wb') as f:
            pickle.dump(y_test, f)
    else:
        print('\nTrain-Test-split is reutrned in following order:    X_train,    X_test,    y_train,    y_test')
        return X_train, X_test, y_train, y_test












#--------------------------------------------------------------------------------------------------------------------------------------------
# FUCNTIONS USED FOR TRAIN_TEST_SPLIT
def preprocess_data(ds_p, 
                    bcp, 
                    predictors=['SIWC','RH_ifs','T','CC', 'W'], 
                    bcp_thresholds={'weak': -0.075, 'medium': -0.75}, 
                    ):
    """
    This functions takes the dataset with the bcp and all the features as input and converts them to a flat pandas.DataFrame,
    suitable for applying train/test-split

    Args:
        ds_p (xr.DataSet):                  DataSet containing al features and temperature tendency of itnerest.
        bcp (string):                       The temerpoature tendency (tsubsi, tevr or tmeltsi).
        predictors (list, optional):        List of featues used to predict bcp. Defaults to ['SIWC','RH_ifs','T','CC', 'W'].
        bcp_thresholds (dict, optional):    Bcp-thresholds for true-labels. Defaults to {'weak': -0.075, 'medium': -0.75}.

    Returns:
        pandas.Dataframe: Datframe with all features and true-labels.  
    """
    ## Extract the predictor variables
    #da_rwc = ds_p['RWC'].to_series().reset_index(drop=True)
    if 'RWC' in predictors:                                     # RWC
        da_rwc = ds_p['RWC'].to_series().values.flatten()
    if 'SIWC' in predictors:                                    # SIWC
        da_siwc = ds_p['SIWC'].values.flatten()
    if 'RH_ifs' in predictors or 'RH' in predictors:            # RH_ifs
        da_rh = ds_p['RH_ifs'].values.flatten()
    if 'T' in predictors or 'Temp' in predictors:               # Temp
        da_temp = ds_p['T'].values.flatten()
    if 'CC' in predictors:                                      # CloudCover
        da_CC = ds_p['CC'].values.flatten()
    if 'V' in predictors or 'V_hor' in predictors:              # V_hor
        da_V = np.sqrt(ds_p.U**2 + ds_p.V**2).values.flatten()  
    if 'W' in predictors or 'OMEGA' in predictors:              # W
        da_W = ds_p['OMEGA'].values.flatten()
    
    # Create labels
    #bcp_label = assign_label(ds_p[bcp], bcp_thresholds, nr_classes=nr_classes)
    bcp_label = get_y_true(ds_p[bcp], bcp_thresholds)
    # Convert the xarray DataArrays to pandas Series
    bcp_label_series = bcp_label.to_series().reset_index(drop=True)

    # Create a dictionary with all possible features
    all_features = {
        'RWC': da_rwc if 'RWC' in predictors else None,
        'SIWC': da_siwc if 'SIWC' in predictors else None,
        'RH_ifs': da_rh if 'RH_ifs' in predictors or 'RH' in predictors else None,
        'Temp': da_temp if 'T' in predictors or 'Temp' in predictors else None,
        'CC': da_CC if 'CC' in predictors else None,
        'W': da_W if 'W' in predictors or 'OMEGA' in predictors else None,
    }

    # Filter the selected features using dictionary comprehension
    selected_features = {k: v for k, v in all_features.items() if v is not None}
    # Create a pandas DataFrame from the Series
    df = pd.DataFrame({'bcp_label': bcp_label_series, **selected_features})

    return df










#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Apply domain filter
def apply_domainfilter(df, 
                       water_type = 'SIWC',
                       verbose=0,
                       temp_filter=False,
                       output_filter=False,
                       ):
    
    if temp_filter == False:
        ## Drop all cloudwater = 0 contents
        domain_filter = df[water_type] > 0
        # Apply the filter on every column
        if verbose > 0:
            print(f'Drop all points where {water_type} == 0')
        filtered_df = df[domain_filter].dropna()
        #filtered_df = df.apply(lambda x: x[domain_filter])
    elif temp_filter == True:
        ## Drop all cloudwater = 0 contents and where the Temp is below 0 degrees  (Sometimes the variable is T and sometimes Temp)
        try:
            domain_filter = (df[water_type] > 0) & (df['Temp'] > 0)
        except:
            domain_filter = (df[water_type] > 0) & (df['T'] > 0)
        # Apply the filter on every column
        if verbose > 0:
            print(f'Drop all points where {water_type} == 0 and Temp < 0')
        #filtered_df = df.apply(lambda x: x[domain_filter])
        filtered_df = df[domain_filter].dropna()


    if verbose >= 2:
        print('=================================================================================================')
        ## Print metrics of dropped dataframe
        len_old_df = df.shape[0]
        print(f'Filtered dataframe size:\t{filtered_df.shape[0]:3.2e} \t{filtered_df.shape[0]*100/len_old_df:.1f}% of size compared to unfiltered. (Points dropped: {len_old_df-filtered_df.shape[0]})')
        print('=================================================================================================')
        
        ## Print balance distribution of new datafram  FOR CLASSIFICATION DATSET
        if 'bcp_label' in df.columns:
            bcp_label = df['bcp_label'].values
        
            zero_frac = bcp_label[bcp_label == 0].shape[0] / bcp_label.shape[0] 
            print(f'Class 0 fraction (before drop):\t{zero_frac *100:.1f}%')     
            print('\nClass Balance New Dataframe:')
            print('------------------------------')
            class0_freq = filtered_df[filtered_df.bcp_label==0].shape[0] *100 / filtered_df.shape[0]
            print(f'Class 0 fraction:\t{class0_freq:2.1f}%\t\t{filtered_df[filtered_df.bcp_label==0].shape[0]:7.0f} Points')
            class1_freq = filtered_df[filtered_df.bcp_label==1].shape[0] *100 / filtered_df.shape[0]
            print(f'Class 1 fraction:\t{class1_freq:2.1f}%\t\t{filtered_df[filtered_df.bcp_label==1].shape[0]:7.0f} Points')
            class2_freq = filtered_df[filtered_df.bcp_label==2].shape[0] *100 / filtered_df.shape[0]
            print(f'Class 2 fraction:\t {class2_freq:2.1f}%\t\t{filtered_df[filtered_df.bcp_label==2].shape[0]:7.0f} Points')
            print('------------------------------\t\t--------------')
            print(f'Toatal points in new df:\t\t{filtered_df.shape[0]:7.0f} Points')
            print('=================================================================================================')

        ## FOR REGRESSION DATASET (--> NO CLASS LABELS)
        else:
            ## Retieve the bcp-name
            if 'tsubsi' in df.columns:
                bcp_name='tsubsi'
            elif 'tmeltsi' in df.columns:
                bcp_name='tmeltsi'
            elif 'tevr' in df.columns:
                bcp_name='tevr'
            else:
                raise ValueError('Below-cloud process not contained in Dataframe')

            print(f'\n\tNo categories for {bcp_name} available! Creating labels and stats now...\n')
            bcp_label_unfiltered = get_y_true(df[bcp_name])
            bcp_label = get_y_true(filtered_df[bcp_name])

            zero_frac = bcp_label_unfiltered[bcp_label_unfiltered == 0].shape[0] / bcp_label_unfiltered.shape[0] 
            print(f'Class 0 fraction (before drop):\t{zero_frac *100:.1f}%')
            print('\nClass Balance New Dataframe:')
            print('------------------------------')
            class0_freq = filtered_df[bcp_label==0].shape[0] *100 / filtered_df.shape[0]
            print(f'Class 0 fraction:\t{class0_freq:2.1f}%\t\t{filtered_df[bcp_label==0].shape[0]:7.0f} Points')
            class1_freq = filtered_df[bcp_label==1].shape[0] *100 / filtered_df.shape[0]
            print(f'Class 1 fraction:\t{class1_freq:2.1f}%\t\t{filtered_df[bcp_label==1].shape[0]:7.0f} Points')
            class2_freq = filtered_df[bcp_label==2].shape[0] *100 / filtered_df.shape[0]
            print(f'Class 2 fraction:\t {class2_freq:2.1f}%\t\t{filtered_df[bcp_label==2].shape[0]:7.0f} Points')
            print('------------------------------\t\t--------------')
            print(f'Toatal points in new df:\t\t{filtered_df.shape[0]:7.0f} Points')
            print('=================================================================================================\n')

    if output_filter == False:
        return filtered_df
    elif output_filter == True:
        # print('Ouput filtered_df and domain_filter')
        return filtered_df, domain_filter




















#=========================================================================================================================================================================================================================
#=========================================================================================================================================================================================================================
#=========================================================================================================================================================================================================================
#=========================================================================================================================================================================================================================
#             Addtional function:


def get_confmat_clasreport(y_true, y_pred, model=''):
    print(f'============================  {model}  ============================')
    print('Confusion matrix:')                      
    conf_mat = confusion_matrix(y_true, y_pred)
    print(conf_mat)
    print('\n\t------------------------------------------------------------')
    print('Classification Report:')
    clas_report = classification_report(y_true, y_pred)
    print(clas_report)
    print('==================================================================')
    # Return confusion matrix
    return conf_mat

#=========================================================================================================================================================================================================================
#=========================================================================================================================================================================================================================
#=========================================================================================================================================================================================================================
#=========================================================================================================================================================================================================================

## Function for functions
def z_score_scaling(df_col):
    print('Applying z-scaling...')
    # Calculate the L2 norm of the column
    col_mean = np.nanmean(df_col)
    col_std = np.nanstd(df_col)
    return (df_col - col_mean) / col_std




#=========================================================================================================================================================================================================================
#=========================================================================================================================================================================================================================




def get_y_baseline(input_data, bcp, thr_vars, thr_metrics = '50%'):
    """
    This function applies hand-picked thresholds (for 3 temperature regimes: warm, mix, cold) for selected featues (thr_vars). 
    The threshold are laoded from a dataset that is vlaid for the case-study cyclone (Attinger et al. 2019), they were calculated using a edge-detection mask and applying the threshold fields within that mask.

    Args:
        input_data (xr.Data.Set): DataSet with all variables of importance (temperature tendencies and features)
        bcp (str): Which below-cloud-process will be focused on.
        thr_vars (list of strings): List of all features that are used to predict bcp-occurence
        thr_metrics (str, optional): Description of the thr_metrics argument. Defaults to '50%' (= median).

    Returns:
        xr.DataArray: With labels 0 or 1, depending on rediciton of hand-picked thresholds and features. Same shape as input_daata.

    """

    #-------------------------------------------------------------------------------------
    ## Select bcp dataset and load Data
    if bcp == 'tsubsi':
        path = '/home/freimax/msc_thesis/data/case_study_ra19/edge_field_stats/tsubsi'
        df1, df2, df3 = 'df_dict_subsi_warm', 'df_dict_subsi_mix', 'df_dict_subsi_cold'
        # Define the color for the plot
        stip_color = 'red'

    elif bcp == 'tmeltsi':
        path = '/home/freimax/msc_thesis/data/case_study_ra19/edge_field_stats/tmeltsi'
        df1, df2, df3 = 'df_dict_meltsi_warm', 'df_dict_meltsi_mix', 'df_dict_meltsi_cold'
        # Define the color for the plot
        stip_color = 'blue'

    elif bcp == 'tevr':
        path = '/home/freimax/msc_thesis/data/case_study_ra19/edge_field_stats/tevr'
        df1, df2, df3 = 'df_dict_evr_warm', 'df_dict_evr_mix', 'df_dict_evr_cold'
        # Define the color for the plot
        stip_color = 'green'
    else:
        raise ValueError('bcp process mus tbe either: tsubsi, tmeltsi, tevr')
    
    # Open the list of dataframes
    with open(f'{path}/{df1}.pkl', 'rb') as f:
        df_warm = pickle.load(f)
    with open(f'{path}/{df2}.pkl', 'rb') as f:
        df_mix = pickle.load(f)
    with open(f'{path}/{df3}.pkl', 'rb') as f:
        df_cold = pickle.load(f)
    #--------------------------------------------------------------------------------------------------------------------------
    # Select variables and corresponding thresholds
    bcp_thr_value = 'thr = -0.1'
    
    thr_warm_RWC, thr_warm_SIWC, thr_warm_RH, thr_warm_T = df_warm[0].loc[thr_metrics, bcp_thr_value], df_warm[1].loc[thr_metrics, bcp_thr_value], df_warm[2].loc[thr_metrics, bcp_thr_value],  df_warm[3].loc[thr_metrics, bcp_thr_value]
    thr_mix_RWC , thr_mix_SIWC , thr_mix_RH, thr_mix_T   = df_mix[0].loc[thr_metrics, bcp_thr_value] , df_mix[1].loc[thr_metrics, bcp_thr_value] , df_mix[2].loc[thr_metrics, bcp_thr_value] ,  df_mix[3].loc[thr_metrics, bcp_thr_value] 
    thr_cold_RWC, thr_cold_SIWC, thr_cold_RH, thr_cold_T = df_cold[0].loc[thr_metrics, bcp_thr_value], df_cold[1].loc[thr_metrics, bcp_thr_value], df_cold[2].loc[thr_metrics, bcp_thr_value],  df_cold[3].loc[thr_metrics, bcp_thr_value]
    
    #-------------------------------------------------------------------------------------
    # Create masks for each temperature regime
    warm_mask = input_data.T >= 0
    mix_mask = (input_data.T > -23) & (input_data.T < 0)
    cold_mask = input_data.T <= -23
    #--------------------------------------------------------------------------------------------------------------------------
    # Define filter masks for each temperature regime and feature
    warmmask, mixmask, coldmask = [], [], []
    
    if 'RWC' in thr_vars:
        mask_rwc_warm  = input_data['RWC'] > thr_warm_RWC
        mask_rwc_mix   = input_data['RWC'] > thr_mix_RWC
        mask_rwc_cold  = input_data['RWC'] > thr_cold_RWC
        warmmask.append(mask_rwc_warm)
        mixmask.append(mask_rwc_mix)
        coldmask.append(mask_rwc_cold)

    if 'SIWC' in thr_vars:
        mask_siwc_warm = input_data['SIWC'] > thr_warm_SIWC
        mask_siwc_mix  = input_data['SIWC'] > thr_mix_SIWC
        mask_siwc_cold = input_data['SIWC'] > thr_cold_SIWC
        warmmask.append(mask_siwc_warm)
        mixmask.append(mask_siwc_mix)
        coldmask.append(mask_siwc_cold)
    
    if 'RH_ifs' in thr_vars:
        mask_rh_warm  = input_data['RH_ifs'] < thr_warm_RH
        mask_rh_mix   = input_data['RH_ifs'] < thr_mix_RH
        mask_rh_cold  = input_data['RH_ifs'] < thr_cold_RH
        warmmask.append(mask_rh_warm)
        mixmask.append(mask_rh_mix)
        coldmask.append(mask_rh_cold)   

    if 'T' in thr_vars:
        mask_T_warm = input_data['T'] > thr_warm_T
        mask_T_mix  = input_data['T'] > thr_mix_T
        mask_T_cold = input_data['T'] > thr_cold_T
        warmmask.append(mask_T_warm)
        mixmask.append(mask_T_mix)
        coldmask.append(mask_T_cold)   

 
    # Combine the masks using a piecewise AND operation
    mask_warm = reduce(operator.and_, warmmask) & warm_mask               ## First reduce list of filters with AND statement, then AND with temperature-mask
    mask_mix  = reduce(operator.and_, mixmask)  & mix_mask
    mask_cold = reduce(operator.and_, coldmask) & cold_mask    

    full_mask = mask_warm | mask_mix | mask_cold
    
    return full_mask.astype(int)













#=============================================================================================================================================================================================================
#=============================================================================================================================================================================================================
## Train a random forest
#=============================================================================================================================================================================================================
#=============================================================================================================================================================================================================

import itertools
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from joblib import dump

def grid_search_with_validation_set_f1(X_train, 
                                       y_train, 
                                       X_val, 
                                       y_val, 
                                       param_grid, 
                                       save_name, 
                                       threshold=-0.1):

    # Get variable of interest
    try:
        bcp_process = y_train.name
    except:
        bcp_process = input('Automation retrival failed...\nWhat it the target variable? (tevr, tmeltsi, tsubsi)')
    print(f'Target variable is: {bcp_process}')

    best_mse = np.inf
    best_f1 = -np.inf
    best_params_mse = None
    best_params_f1 = None
    best_model_mse = None
    best_model_f1 = None

    total_iter = 1
    for key in param_grid.keys():
        total_iter *= len(param_grid[key])
    current_i = 0
    print('\nStart Grid-Search:')
    # Loop over the parameter grid
    for params in itertools.product(*param_grid.values()):
        current_i += 1
        n_estimators, max_depth, max_features, min_samples_split, min_samples_leaf, criterion = params

        # Create and train a Random Forest model
        rf = RandomForestRegressor(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   max_features=max_features,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   criterion=criterion,
                                   n_jobs=60)
        rf.fit(X_train, y_train)

        # Predict the validation set results and compute the mean squared error
        y_pred = rf.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        f1 = f1_score((y_val >= threshold), (y_pred >= threshold), average='macro')

        precision = precision_score((y_val >= threshold), (y_pred >= threshold), average='macro')
        recall = recall_score((y_val >= threshold), (y_pred >= threshold), average='macro')
        
        if (mse < best_mse) and (f1 > best_f1):
            print(f'Model {current_i:3} of {total_iter:3} | n_estiM ={n_estimators:4} | max_depth = {max_depth} | max_features ={max_features:3} | min_samp_split ={min_samples_split:3}| min_samp_leaf ={min_samples_leaf:2} | crit = {criterion} | precision: {precision:.3f}  | recall: {recall:.3f} | F1: {f1:.4f}  |  RMSE: {np.sqrt(mse):.4f}  (NEW BEST F1 AND RMSE)')
        elif mse < best_mse:
            print(f'Model {current_i:3} of {total_iter:3} | n_estiM ={n_estimators:4} | max_depth = {max_depth} | max_features ={max_features:3} | min_samp_split ={min_samples_split:3}| min_samp_leaf ={min_samples_leaf:2} | crit = {criterion} | precision: {precision:.3f}  | recall: {recall:.3f} | F1: {f1:.4f}  |  RMSE: {np.sqrt(mse):.4f}  (NEW BEST RMSE)')
        elif f1 > best_f1:
            print(f'Model {current_i:3} of {total_iter:3} | n_estiM ={n_estimators:4} | max_depth = {max_depth} | max_features ={max_features:3} | min_samp_split ={min_samples_split:3}| min_samp_leaf ={min_samples_leaf:2} | crit = {criterion} | precision: {precision:.3f}  | recall: {recall:.3f} | F1: {f1:.4f}  |  RMSE: {np.sqrt(mse):.4f}  (NEW BEST F1)')
        
        elif current_i % 10 == 0:
            print(f'Current Model: {current_i} of of {total_iter:3}')
        
        
        # If this is the best model so far based on mse, save its parameters and score
        if mse < best_mse:
            best_mse = mse
            best_params_mse = {'n_estimators': n_estimators, 
                               'max_depth': max_depth, 
                               'max_features': max_features, 
                               'min_samples_split': min_samples_split,
                               'min_samples_leaf': min_samples_leaf,
                               'criterion': criterion}
            best_model_mse = rf

        
        # If this is the best model so far based on f1, save its parameters and score
        if f1 > best_f1:
            best_f1 = f1
            best_params_f1 = {'n_estimators': n_estimators, 
                              'max_depth': max_depth, 
                              'max_features': max_features, 
                              'min_samples_split': min_samples_split,
                              'min_samples_leaf': min_samples_leaf,
                              'criterion': criterion}
            best_model_f1 = rf

            
            

    print(f'\nBest parameters based on MSE: {best_params_mse}')
    print(f'Best MSE: {best_mse}  -->  Best RMSE: {np.sqrt(best_mse)}')
    print(f'\nBest parameters based on F1: {best_params_f1}')
    print(f'Best F1: {best_f1}')

    dump(best_model_mse, f'/net/helium/atmosdyn/freimax/data_msc/IFS-18/rf_models/{bcp_process}/{save_name}_mse.joblib')
    dump(best_model_f1,  f'/net/helium/atmosdyn/freimax/data_msc/IFS-18/rf_models/{bcp_process}/{save_name}_f1.joblib')

    return best_model_mse, best_params_mse, best_model_f1, best_params_f1

