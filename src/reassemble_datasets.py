'''
Module: reassemble_datasets.py
Description: this module includes functions for building datasets relevant to the selected task
out of CESNET data structure
'''
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD

#=====================================================================================================#
def building_timestamps_dataset(autosave=True):
    proj_root = os.environ['PYTHONPATH']

    timestamps_df = pd.read_csv(os.path.join(proj_root, 'data', 'times', 'times_1_day.csv'))
    timestamps_df['day'] = pd.to_datetime(timestamps_df['time']).map(lambda x: x.date())
    timestamps_df.drop_duplicates(inplace=True)

    holidays_df = pd.read_csv(os.path.join(proj_root, 'data', 'weekends_and_holidays.csv'))
    holidays_df['day'] = pd.to_datetime(holidays_df['Date']).map(lambda x: x.date())
    holidays_df.drop_duplicates(inplace=True)

    ts_days_df = timestamps_df[['id_time', 'day']].merge(holidays_df[['Type', 'day']], on='day', how='left')
    #ts_days_df['week_no'] = ts_days_df['day'].map(lambda x: x.isocalendar().week)
    ts_days_df['day_of_week'] = ts_days_df['day'].map(lambda x: x.weekday())
    ts_days_df['is_working_day'] = ts_days_df['Type'].isna().map(int)
    ts_days_df.drop(columns=['Type'], inplace=True)

    if autosave:
        ts_days_df.to_csv(os.path.join(proj_root, 'preprocessed_datasets', 'ts_days_df.csv'), index=False)

    return ts_days_df


#=====================================================================================================#
def reassemble_predictors_datasets(drop_inst_ids_list: list, flds_selection: list, nof_comp=260, autosave=True
                                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    '''
    this function re-assembles the initial (not windowed) dataset for selectes params:
    - drop_inst_ids_list: list of institution IDs to drop (to manage autocorrelation)
    - flds_selection: list of fields to keep in ML predictors
    - nof_comp: number of components to use in PCA and SVD transformation (default: 260)
    - autosave: whether to automatically save the resulting DataFrames (default: True)

    Returns:
    - predictors: the reassembled predictors DataFrame
    - predictors_scaled: same DataFrame, but scaled using StandardScaler
    - X_pca: the transformed data using PCA (from unscaled predictors)
    - X_svd: the transformed data using SVD (from scaled predictors)
    - feature_names: list of feature names as predictors_df was assembled

    NB! the sources for assembling are the (preprocessed) CESNET data folders not put into patrams
    '''
    proj_root = os.environ['PYTHONPATH']
    
    inst_ids_list = sorted([ int(el.split('.')[0]) 
                         for el in os.listdir(os.path.join(proj_root,'data','institutions','agg_1_day')) 
                         if el.endswith('.csv') ])

    inst_ids_list = [el for el in inst_ids_list if el not in drop_inst_ids_list]

    predictors_df = pd.read_csv(os.path.join(proj_root,'preprocessed_datasets','ts_days_df.csv')
                                ).drop(columns=['day']) 

    for inst_id in inst_ids_list:

        tmp_df = pd.read_csv(os.path.join(proj_root,'data','institutions','agg_1_day', f'{inst_id}.csv')
                             )[flds_selection]
    
        for col in tmp_df.columns:
            if col != 'id_time':
                tmp_df[col] = tmp_df[col].astype(float)
            
        tmp_df.columns = [el + '_' + str(inst_id) for el in tmp_df.columns]
        tmp_df.rename(columns={'id_time' + '_' + str(inst_id): 'id_time'}, inplace=True)

        predictors_df = predictors_df.merge(tmp_df, on='id_time', how='left')

        predictors_df.fillna(predictors_df.mean(), inplace=True)

    scaler = StandardScaler()
    predictors_scaled_df = predictors_df.copy(deep=True)
    predictors_scaled_df.iloc[:,3:] = scaler.fit_transform(predictors_df.iloc[:,3:])
            
    pca_object = PCA(n_components=nof_comp)
    X_pca = pca_object.fit_transform(predictors_df.iloc[:,1:])
        
    svd_object = TruncatedSVD(n_components=nof_comp)
    X_svd = svd_object.fit_transform(predictors_df.iloc[:,1:])

    if autosave:
        predictors_df.to_csv(os.path.join(proj_root, 'preprocessed_datasets', 'predictors_df.csv'), index=False)
                
    return (predictors_df.set_index('id_time').values, 
            predictors_scaled_df.set_index('id_time').values, 
            X_pca, 
            X_svd, 
            list(predictors_df.columns[1:]))

#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#