import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def run_windowed_predict(X_flat: np.ndarray, y_full: np.ndarray, 
                         post_window: int=7, horizon: int=1, nof_svd_comp_left: int=0):
    '''
    this function is intended for grid search of optimal parameters and data preprocessing,
    it starts from reformatting flat data into windowed format and cutting target array accordingly

    Parameters:
    - X_flat (np.ndarray): The flattened input data.
    - y_full (np.ndarray): The full target data.
    - post_window (int): The number of time steps to include into predictors window.
    - horizon (int): The number of time steps to predict into the future.
    - nof_svd_comp_left (int): The number of SVD (or PCA) main components to use.

    Returns:
    dict: A dictionary containing the evaluation metrics for both models.
    '''

    if nof_svd_comp_left > 0:
        X_wd = np.stack([
                         X_flat[i-post_window:i, :nof_svd_comp_left].flatten().copy() 
                         for i in range(post_window, X_flat.shape[0]-horizon)
                        ], axis=0)
    else:
        X_wd = np.stack([
                         X_flat[i-post_window:i].flatten().copy() 
                         for i in range(post_window, X_flat.shape[0]-horizon)
                        ], axis=0)
        
    y_wd = y_full[post_window + horizon : X_flat.shape[0]]

    X_wd_train = X_wd[:-50]
    X_wd_test = X_wd[-50:]

    y_wd_train = y_wd[:-50]
    y_wd_test = y_wd[-50:]

    #model_RF = RandomForestClassifier()
    model_HGB = HistGradientBoostingClassifier()

    #model_RF.fit(X_wd_train, y_wd_train)
    model_HGB.fit(X_wd_train, y_wd_train)

    #y_wd_pred_RF = model_RF.predict(X_wd_test)
    y_wd_pred_HGB = model_HGB.predict(X_wd_test)

    #y_wd_proba_RF = model_RF.predict_proba(X_wd_test)[:, 1]
    y_wd_proba_HGB = model_HGB.predict_proba(X_wd_test)[:, 1]

    
    return [
        model_HGB.score(X_wd_test, y_wd_test),
        roc_auc_score(y_wd_test, y_wd_proba_HGB),
        precision_score(y_wd_test, y_wd_pred_HGB),
        recall_score(y_wd_test, y_wd_pred_HGB),
        f1_score(y_wd_test, y_wd_pred_HGB)
    ], model_HGB

    #return {
    #   'accu_RF': model_RF.score(X_wd_test, y_wd_test),
    #   'accu_HGB': model_HGB.score(X_wd_test, y_wd_test),
    #   'roc_auc_RF': roc_auc_score(y_wd_test, y_wd_proba_RF),
    #   'roc_auc_HGB': roc_auc_score(y_wd_test, y_wd_proba_HGB),
    #   'prec_RF': precision_score(y_wd_test, y_wd_pred_RF),
    #   'prec_HGB': precision_score(y_wd_test, y_wd_pred_HGB),
    #   'rec_RF': recall_score(y_wd_test, y_wd_pred_RF),
    #   'rec_HGB': recall_score(y_wd_test, y_wd_pred_HGB),
    #   'f1_RF': f1_score(y_wd_test, y_wd_pred_RF),
    #   'f1_HGB': f1_score(y_wd_test, y_wd_pred_HGB),
    #   'model_RF': model_RF,
    #   'model_HGB': model_HGB
    #}