

from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import gc, pytrec_eval

def evaluator(y_pred: np.ndarray, y_gt: np.ndarray, mnar: False, pscore: None) -> Tuple:
    """
    evaluator Return evaluation metrics

    Args:
        history ([type]): [description]
        X_test (np.ndarray): [description]

    Returns:
        Tuple: [description]
    """    
    # Generate predictions
    #y_pred = history.model.predict(X_test)
    if not mnar:
        mae, mse = mean_absolute_error(y_gt, y_pred), mean_squared_error(y_gt, y_pred)
    else:
        # if isinstance(y_gt, np.ndarray) and isinstance(y_pred, np.ndarray):
        #     mae = np.mean( (np.abs(y_gt - y_pred) )  * pscore )
        #     mse = np.mean( np.square(np.abs(y_gt - y_pred) )  * pscore ) 
        # else:
        #     y_gt = np.array(y_gt)
        #     y_pred = np.array(y_pred)
        y_gt = np.ndarray.flatten(y_gt)
        y_pred = np.ndarray.flatten(y_pred)
        mae = np.mean( (np.abs(y_gt - y_pred) )  * pscore )
        mse = np.mean( np.square(np.abs(y_gt - y_pred) )  * pscore ) 
        gc.collect()
    return mae, mse

def evaluator_ranking(data) -> Tuple:
    """
    evaluator Return evaluation metrics

    Args:
        history ([type]): [description]
        X_test (np.ndarray): [description]

    Returns:
        Tuple: [description]
    """    
    # Generate predictions
    #y_pred = history.model.predict(X_test)
    data['uid_c'] = data['uid'].apply(lambda x: 'q' + str(x))
    data['pids_c'] = data['pids'].apply(lambda x: 'd' + str(x))
    # data['rating_c'] = data['rating'].apply(lambda x: str(x))
    # data['scores_c'] = data['scores'].apply(lambda x: str(x))
    qrel = {k: f.groupby('pids_c')['rating'].apply(lambda x:list(x)[0]).to_dict()
     for k, f in data.groupby('uid_c')}
    run = {k: f.groupby('pids_c')['scores'].apply(lambda x:list(x)[0]).to_dict()
     for k, f in data.groupby('uid_c')}
    evaluator = pytrec_eval.RelevanceEvaluator(
    qrel, {'map_cut_3', 'ndcg_cut_3'})
    result = evaluator.evaluate(run)
    Map = np.mean([v['map_cut_3'] for k, v in result.items()])
    ndcg = np.mean([v['ndcg_cut_3'] for k, v in result.items()]) 
    return Map, ndcg

