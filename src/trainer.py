

import yaml
from pathlib import Path
from typing import Tuple, Dict

import pandas as pd
import numpy as np
from tensorflow import keras
from scipy import sparse

from sklearn.model_selection import StratifiedKFold

from keras.models import Model
from keras.layers import Input, Reshape, Dot
from keras.layers.embeddings import Embedding

from models.mf import RecommenderV1, Recommender
from models.expoMF import ExpoMF
from evaluate.metrics import evaluator, evaluator_ranking

import gc
import tensorflow as tf


def train_mf_kfold(train_ua_df: pd.DataFrame, test_ua_df: pd.DataFrame, prop_ua_df: pd.DataFrame,\
                 ulf_dim: int, ilf_dim:int, reg_param: float, ips: bool, config:Dict, job_id: int, logging) -> Tuple:
    """
    Generates k-fold cross-val param performance for MF w/ or w/o IPS 
    Args:
        train_ua_df (pd.DataFrame): user * attribute rating df
        test_ua_df (pd.DataFrame):      ""      

    Returns:
        model: return trained model
    """   
    n_folds = config['k_folds'] 
    # Get propensity scores
    # Method-1 naive bayes. Compute propensity score as per ratings only
    N = float(train_ua_df.shape[0])
    Nr = len(np.unique(train_ua_df.rating))
    # Total possible ratings 
    M = float(len(np.unique(train_ua_df.uid)) * len(np.unique(train_ua_df.pid)))
    mnar_rating_prop = np.unique(train_ua_df.rating, return_counts=True)[1]/N
    observation_prob = N/M
    mcar_rating_prop = np.unique(prop_ua_df.rating, return_counts=True)[1]/float(prop_ua_df.shape[0])
    prop_score = (mnar_rating_prop * observation_prob)/mcar_rating_prop
    prop_score_dict = dict(zip(range(1, Nr+1), prop_score))

    # Compute sample/IPS weights 
    train_ua_df['p_score'] =train_ua_df['rating'].apply(lambda x: 1/prop_score_dict[x])
    test_ua_df['p_score'] =test_ua_df['rating'].apply(lambda x: 1/prop_score_dict[x])

    # n_users, n_items = len(np.unique(train_ua_df.uid)) ,\
    #      len(np.unique(train_ua_df.pid))

    n_users, n_items = max(np.unique(train_ua_df.uid)) +1 ,\
         max(np.unique(train_ua_df.pid)) + 1


    model = RecommenderV1(n_users, n_items, ulf_dim, ilf_dim, reg_param)
    model.summary()
    #model.save('./models/model_%d'%job_id)

    # K-fold cross validation for hyper-param tuning
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=12345)

    #X_train = [train_ua_df[['uid', 'pid']].values[:, 0], train_ua_df[['uid', 'pid']].values[:, 1]]
    #X_test = [test_ua_df[['uid', 'pid']].values[:, 0], test_ua_df[['uid', 'pid']].values[:, 1]]


    if ips == 0:
        logging.info('Starting K-fold cross-val for plain MF case')
        mae_folds = []
        mae_kfold = 0.0
        for train_index, test_index in skf.split(train_ua_df[['uid','pid']], train_ua_df['rating']):
            logging.info('Stratified split done')
            X_train = [train_ua_df[['uid', 'pid']].iloc[train_index].values[:, 0],\
                 train_ua_df[['uid', 'pid']].iloc[train_index].values[:, 1]]
            
            y_train = train_ua_df.iloc[train_index]['rating'].values
            model = RecommenderV1(n_users, n_items, ulf_dim, ilf_dim, reg_param)
            #model = keras.models.load_model('./models/model_%d'%job_id)

            model.summary()
            logging.info('Starting model training')
            history = model.fit(x=X_train, y=y_train,\
                            batch_size=config['batch_size'], epochs=config['max_iters'],\
                            verbose=0)
            logging.info('Finished model training')
            X_test = [train_ua_df[['uid', 'pid']].iloc[test_index].values[:, 0], \
                        train_ua_df[['uid', 'pid']].iloc[test_index].values[:, 1]]
            y_pred = model.predict(X_test)
            logging.info('Type of Ypred is %s'%type(y_pred))
            y_gt = train_ua_df.iloc[test_index]['rating'].values.astype(np.int)
            logging.info('Started model eval for pred of size %s'%str(y_pred.shape))
            pred_df = pd.DataFrame({'uid' : X_test[0],
                                'pids' : X_test[1],
                                'rating' : y_gt, 
                              'scores': y_pred.flatten() })
            pscore = (train_ua_df.iloc[test_index]['p_score'].values)/n_folds
            mae_mf, mse_mf = evaluator(y_pred, y_gt, mnar=True, pscore=pscore)
            map_mf, ndcg_mf = evaluator_ranking(pred_df)
            logging.info('Finished model eval')
            mae_folds.append(mae_mf)
            logging.info('1 round of K-fold for MF done')
            tf.keras.backend.clear_session()
        gc.collect()
        mae_kfold += np.mean(mae_folds)

    else:
        logging.info('Starting K-fold cross-val for MF-IPS case')
        mae_folds = []
        mae_kfold = 0.0
        for train_index, test_index in skf.split(train_ua_df[['uid','pid']], train_ua_df['rating']):
            logging.info('Stratified split done')
            X_train = [train_ua_df[['uid', 'pid']].iloc[train_index].values[:, 0],\
                 train_ua_df[['uid', 'pid']].iloc[train_index].values[:, 1]]
            
            y_train = train_ua_df.iloc[train_index]['rating'].values
            pscore_train = (train_ua_df.iloc[train_index]['p_score'].values) * float((n_folds-1)/n_folds)
            pscore_test = (train_ua_df.iloc[test_index]['p_score'].values)/n_folds
            #model = keras.models.load_model('./models/model_%d'%job_id)
            model = RecommenderV1(n_users, n_items, ulf_dim, ilf_dim, reg_param)
            model.summary()
            logging.info('Starting model training')
            history = model.fit(x=X_train, y=y_train,\
                            batch_size=config['batch_size'], epochs=config['max_iters'],\
                            verbose=0, sample_weight=pscore_train)
            logging.info('Finished model training')
            X_test = [train_ua_df[['uid', 'pid']].iloc[test_index].values[:, 0], \
                        train_ua_df[['uid', 'pid']].iloc[test_index].values[:, 1]]
            y_pred = model.predict(X_test)
            y_gt = train_ua_df.iloc[test_index]['rating'].values.astype(np.int)
            pred_df = pd.DataFrame({'uid' : X_test[0],
                                'pids' : X_test[1],
                                'rating' : y_gt, 
                              'scores': y_pred.flatten() })
            logging.info('Started model eval for pred of size %s'%str(y_pred.shape))
            mae_mf, mse_mf = evaluator(y_pred, y_gt, mnar=True, pscore=pscore_test)
            map_mf, ndcg_mf = evaluator_ranking(pred_df)
            logging.info('Finished model eval')
            mae_folds.append(mae_mf)
            logging.info('1 round of K-fold for ML-IPS done')
            tf.keras.backend.clear_session()
        gc.collect()
        mae_kfold += np.mean(mae_folds)


        # history = model.fit(x=X_train, y=train_ua_df['rating'].values, 
        #                 batch_size=config['batch_size'], epochs=config['max_iters'], 
        #                 verbose=1, sample_weight=train_ua_df['p_score'].values)
    

    return mae_kfold


def train_expoMF_kfold(train_ua_df: pd.DataFrame, test_ua_df: pd.DataFrame, prop_ua_df: pd.DataFrame,\
                 ulf_dim: int, reg_param: float, ips: bool, config:Dict, job_id: int, logging) -> Tuple:
    """
    Generates k-fold cross-val param performance for training expoMF 
    Args:
        train_ua_df (pd.DataFrame): user * attribute rating df
        test_ua_df (pd.DataFrame):      ""      

    Returns:
        model: return trained model
    """   
    n_folds = config['k_folds'] 
    # Get propensity scores
    # Method-1 naive bayes. Compute propensity score as per ratings only
    N = float(train_ua_df.shape[0])
    Nr = len(np.unique(train_ua_df.rating))
    # Total possible ratings 
    M = float(len(np.unique(train_ua_df.uid)) * len(np.unique(train_ua_df.pid)))
    mnar_rating_prop = np.unique(train_ua_df.rating, return_counts=True)[1]/N
    observation_prob = N/M
    mcar_rating_prop = np.unique(prop_ua_df.rating, return_counts=True)[1]/float(prop_ua_df.shape[0])
    prop_score = (mnar_rating_prop * observation_prob)/mcar_rating_prop
    prop_score_dict = dict(zip(range(1, Nr+1), prop_score))

    # Compute sample/IPS weights 
    train_ua_df['p_score'] =train_ua_df['rating'].apply(lambda x: 1/prop_score_dict[x])
    test_ua_df['p_score'] =test_ua_df['rating'].apply(lambda x: 1/prop_score_dict[x])

    n_users, n_items = max(np.unique(train_ua_df.uid)) +1 ,\
         max(np.unique(train_ua_df.pid)) + 1

    #model = ExpoMF(n_components=ulf_dim, max_iter=10, batch_size=1000)
    #model.save('./models/model_%d'%job_id)

    # K-fold cross validation for hyper-param tuning
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=12345)

    #X_train = [train_ua_df[['uid', 'pid']].values[:, 0], train_ua_df[['uid', 'pid']].values[:, 1]]
    #X_test = [test_ua_df[['uid', 'pid']].values[:, 0], test_ua_df[['uid', 'pid']].values[:, 1]]

    logging.info('Starting K-fold cross-val for plain MF case')
    map_folds = []
    map_kfold = 0.0
    for train_index, test_index in skf.split(train_ua_df[['uid','pid']], train_ua_df['rating']):
        logging.info('Stratified split done')
        # construct dense user * item matrix from rating matrix
        # train_users, train_items, train_ratings = train_ua_df[['uid', 'pid']].iloc[train_index].values[:, 0], \
        #             train_ua_df[['uid', 'pid']].iloc[train_index].values[:, 1], train_ua_df[['rating']].iloc[train_index].values
        X_train = sparse.coo_matrix(( train_ua_df.iloc[train_index]['rating'].values, (train_ua_df.iloc[train_index][['uid', 'pid']].values[:, 0], train_ua_df.iloc[train_index][['uid', 'pid']].values[:, 1]))).tocsr()
        logging.info('X_train dim is %s'%str(X_train.shape))
        #y_train = train_ua_df.iloc[train_index]['rating'].values
        pscore_train = (train_ua_df.iloc[train_index]['p_score'].values) * float((n_folds-1)/n_folds)
        pscore_test = (train_ua_df.iloc[test_index]['p_score'].values)/n_folds
        model = ExpoMF(n_components=ulf_dim, max_iter=10, batch_size=1000)
        #model = keras.models.load_model('./models/model_%d'%job_id)
        logging.info('Starting ExpoMF model training')
        model = model.fit(X_train)
        logging.info('Finished model training')
        logging.info('alpha param dom is %s'%str(model.theta.shape))
        logging.info('beta param dom is %s'%str(model.beta.shape))
        X_test = [train_ua_df[['uid', 'pid']].iloc[test_index].values[:, 0], \
                    train_ua_df[['uid', 'pid']].iloc[test_index].values[:, 1]]
        y_pred = model.predict(X_test[0], X_test[1])
        logging.info('Type of Ypred is %s'%type(y_pred))
        y_gt = train_ua_df.iloc[test_index]['rating'].values.astype(np.int)
        pred_df = pd.DataFrame({'uid' : X_test[0],
                                'pids' : X_test[1],
                                'rating' : y_gt, 
                              'scores': y_pred })
        logging.info('Started model eval for pred of size %s'%str(y_pred.shape))
        map_expomf, ndcg_expomf = evaluator_ranking(pred_df)
        mae_expomf, mse_expomf = evaluator(y_pred, y_gt, mnar=True, pscore=pscore_test)
        logging.info('Finished model eval')
        map_folds.append(mae_expomf)
        logging.info('1 round of K-fold for MF done')
        tf.keras.backend.clear_session()
    gc.collect()
    map_kfold += np.mean(map_folds)
    return map_kfold


def train_expoMF(train_ua_df: pd.DataFrame, test_ua_df: pd.DataFrame,\
                 ulf_dim: int, reg_param: float, ips: bool, config:Dict, logging) -> Tuple:
    """
    Generates k-fold cross-val param performance for training expoMF 
    Args:
        train_ua_df (pd.DataFrame): user * attribute rating df
        test_ua_df (pd.DataFrame):      ""      

    Returns:
        model: return trained model
    """   
    n_folds = config['k_folds'] 
    # Get propensity scores
    # Method-1 naive bayes. Compute propensity score as per ratings only
    N = float(train_ua_df.shape[0])
    Nr = len(np.unique(train_ua_df.rating))
    # Total possible ratings 
    M = float(len(np.unique(train_ua_df.uid)) * len(np.unique(train_ua_df.pid)))
    
    # n_users, n_items = len(np.unique(train_ua_df.uid)) ,\
    #      len(np.unique(train_ua_df.pid))

    n_users, n_items = max(np.unique(train_ua_df.uid)) +1 ,\
         max(np.unique(train_ua_df.pid)) + 1

    #model = ExpoMF(n_components=ulf_dim, max_iter=10, batch_size=1000)
    #model.save('./models/model_%d'%job_id)

    # K-fold cross validation for hyper-param tuning
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=12345)

    #X_train = [train_ua_df[['uid', 'pid']].values[:, 0], train_ua_df[['uid', 'pid']].values[:, 1]]
    #X_test = [test_ua_df[['uid', 'pid']].values[:, 0], test_ua_df[['uid', 'pid']].values[:, 1]]

    logging.info('Starting K-fold cross-val for plain MF case')
    map_folds = []
    map_kfold = 0.0
    
    # construct dense user * item matrix from rating matrix
    # train_users, train_items, train_ratings = train_ua_df[['uid', 'pid']].iloc[train_index].values[:, 0], \
    #             train_ua_df[['uid', 'pid']].iloc[train_index].values[:, 1], train_ua_df[['rating']].iloc[train_index].values
    X_train = sparse.coo_matrix(( train_ua_df['rating'].values, (train_ua_df[['uid', 'pid']].values[:, 0], train_ua_df[['uid', 'pid']].values[:, 1]))).tocsr()
    logging.info('X_train dim is %s'%str(X_train.shape))
    #y_train = train_ua_df.iloc[train_index]['rating'].values
    model = ExpoMF(n_components=ulf_dim, max_iter=10, batch_size=1000)
    #model = keras.models.load_model('./models/model_%d'%job_id)
    logging.info('Starting ExpoMF model training')
    model = model.fit(X_train)
    # logging.info('Finished model training')
    # logging.info('alpha param dom is %s'%str(model.theta.shape))
    # logging.info('beta param dom is %s'%str(model.beta.shape))
    # X_test = [train_ua_df[['uid', 'pid']].values[:, 0], \
    #             train_ua_df[['uid', 'pid']].values[:, 1]]
    # y_pred = model.predict(X_test[0], X_test[1])
    # logging.info('Type of Ypred is %s'%type(y_pred))
    # y_gt = train_ua_df['rating'].values.astype(np.int)
    # pred_df = pd.DataFrame({'uid' : X_test[0],
    #                         'pids' : X_test[1],
    #                         'rating' : y_gt, 
    #                         'scores': y_pred })
    # logging.info('Started model eval for pred of size %s'%str(y_pred.shape))
    # map_expomf, ndcg_expomf = evaluator_ranking(pred_df)
    # logging.info('Finished model eval')
    # map_folds.append(ndcg_expomf)
    # logging.info('1 round of K-fold for MF done')
    # tf.keras.backend.clear_session()
    # gc.collect()
    # map_kfold += np.mean(map_folds)
    # return map_kfold
    return model


def train_mf(train_ua_df: pd.DataFrame, test_ua_df: pd.DataFrame, prop_ua_df: pd.DataFrame,\
                 ulf_dim: int, ilf_dim:int, reg_param: float, ips: bool, config:Dict) -> Tuple:
    """
    trains MF w/ or w/o IPS 
    Args:
        train_ua_df (pd.DataFrame): user * attribute rating df
        test_ua_df (pd.DataFrame):      ""      

    Returns:
        model: return trained model
    """    
    # Get propensity scores
    # Method-1 naive bayes. Compute propensity score as per ratings only
    N = float(train_ua_df.shape[0])
    Nr = len(np.unique(train_ua_df.rating))
    # Total possible ratings 
    M = float(len(np.unique(train_ua_df.uid)) * len(np.unique(train_ua_df.pid)))
    mnar_rating_prop = np.unique(train_ua_df.rating, return_counts=True)[1]/N
    observation_prob = N/M
    mcar_rating_prop = np.unique(prop_ua_df.rating, return_counts=True)[1]/float(prop_ua_df.shape[0])
    prop_score = (mnar_rating_prop * observation_prob)/mcar_rating_prop
    prop_score_dict = dict(zip(range(1, Nr+1), prop_score))

    # Compute sample/IPS weights 
    train_ua_df['p_score'] =train_ua_df['rating'].apply(lambda x: 1/prop_score_dict[x])
    test_ua_df['p_score'] =test_ua_df['rating'].apply(lambda x: 1/prop_score_dict[x])

    # n_users, n_items = len(np.unique(train_ua_df.uid)) ,\
    #      len(np.unique(train_ua_df.pid))
    
    n_users, n_items = max(np.unique(train_ua_df.uid)) +1 ,\
         max(np.unique(train_ua_df.pid)) + 1

    model = RecommenderV1(n_users, n_items, ulf_dim, ilf_dim, reg_param)
    model.summary()

    # K-fold cross validation for hyper-param tuning
    #skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=12345)

    X_train = [train_ua_df[['uid', 'pid']].values[:, 0], train_ua_df[['uid', 'pid']].values[:, 1]]
    #X_test = [test_ua_df[['uid', 'pid']].values[:, 0], test_ua_df[['uid', 'pid']].values[:, 1]]
    y_train = train_ua_df['rating'].values

    if ips == 0:
        
        history = model.fit(x=X_train, y=y_train,\
                        batch_size=config['batch_size'], epochs=config['max_iters'],\
                        verbose=0)
        

    else:
            
        history = model.fit(x=X_train, y=train_ua_df['rating'].values, 
                        batch_size=config['batch_size'], epochs=config['max_iters'], 
                        verbose=0, sample_weight=train_ua_df['p_score'].values)
    

    return history



