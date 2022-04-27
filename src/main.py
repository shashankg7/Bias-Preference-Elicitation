
import argparse
from numpy.core.numeric import cross
import yaml
import warnings
import pandas as pd
import glob
import logging
from scipy import stats

import tensorflow as tf
import numpy as np                                                         
from utils.data_loader import load_coat, load_synthetic, load_yahoo
from trainer import train_mf, train_mf_kfold, train_expoMF
from evaluate.metrics import evaluator, evaluator_ranking

import argparse


possible_model_names = ['cold-start', 'casual-user']
data = ['coat', 'yahoo', 'synthetic']


parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', type=str,
                    required=True)
# parser.add_argument('-job_id', type=int,
#                  required=True)
# parser.add_argument('-user_lf', type=int,
#                  required=True)
# parser.add_argument('-reg_param', type=float,
#                  required=True)
parser.add_argument('--alpha', type=float,
                 required=False)

args = parser.parse_args()
embed_dim = 0
embedding_file = open('./models/embeddings/vectors_v.dat', 'r')
for line in embedding_file:
    embed_dim = len(line.strip().split(' ')[1:])
    break


# else:
#     logging.basicConfig(filename=log_file_name, filemode='a+', format='%(name)s - %(levelname)s - %(message)s', \
#                     level=logging.INFO)

config_file_name = './reports/results/final_results.txt'
# if not os.path.isfile(config_file_name):
cross_val_file = open(config_file_name, 'w')
# else:
#     cross_val_file = open(config_file_name, 'a+')
headers = 'Dataset' + '\t' + 'Method' +  '\t' + 'MSE' + '\t' + 'MAE' + '\n'
cross_val_file.write(headers)
# parser.add_argument('--data_type', '-d', type=str,
#                     choices=data, required=True)

if __name__ == '__main__':
    print("starting code")
    
    config = yaml.safe_load(open('./config/consts.yaml', 'rb'))
    print(config)
    
    if args.data == 'coat':
        log_file_name = './logs/pref_elicit_main_coat.log' 

        logging.basicConfig(filename=log_file_name, filemode='w', format='%(name)s - %(levelname)s - %(message)s', \
                level=logging.INFO)
        res = load_coat()
        assert len(res) == 6
        logging.info('Coat data loaded, loading cross-val results')
        cross_val_files = glob.glob('./reports/results/cross_val_results_*')
        df_results = []
        for cross_val_file in cross_val_files:
            df_res = pd.read_csv(cross_val_file, delimiter='\t')
            df_results.append(df_res)
        df_params = pd.concat(df_results, ignore_index=True)
        print(df_params)
        logging.info('columns are %s'%df_params.columns[0])
        
        df_params = df_params[df_params['Dataset'] == 'COAT'].reset_index()
        df_params_mf = df_params[df_params['Method'] == 'MF'].reset_index()
        df_params_mfips = df_params[df_params['Method'] == 'MF-IPS'].reset_index()
        df_params_expomf = df_params[df_params['Method'] == 'ExpoMF'].reset_index()

        best_param_idx_mf = df_params_mf['MSE'].idxmin()
        best_param_idx_mfips = df_params_mfips['MSE'].idxmin()
        best_param_idx_expomf = df_params_expomf['MSE'].idxmin()

        lf_dim_best_mf, reg_param_best_mf = df_params_mf.iloc[best_param_idx_mf]['lf_dim'],\
                                    df_params_mf.iloc[best_param_idx_mf]['reg_param']

        lf_dim_best_mfips, reg_param_best_mfips = df_params_mfips.iloc[best_param_idx_mfips]['lf_dim'],\
                                    df_params_mfips.iloc[best_param_idx_mfips]['reg_param']

        lf_dim_best_expomf, reg_param_best_expomf = df_params_expomf.iloc[best_param_idx_expomf]['lf_dim'],\
                                    df_params_expomf.iloc[best_param_idx_expomf]['reg_param']



        # df_params = df_params[df_params['Dataset'] == 'COAT'].reset_index()
        # best_param_idx = df_params[df_params['Method'] == 'MF']['MSE'].idxmin()
        # lf_dim_best, reg_param_best = df_params[df_params['Method'] == 'MF'].iloc[best_param_idx]['lf_dim'],\
        #                             df_params[df_params['Method'] == 'MF'].iloc[best_param_idx]['reg_param']

        # idx_min_coat = df_res[df_res['Dataset'] == 'COAT']['MSE'].idxmin()
            # #idx_min_yahoo = df_res[df_res['Dataset'] == 'Yahoo']['MSE'].idxmin()
            # lf_dim_MF = 


        train_ua_df, test_ua_df, prop_ua_df = res[2], res[3], res[4]
        # ulf_range = config['user_lf_range']
        # ilf_range = config['item_lf_range']
        #reg_param_range = config['reg_param_range']
        X_test = [test_ua_df[['uid', 'pid']].values[:, 0], test_ua_df[['uid', 'pid']].values[:, 1]]
        y_gt = test_ua_df['rating'].values
        # train MF plain model
        per_dict = {}
        per_dict_ips = {}
        print("Starting cross-val hyper-param tuning")
        #for ulf_dim in ulf_range:
        #for ilf_dim in ilf_range:
        #for reg_param in reg_param_range:
        #ulf_dim = args.user_lf
        #reg_param = args.reg_param
        
        maes_mf, mses_mf, maes_mfips, mses_mfips, maes_expomf, mses_expomf = [], [], [], [], [], []
        ndcgs_mf, ndcgs_mfips, ndcgs_expomf = [], [], []

        logging.info('Starting final %d runs for COAT data to get models performance metric'%(config['runs']))
        for n_runs in range(config['runs']):
            history_mf = train_mf(train_ua_df, test_ua_df, prop_ua_df, \
                        ulf_dim=lf_dim_best_mf, ilf_dim=lf_dim_best_mf, reg_param=reg_param_best_mf, \
                        ips=0, config=config)
            history_mfips = train_mf(train_ua_df, test_ua_df, prop_ua_df, \
                            ulf_dim=lf_dim_best_mfips, ilf_dim=lf_dim_best_mfips, reg_param=reg_param_best_mfips, \
                                ips=1, config=config)
            history_expomf = train_expoMF(train_ua_df, test_ua_df,  \
                            ulf_dim=lf_dim_best_expomf, reg_param=reg_param_best_expomf, \
                                ips=1, config=config, logging=logging)

            y_pred_mf = history_mf.model.predict(X_test)
            y_pred_mfips = history_mfips.model.predict(X_test)
            y_pred_expomf = history_expomf.predict(X_test[0], X_test[1])
            mae_mf, mse_mf = evaluator(y_pred_mf, y_gt, mnar=False, pscore=None)
            mae_mfips, mse_mfips = evaluator(y_pred_mfips, y_gt, mnar=False, pscore=None)
            mae_expomf, mse_expomf = evaluator(y_pred_expomf, y_gt, mnar=False, pscore=None)
            #y_gt = train_ua_df['rating'].values.astype(np.int)
            pred_df = pd.DataFrame({'uid' : X_test[0],
                                    'pids' : X_test[1],
                                    'rating' : y_gt.astype(np.int), 
                                'scores': y_pred_expomf})
            #logging.info('Started model eval for pred of size %s'%str(y_pred.shape))
            map_expomf, ndcg_expomf = evaluator_ranking(pred_df)
            pred_df = pd.DataFrame({'uid' : X_test[0],
                                    'pids' : X_test[1],
                                    'rating' : y_gt.astype(np.int), 
                                'scores': y_pred_mf.flatten()})
            #logging.info('Started model eval for pred of size %s'%str(y_pred.shape))
            map_mf, ndcg_mf = evaluator_ranking(pred_df)
            pred_df = pd.DataFrame({'uid' : X_test[0],
                                    'pids' : X_test[1],
                                    'rating' : y_gt.astype(np.int), 
                                'scores': y_pred_mfips.flatten()})
            #logging.info('Started model eval for pred of size %s'%str(y_pred.shape))
            map_mfips, ndcg_mfips = evaluator_ranking(pred_df)

            maes_mf.append(mae_mf)
            mses_mf.append(mse_mf)
            maes_expomf.append(mae_expomf)

            mses_expomf.append(mse_expomf)
            maes_mfips.append(mae_mfips)
            mses_mfips.append(mse_mfips)

            ndcgs_mf.append(ndcg_mf)
            ndcgs_mfips.append(ndcg_mfips)
            ndcgs_expomf.append(ndcg_expomf)

            logging.info('Finished one run')
        logging.info('Finished all runs')
        logging.info('Results for COAT dataset are')

        logging.info("MAE_MF: mean %f +- std %f"%(np.mean(maes_mf), np.std(maes_mf)))
        logging.info("MSE_MF: mean %f +- std %f"%(np.mean(mses_mf), np.std(mses_mf)))
        logging.info("MAE_MFIPS: mean %f +- std %f"%(np.mean(maes_mfips), np.std(maes_mfips)))
        logging.info("MSE_MFIPS: mean %f +- std %f"%(np.mean(mses_mfips), np.std(mses_mfips)))
        logging.info("MAE_expoMF: mean %f +- std %f"%(np.mean(maes_expomf), np.std(maes_expomf)))
        logging.info("MSE_expoMF: mean %f +- std %f"%(np.mean(mses_expomf), np.std(mses_expomf)))
        logging.info("NDCG@3_MF: mean %f +- std %f"%(np.mean(ndcgs_mf), np.std(ndcgs_mf)))
        logging.info("NDCG@3_MFIPS: mean %f +- std %f"%(np.mean(ndcgs_mfips), np.std(ndcgs_mfips)))
        logging.info("NDCG@3_ExpoMF: mean %f +- std %f"%(np.mean(ndcgs_expomf), np.std(ndcgs_expomf)))
        
        logging.info("MAE student t-test diff %f p-value std %f"%((stats.ttest_rel(maes_mf, maes_mfips)[0]), \
                        stats.ttest_rel(maes_mf, maes_mfips)[1]))
        logging.info("MSE student t-test diff %f p-value std %f"%((stats.ttest_rel(mses_mf, mses_mfips)[0]), \
                        stats.ttest_rel(mses_mf, mses_mfips)[1]))
        # run = 'COAT' + '\t' + 'MF' + '\t' + str(ulf_dim) + '\t' + str(reg_param) + '\t' + str(mse_mf) + '\n'
        # cross_val_file.write(run)
        # run = 'COAT' + '\t' + 'MF-IPS' + '\t' + str(ulf_dim) + '\t' + str(reg_param) + '\t' + str(mse_mf_ips) + '\n'
        # cross_val_file.write(run)

        
    elif args.data == 'yahoo':
        log_file_name = './logs/pref_elicit_main_%d.log'%embed_dim 

        logging.basicConfig(filename=log_file_name, filemode='w', format='%(name)s - %(levelname)s - %(message)s', \
                level=logging.INFO)
        res = load_yahoo()
        assert len(res) == 6
        logging.info('Yahoo data loaded, loading cross-val results')
        cross_val_files = glob.glob('./reports/results/cross_val_results_*')
        df_results = []
        for cross_val_file in cross_val_files:
            df_res = pd.read_csv(cross_val_file, delimiter='\t')
            df_results.append(df_res)
        df_params = pd.concat(df_results, ignore_index=True)
        df_params = df_params[df_params['Dataset'] == 'Yahoo'].reset_index()
        df_params_mf = df_params[df_params['Method'] == 'MF'].reset_index()
        df_params_mfips = df_params[df_params['Method'] == 'MF-IPS'].reset_index()
        df_params_expomf = df_params[df_params['Method'] == 'ExpoMF'].reset_index()

        best_param_idx_mf = df_params_mf['MSE'].idxmin()
        best_param_idx_mfips = df_params_mfips['MSE'].idxmin()
        best_param_idx_expomf = df_params_expomf['MSE'].idxmin()

        lf_dim_best_mf, reg_param_best_mf = df_params_mf.iloc[best_param_idx_mf]['lf_dim'],\
                                    df_params_mf.iloc[best_param_idx_mf]['reg_param']

        lf_dim_best_mfips, reg_param_best_mfips = df_params_mfips.iloc[best_param_idx_mfips]['lf_dim'],\
                                    df_params_mfips.iloc[best_param_idx_mfips]['reg_param']

        lf_dim_best_expomf, reg_param_best_expomf = df_params_expomf.iloc[best_param_idx_expomf]['lf_dim'],\
                                    df_params_expomf.iloc[best_param_idx_expomf]['reg_param']


        
        # idx_min_coat = df_res[df_res['Dataset'] == 'COAT']['MSE'].idxmin()
            # #idx_min_yahoo = df_res[df_res['Dataset'] == 'Yahoo']['MSE'].idxmin()
            # lf_dim_MF = 


        train_ua_df, test_ua_df, prop_ua_df = res[2], res[3], res[4]
        # ulf_range = config['user_lf_range']
        # ilf_range = config['item_lf_range']
        #reg_param_range = config['reg_param_range']
        X_test = [test_ua_df[['uid', 'pid']].values[:, 0], test_ua_df[['uid', 'pid']].values[:, 1]]
        y_gt = test_ua_df['rating'].values
        # train MF plain model
        per_dict = {}
        per_dict_ips = {}
        print("Starting cross-val hyper-param tuning")
        #for ulf_dim in ulf_range:
        #for ilf_dim in ilf_range:
        #for reg_param in reg_param_range:
        #ulf_dim = args.user_lf
        #reg_param = args.reg_param
        
        maes_mf, mses_mf, maes_mfips, mses_mfips, maes_expomf, mses_expomf = [], [], [], [], [], []
        ndcgs_mf, ndcgs_mfips, ndcgs_expomf = [], [], []

        logging.info('Starting final %d runs for YAHOO dataset to get models performance metric'%(config['runs']))
        for n_runs in range(config['runs']):
            history_mf = train_mf(train_ua_df, test_ua_df, prop_ua_df, \
                        ulf_dim=lf_dim_best_mf, ilf_dim=lf_dim_best_mf, reg_param=reg_param_best_mf, \
                        ips=0, config=config)
            history_mfips = train_mf(train_ua_df, test_ua_df, prop_ua_df, \
                            ulf_dim=lf_dim_best_mfips, ilf_dim=lf_dim_best_mfips, reg_param=reg_param_best_mfips, \
                                ips=1, config=config)
            history_expomf = train_expoMF(train_ua_df, test_ua_df, \
                            ulf_dim=lf_dim_best_expomf, reg_param=reg_param_best_expomf, \
                                ips=1, config=config, logging=logging)

            y_pred_mf = history_mf.model.predict(X_test)
            y_pred_mfips = history_mfips.model.predict(X_test)
            y_pred_expomf = history_expomf.predict(X_test[0], X_test[1])
            mae_mf, mse_mf = evaluator(y_pred_mf, y_gt, mnar=False, pscore=None)
            mae_mfips, mse_mfips = evaluator(y_pred_mfips, y_gt, mnar=False, pscore=None)
            mae_expomf, mse_expomf = evaluator(y_pred_expomf, y_gt, mnar=False, pscore=None)
            #y_gt = train_ua_df['rating'].values.astype(np.int)
            pred_df = pd.DataFrame({'uid' : X_test[0],
                                    'pids' : X_test[1],
                                    'rating' : y_gt.astype(np.int), 
                                'scores': y_pred_expomf})
            #logging.info('Started model eval for pred of size %s'%str(y_pred.shape))
            map_expomf, ndcg_expomf = evaluator_ranking(pred_df)
            pred_df = pd.DataFrame({'uid' : X_test[0],
                                    'pids' : X_test[1],
                                    'rating' : y_gt.astype(np.int), 
                                'scores': y_pred_mf.flatten()})
            #logging.info('Started model eval for pred of size %s'%str(y_pred.shape))
            map_mf, ndcg_mf = evaluator_ranking(pred_df)
            pred_df = pd.DataFrame({'uid' : X_test[0],
                                    'pids' : X_test[1],
                                    'rating' : y_gt.astype(np.int), 
                                'scores': y_pred_mfips.flatten()})
            #logging.info('Started model eval for pred of size %s'%str(y_pred.shape))
            map_mfips, ndcg_mfips = evaluator_ranking(pred_df)

            maes_mf.append(mae_mf)
            mses_mf.append(mse_mf)
            maes_expomf.append(mae_expomf)

            mses_expomf.append(mse_expomf)
            maes_mfips.append(mae_mfips)
            mses_mfips.append(mse_mfips)

            ndcgs_mf.append(ndcg_mf)
            ndcgs_mfips.append(ndcg_mfips)
            ndcgs_expomf.append(ndcg_expomf)

            logging.info('Finished one run')
        logging.info('Finished all runs')
        logging.info('Results for Yahoo dataset are')
        

        logging.info("MAE_MF: mean %f +- std %f"%(np.mean(maes_mf), np.std(maes_mf)))
        logging.info("MSE_MF: mean %f +- std %f"%(np.mean(mses_mf), np.std(mses_mf)))
        logging.info("MAE_MFIPS: mean %f +- std %f"%(np.mean(maes_mfips), np.std(maes_mfips)))
        logging.info("MSE_MFIPS: mean %f +- std %f"%(np.mean(mses_mfips), np.std(mses_mfips)))
        logging.info("MAE_expoMF: mean %f +- std %f"%(np.mean(maes_expomf), np.std(maes_expomf)))
        logging.info("MSE_expoMF: mean %f +- std %f"%(np.mean(mses_expomf), np.std(mses_expomf)))
        logging.info("NDCG@3_MF: mean %f +- std %f"%(np.mean(ndcgs_mf), np.std(ndcgs_mf)))
        logging.info("NDCG@3_MFIPS: mean %f +- std %f"%(np.mean(ndcgs_mfips), np.std(ndcgs_mfips)))
        logging.info("NDCG@3_ExpoMF: mean %f +- std %f"%(np.mean(ndcgs_expomf), np.std(ndcgs_expomf)))

        logging.info("MAE student t-test diff %f p-value std %f"%((stats.ttest_rel(maes_mf, maes_mfips)[0]), \
                        stats.ttest_rel(maes_mf, maes_mfips)[1]))
        logging.info("MSE student t-test diff %f p-value std %f"%((stats.ttest_rel(mses_mf, mses_mfips)[0]), \
                        stats.ttest_rel(mses_mf, mses_mfips)[1]))
        #logging.info("MAE_MF-IPS %f, MSE_MF-IPS %f"%(mae_mfips, mse_mfips))
        # run = 'COAT' + '\t' + 'MF' + '\t' + str(ulf_dim) + '\t' + str(reg_param) + '\t' + str(mse_mf) + '\n'
        # cross_val_file.write(run)
        # run = 'COAT' + '\t' + 'MF-IPS' + '\t' + str(ulf_dim) + '\t' + str(reg_param) + '\t' + str(mse_mf_ips) + '\n'
        # cross_val_file.write(run)

    else:
        log_file_name = './logs/pref_elicit_main_syn%f.log'%(args.alpha)

        logging.basicConfig(filename=log_file_name, filemode='w', format='%(name)s - %(levelname)s - %(message)s', \
                level=logging.INFO)
        res = load_synthetic(args.alpha)
        assert len(res) == 3
        logging.info('Coat data loaded, loading cross-val results')
        cross_val_files = glob.glob('./reports/results/cross_val_results_*')
        df_results = []
        for cross_val_file in cross_val_files:
            df_res = pd.read_csv(cross_val_file, delimiter='\t')
            df_results.append(df_res)
        df_params = pd.concat(df_results, ignore_index=True)
        print(df_params)
        logging.info('columns are %s'%df_params.columns[0])
        
        df_params = df_params[df_params['Dataset'] == 'Synthetic'].reset_index()
        df_params_mf = df_params[df_params['Method'] == 'MF'].reset_index()
        df_params_mfips = df_params[df_params['Method'] == 'MF-IPS'].reset_index()
        df_params_expomf = df_params[df_params['Method'] == 'ExpoMF'].reset_index()

        best_param_idx_mf = df_params_mf['MSE'].idxmin()
        best_param_idx_mfips = df_params_mfips['MSE'].idxmin()
        best_param_idx_expomf = df_params_expomf['MSE'].idxmin()

        lf_dim_best_mf, reg_param_best_mf = df_params_mf.iloc[best_param_idx_mf]['lf_dim'],\
                                    df_params_mf.iloc[best_param_idx_mf]['reg_param']

        lf_dim_best_mfips, reg_param_best_mfips = df_params_mfips.iloc[best_param_idx_mfips]['lf_dim'],\
                                    df_params_mfips.iloc[best_param_idx_mfips]['reg_param']

        lf_dim_best_expomf, reg_param_best_expomf = df_params_expomf.iloc[best_param_idx_expomf]['lf_dim'],\
                                    df_params_expomf.iloc[best_param_idx_expomf]['reg_param']



        # df_params = df_params[df_params['Dataset'] == 'COAT'].reset_index()
        # best_param_idx = df_params[df_params['Method'] == 'MF']['MSE'].idxmin()
        # lf_dim_best, reg_param_best = df_params[df_params['Method'] == 'MF'].iloc[best_param_idx]['lf_dim'],\
        #                             df_params[df_params['Method'] == 'MF'].iloc[best_param_idx]['reg_param']

        # idx_min_coat = df_res[df_res['Dataset'] == 'COAT']['MSE'].idxmin()
            # #idx_min_yahoo = df_res[df_res['Dataset'] == 'Yahoo']['MSE'].idxmin()
            # lf_dim_MF = 


        train_ua_df, test_ua_df, prop_ua_df = res[0], res[1], res[2]
        # ulf_range = config['user_lf_range']
        # ilf_range = config['item_lf_range']
        #reg_param_range = config['reg_param_range']
        X_test = [test_ua_df[['uid', 'pid']].values[:, 0], test_ua_df[['uid', 'pid']].values[:, 1]]
        y_gt = test_ua_df['rating'].values
        # train MF plain model
        per_dict = {}
        per_dict_ips = {}
        print("Starting cross-val hyper-param tuning")
        #for ulf_dim in ulf_range:
        #for ilf_dim in ilf_range:
        #for reg_param in reg_param_range:
        #ulf_dim = args.user_lf
        #reg_param = args.reg_param
        
        maes_mf, mses_mf, maes_mfips, mses_mfips, maes_expomf, mses_expomf = [], [], [], [], [], []
        ndcgs_mf, ndcgs_mfips, ndcgs_expomf = [], [], []

        logging.info('Starting final %d runs for synthetic data to get models performance metric'%(config['runs']))
        for n_runs in range(config['runs']):
            history_mf = train_mf(train_ua_df, test_ua_df, prop_ua_df, \
                        ulf_dim=lf_dim_best_mf, ilf_dim=lf_dim_best_mf, reg_param=reg_param_best_mf, \
                        ips=0, config=config)
            history_mfips = train_mf(train_ua_df, test_ua_df, prop_ua_df, \
                            ulf_dim=lf_dim_best_mfips, ilf_dim=lf_dim_best_mfips, reg_param=reg_param_best_mfips, \
                                ips=1, config=config)
            history_expomf = train_expoMF(train_ua_df, test_ua_df, \
                            ulf_dim=lf_dim_best_mfips, reg_param=reg_param_best_mfips, \
                                ips=1, config=config, logging=logging)

            y_pred_mf = history_mf.model.predict(X_test)
            y_pred_mfips = history_mfips.model.predict(X_test)
            y_pred_expomf = history_expomf.predict(X_test[0], X_test[1])
            mae_mf, mse_mf = evaluator(y_pred_mf, y_gt, mnar=False, pscore=None)
            mae_mfips, mse_mfips = evaluator(y_pred_mfips, y_gt, mnar=False, pscore=None)
            mae_expomf, mse_expomf = evaluator(y_pred_expomf, y_gt, mnar=False, pscore=None)
            #y_gt = train_ua_df['rating'].values.astype(np.int)
            pred_df = pd.DataFrame({'uid' : X_test[0],
                                    'pids' : X_test[1],
                                    'rating' : y_gt.astype(np.int), 
                                'scores': y_pred_expomf})
            #logging.info('Started model eval for pred of size %s'%str(y_pred.shape))
            map_expomf, ndcg_expomf = evaluator_ranking(pred_df)
            pred_df = pd.DataFrame({'uid' : X_test[0],
                                    'pids' : X_test[1],
                                    'rating' : y_gt.astype(np.int), 
                                'scores': y_pred_mf.flatten()})
            #logging.info('Started model eval for pred of size %s'%str(y_pred.shape))
            map_mf, ndcg_mf = evaluator_ranking(pred_df)
            pred_df = pd.DataFrame({'uid' : X_test[0],
                                    'pids' : X_test[1],
                                    'rating' : y_gt.astype(np.int), 
                                'scores': y_pred_mfips.flatten()})
            #logging.info('Started model eval for pred of size %s'%str(y_pred.shape))
            map_mfips, ndcg_mfips = evaluator_ranking(pred_df)

            maes_mf.append(mae_mf)
            mses_mf.append(mse_mf)
            maes_expomf.append(mae_expomf)

            mses_expomf.append(mse_expomf)
            maes_mfips.append(mae_mfips)
            mses_mfips.append(mse_mfips)

            ndcgs_mf.append(ndcg_mf)
            ndcgs_mfips.append(ndcg_mfips)
            ndcgs_expomf.append(ndcg_expomf)

            logging.info('Finished one run')
            
        logging.info('Finished all runs')
        logging.info('Results for Synthetic dataset are')

        logging.info("MAE_MF: mean %f +- std %f"%(np.mean(maes_mf), np.std(maes_mf)))
        logging.info("MSE_MF: mean %f +- std %f"%(np.mean(mses_mf), np.std(mses_mf)))
        logging.info("MAE_MFIPS: mean %f +- std %f"%(np.mean(maes_mfips), np.std(maes_mfips)))
        logging.info("MSE_MFIPS: mean %f +- std %f"%(np.mean(mses_mfips), np.std(mses_mfips)))
        logging.info("MAE_expoMF: mean %f +- std %f"%(np.mean(maes_expomf), np.std(maes_expomf)))
        logging.info("MSE_expoMF: mean %f +- std %f"%(np.mean(mses_expomf), np.std(mses_expomf)))
        logging.info("NDCG@3_MF: mean %f +- std %f"%(np.mean(ndcgs_mf), np.std(ndcgs_mf)))
        logging.info("NDCG@3_MFIPS: mean %f +- std %f"%(np.mean(ndcgs_mfips), np.std(ndcgs_mfips)))
        logging.info("NDCG@3_ExpoMF: mean %f +- std %f"%(np.mean(ndcgs_expomf), np.std(ndcgs_expomf)))

        logging.info("MAE student t-test diff %f p-value std %f"%((stats.ttest_rel(maes_mf, maes_mfips)[0]), \
                        stats.ttest_rel(maes_mf, maes_mfips)[1]))
        logging.info("MSE student t-test diff %f p-value std %f"%((stats.ttest_rel(mses_mf, mses_mfips)[0]), \
                        stats.ttest_rel(mses_mf, mses_mfips)[1]))
        # run = 'COAT' + '\t' + 'MF' + '\t' + str(ulf_dim) + '\t' + str(reg_param) + '\t' + str(mse_mf) + '\n'
        # cross_val_file.write(run)
        # run = 'COAT' + '\t' + 'MF-IPS' + '\t' + str(ulf_dim) + '\t' + str(reg_param) + '\t' + str(mse_mf_ips) + '\n'
        # cross_val_file.write(run)
    