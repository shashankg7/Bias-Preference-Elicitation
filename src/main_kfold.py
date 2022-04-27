
import argparse
from numpy.core.numeric import cross
import yaml
import warnings
import os
import logging


import tensorflow as tf
import numpy as np                                                         
from utils.data_loader import load_coat, load_synthetic, load_yahoo
from trainer import train_mf, train_mf_kfold, train_expoMF_kfold
from evaluate.metrics import evaluator
import argparse


possible_model_names = ['cold-start', 'casual-user']
data = ['coat', 'yahoo', 'synthetic']


parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', type=str,
                    required=True)
parser.add_argument('--user_type', '-u', type=str,
                    choices=possible_model_names, required=False)
parser.add_argument('-job_id', type=int,
                 required=True)
parser.add_argument('-user_lf', type=int,
                 required=True)
parser.add_argument('-reg_param', type=float,
                 required=True)
parser.add_argument('--alpha', type=float,
                 required=False)


args = parser.parse_args()
log_file_name = './logs/pref_elicit_%d.log'%(args.job_id)

logging.basicConfig(filename=log_file_name, filemode='w', format='%(name)s - %(levelname)s - %(message)s', \
                level=logging.INFO)
# else:
#     logging.basicConfig(filename=log_file_name, filemode='a+', format='%(name)s - %(levelname)s - %(message)s', \
#                     level=logging.INFO)

config_file_name = './reports/results/cross_val_results_%d.txt'%(args.job_id)
# if not os.path.isfile(config_file_name):
cross_val_file = open(config_file_name, 'w')
# else:
#     cross_val_file = open(config_file_name, 'a+')
headers = 'Dataset' + '\t' + 'Method' + '\t' + 'lf_dim' + '\t' + 'reg_param' + '\t' + 'MSE' + '\n'
cross_val_file.write(headers)
# parser.add_argument('--data_type', '-d', type=str,
#                     choices=data, required=True)

if __name__ == '__main__':
    print("starting code")
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:

    config = yaml.safe_load(open('./config/consts.yaml', 'rb'))
    print(config)
    ## Code for active user's case
    
    if args.data == 'coat':
        res = load_coat()
        assert len(res) == 6
        logging.info('Coat data loaded, starting cross-valid')

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
        ulf_dim = args.user_lf
        reg_param = args.reg_param
        ndcg_expomf = train_expoMF_kfold(train_ua_df, test_ua_df, prop_ua_df, \
                    ulf_dim=ulf_dim, reg_param=reg_param, \
                    ips=0, config=config, job_id=args.job_id, logging=logging)
        ndcg_mf = train_mf_kfold(train_ua_df, test_ua_df, prop_ua_df, \
                    ulf_dim=ulf_dim, ilf_dim=ulf_dim, reg_param=reg_param, \
                    ips=0, config=config, job_id=args.job_id, logging=logging)
        ndcg_mf_ips = train_mf_kfold(train_ua_df, test_ua_df, prop_ua_df, \
                        ulf_dim=ulf_dim, ilf_dim=ulf_dim, reg_param=reg_param, \
                            ips=1, config=config, job_id=args.job_id, logging=logging)
        logging.info('Finished one cycle of hyper-parm')
        logging.info('NDCG@3 with %d lf dim and %f reg_parm is %f'%(ulf_dim, reg_param, ndcg_mf))
        logging.info('NDCG@3 with IPS with %d lf dim and %f reg_parm is %f'%(ulf_dim, reg_param, ndcg_mf_ips))
        run = 'COAT' + '\t' + 'MF' + '\t' + str(ulf_dim) + '\t' + str(reg_param) + '\t' + str(ndcg_mf) + '\n'
        cross_val_file.write(run)
        run = 'COAT' + '\t' + 'MF-IPS' + '\t' + str(ulf_dim) + '\t' + str(reg_param) + '\t' + str(ndcg_mf_ips) + '\n'
        cross_val_file.write(run)
        run = 'COAT' + '\t' + 'ExpoMF' + '\t' + str(ulf_dim) + '\t' + str(reg_param) + '\t' + str(ndcg_expomf) + '\n'
        cross_val_file.write(run)

        
    elif args.data == 'yahoo':
        res = load_yahoo()
        assert len(res) == 6
        logging.info('Yahoo data loaded, starting cross-valid')

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
        ulf_dim = args.user_lf
        reg_param = args.reg_param
        ndcg_expomf = train_expoMF_kfold(train_ua_df, test_ua_df, prop_ua_df ,\
                    ulf_dim=ulf_dim, reg_param=reg_param, \
                    ips=0, config=config, job_id=args.job_id, logging=logging)
        ndcg_mf = train_mf_kfold(train_ua_df, test_ua_df, prop_ua_df, \
                    ulf_dim=ulf_dim, ilf_dim=ulf_dim, reg_param=reg_param, \
                    ips=0, config=config, job_id=args.job_id, logging=logging)
        ndcg_mf_ips = train_mf_kfold(train_ua_df, test_ua_df, prop_ua_df, \
                        ulf_dim=ulf_dim, ilf_dim=ulf_dim, reg_param=reg_param, \
                            ips=1, config=config, job_id=args.job_id, logging=logging)
        logging.info('Finished one cycle of hyper-parm')
        
        run = 'Yahoo' + '\t' + 'MF' + '\t' + str(ulf_dim) + '\t' + str(reg_param) + '\t' + str(ndcg_mf) + '\n'
        cross_val_file.write(run)
        run = 'Yahoo' + '\t' + 'MF-IPS' + '\t' + str(ulf_dim) + '\t' + str(reg_param) + '\t' + str(ndcg_mf_ips) + '\n'
        cross_val_file.write(run)
        run = 'Yahoo' + '\t' + 'ExpoMF' + '\t' + str(ulf_dim) + '\t' + str(reg_param) + '\t' + str(ndcg_expomf) + '\n'
        cross_val_file.write(run)

    
    elif args.data == 'synthetic':
        res = load_synthetic(args.alpha)
        assert len(res) == 3
        logging.info('Synthetic data loaded, starting cross-valid')

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
        ulf_dim = args.user_lf
        reg_param = args.reg_param
        ndcg_expomf = train_expoMF_kfold(train_ua_df, test_ua_df, prop_ua_df, \
                    ulf_dim=ulf_dim, reg_param=reg_param, \
                    ips=0, config=config, job_id=args.job_id, logging=logging)
        ndcg_mf = train_mf_kfold(train_ua_df, test_ua_df, prop_ua_df, \
                    ulf_dim=ulf_dim, ilf_dim=ulf_dim, reg_param=reg_param, \
                    ips=0, config=config, job_id=args.job_id, logging=logging)
        ndcg_mf_ips = train_mf_kfold(train_ua_df, test_ua_df, prop_ua_df, \
                        ulf_dim=ulf_dim, ilf_dim=ulf_dim, reg_param=reg_param, \
                            ips=1, config=config, job_id=args.job_id, logging=logging)
        logging.info('Finished one cycle of hyper-parm')
        logging.info('NDCG@3 with %d lf dim and %f reg_parm is %f'%(ulf_dim, reg_param, ndcg_mf))
        logging.info('NDCG@3 with IPS with %d lf dim and %f reg_parm is %f'%(ulf_dim, reg_param, ndcg_mf_ips))
        run = 'Synthetic' + '\t' + 'MF' + '\t' + str(ulf_dim) + '\t' + str(reg_param) + '\t' + str(ndcg_mf) + '\n'
        cross_val_file.write(run)
        run = 'Synthetic' + '\t' + 'MF-IPS' + '\t' + str(ulf_dim) + '\t' + str(reg_param) + '\t' + str(ndcg_mf_ips) + '\n'
        cross_val_file.write(run)
        run = 'Synthetic' + '\t' + 'ExpoMF' + '\t' + str(ulf_dim) + '\t' + str(reg_param) + '\t' + str(ndcg_expomf) + '\n'
        cross_val_file.write(run)

        

    
    ## Code for cold-start user's case
    