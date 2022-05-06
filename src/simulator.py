
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
        logging.info('Coat data simulater')
        # Getting the corresponding train test split and the propensity matrix
        train_ua_df, test_ua_df, prop_ua_df = res[2], res[3], res[4]
        # YOUR CODE for USING THE SIMULATED COAT DATA FOLLOWS

        
    elif args.data == 'yahoo':
        log_file_name = './logs/pref_elicit_main_%d.log'%embed_dim 

        logging.basicConfig(filename=log_file_name, filemode='w', format='%(name)s - %(levelname)s - %(message)s', \
                level=logging.INFO)
        res = load_yahoo()
        assert len(res) == 6
        logging.info('Yahoo data simulated')
        # YOUR CODE for USING THE SIMULATED Yahoo DATA FOLLOWS
        train_ua_df, test_ua_df, prop_ua_df = res[2], res[3], res[4]
        

    else:
        log_file_name = './logs/pref_elicit_main_syn%f.log'%(args.alpha)

        logging.basicConfig(filename=log_file_name, filemode='w', format='%(name)s - %(levelname)s - %(message)s', \
                level=logging.INFO)
        res = load_synthetic(args.alpha)
        assert len(res) == 3
        logging.info('Fully synthetic data simulated')
        train_ua_df, test_ua_df, prop_ua_df = res[0], res[1], res[2]
        # YOUR CODE FOR USING THE FULLY SYNTHETIC DATA FOLLOWS
    