

import numpy as np
import os
import pandas as pd
#from typing import Tuple
import pdb


def transform_yahoo():
    """
    load_yahoo Loads yahoo dataset

    Returns:
        Tuple[np.ndarray, np.ndarray]: train, test dataset
    """    
    
    cols = ['uid', 'pid', 'rating']
    dir_path = './data/raw/yahoo'
    train_df = pd.read_csv(os.path.join(dir_path, 'ydata-ymusic-rating-study-v1_0-train.txt'),\
                    delimiter='\t', names=cols)
    
    test_df = pd.read_csv(os.path.join(dir_path, 'ydata-ymusic-rating-study-v1_0-test.txt'),\
                    delimiter='\t', names=cols)
    np.random.seed(12345)
    msk = np.random.rand(len(test_df)) < 0.8
    test_df1 = test_df[msk]
    val_df = test_df[~msk]
    test_df1[['uid', 'pid', 'rating']].to_csv('./data/processed/eval_split.dat', sep='\t', \
                                         index=False, header=False)
    val_df[['uid', 'pid', 'rating']].to_csv('./data/processed/prop_split.dat', sep='\t', \
                                         index=False, header=False)
    

    val_df['uid1'] = val_df['uid'].apply(lambda x: 'u' + str(x))
    val_df['pid1'] = val_df['pid'].apply(lambda x: 'i' + str(x))

    val_df[['uid1', 'pid1', 'rating']].to_csv('./data/processed/bine_embed_data.dat', sep='\t', \
                                         index=False, header=False)
    



if __name__ == '__main__':
    transform_yahoo()



    

