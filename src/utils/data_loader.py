
import numpy as np
import os
import pandas as pd
from typing import Tuple
import pdb
from sklearn import preprocessing 
from decimal import Decimal


def load_coat() -> Tuple[np.ndarray, np.ndarray]:
    """
    load_coat: Loads coat dataset

    Returns:
        List[np.ndarray, np.ndarray, np.ndarray]: returns train dataframe, test dataframe joined with item feats.
    """    
    train_raw_matrix = np.loadtxt(os.path.join('./data/raw/coat', 'train.ascii'))
    test_raw_matrix = np.loadtxt(os.path.join('./data/raw/coat', 'test.ascii'))
    item_feats = np.loadtxt(os.path.join('./data/raw/coat/user_item_features', 'item_features.ascii'))
    train_df = pd.DataFrame(
        {'uid': np.nonzero(train_raw_matrix)[0].tolist(),
        'pid': np.nonzero(train_raw_matrix)[1].tolist(),
        'rating': train_raw_matrix[np.nonzero(train_raw_matrix)]
        })
    
    test_df = pd.DataFrame(
        {'uid': np.nonzero(test_raw_matrix)[0].tolist(),
        'pid': np.nonzero(test_raw_matrix)[1].tolist(),
        'rating': test_raw_matrix[np.nonzero(test_raw_matrix)]
        })

    col_names = []
    for line in open(os.path.join('./data/raw/coat/user_item_features', 'item_features_map.txt'), 'r'):
        col_names.append(line.strip())
    
    col_name_df = {}
    for i, col_name in enumerate(col_names):
        col_name_df[col_name] = item_feats[:, i]
        col_name_df['pid'] = range(0, item_feats.shape[0])
    
    item_feats_df = pd.DataFrame(col_name_df)
    train_df_item = train_df.merge(item_feats_df, on='pid')
    test_df_item = test_df.merge(item_feats_df, on='pid')

    # Generating user * attribute matrix
    col_names_idx = dict(zip(col_names, range(len(col_names))))
    user_attr_train = np.zeros((train_raw_matrix.shape[0], len(col_names)))
    user_attr_train_count = np.zeros((train_raw_matrix.shape[0], len(col_names)))

    user_attr_test = np.zeros((test_raw_matrix.shape[0], len(col_names)))
    user_attr_test_count = np.zeros((test_raw_matrix.shape[0], len(col_names)))

    for j, row in train_df_item.iterrows():
        for col_name, idx in col_names_idx.items():
            if int(row[col_name]) == 1:
                user_attr_train[int(row['uid']), idx] += row['rating']
    
    for j, row in train_df_item.iterrows():
        for col_name, idx in col_names_idx.items():
            if int(row[col_name]) == 1:
                user_attr_train_count[int(row['uid']), idx] += 1
            
    for j, row in test_df_item.iterrows():
        for col_name, idx in col_names_idx.items():
            if int(row[col_name]) == 1:
                user_attr_test[int(row['uid']), idx] += row['rating']
    
    for j, row in test_df_item.iterrows():
        for col_name, idx in col_names_idx.items():
            if int(row[col_name]) == 1:
                user_attr_test_count[int(row['uid']), idx] += 1

    # replace count of ratings with average rating
    user_attr_train /= user_attr_train_count
    user_attr_test /= user_attr_test_count
    # replace division with 0 with -1
    user_attr_train[np.isnan(user_attr_train)] = 0
    user_attr_test[np.isnan(user_attr_test)] = 0
    # Convert float values to nearest int
    user_attr_train = np.ceil(user_attr_train)
    user_attr_test = np.ceil(user_attr_test)

    # train & test user * attribute dataframe
    train_ua_df = pd.DataFrame(
    {'uid': np.nonzero(user_attr_train)[0].tolist(),
     'pid': np.nonzero(user_attr_train)[1].tolist(),
     'rating': user_attr_train[np.nonzero(user_attr_train)]
    })

    # np.random.seed(12345)
    # msk = np.random.rand(len(train_ua_df)) < 0.9
    
    # train_split_df = train_ua_df[msk]
    # valid_split_df = train_ua_df[~msk]


    test_ua_df = pd.DataFrame(
    {'uid': np.nonzero(user_attr_test)[0].tolist(),
     'pid': np.nonzero(user_attr_test)[1].tolist(),
     'rating': user_attr_test[np.nonzero(user_attr_test)]
    })
    np.random.seed(12345)
    msk = np.random.rand(len(test_ua_df)) < 0.8
    
    eval_split_df = test_ua_df[msk]
    prop_split_df = test_ua_df[~msk]

    return (train_df_item, test_df_item, train_ua_df, eval_split_df, prop_split_df, item_feats_df)

def load_synthetic(alpha) -> Tuple[np.ndarray, np.ndarray]:
    """
    load_synthetic: loads fully synthetic dataset

    Returns:
        List[np.ndarray, np.ndarray, np.ndarray]: returns train dataframe, test dataframe joined with item feats.
    """  
    if int(alpha) == 1:  
        dir_path = './data/raw/simulated/alpha'  + str(1.0)
    else:
        dir_path = './data/raw/simulated/alpha'  + str(Decimal(alpha).normalize())
    train_raw_matrix = np.loadtxt(os.path.join(dir_path, 'train.ascii'))
    test_raw_matrix = np.loadtxt(os.path.join(dir_path, 'test.ascii'))
    #item_feats = np.loadtxt(os.path.join('./data/raw/coat/user_item_features', 'item_features.ascii'))
    train_df = pd.DataFrame(
        {'uid': np.nonzero(train_raw_matrix)[0].tolist(),
        'pid': np.nonzero(train_raw_matrix)[1].tolist(),
        'rating': train_raw_matrix[np.nonzero(train_raw_matrix)]
        })
    
    test_df = pd.DataFrame(
        {'uid': np.nonzero(test_raw_matrix)[0].tolist(),
        'pid': np.nonzero(test_raw_matrix)[1].tolist(),
        'rating': test_raw_matrix[np.nonzero(test_raw_matrix)]
        })
    
    np.random.seed(12345)
    msk = np.random.rand(len(test_df)) < 0.8
    
    eval_split_df = test_df[msk]
    prop_split_df = test_df[~msk]

    return (train_df, eval_split_df, prop_split_df)

def load_yahoo() -> Tuple[np.ndarray, np.ndarray]:
    """
    load_yahoo Loads yahoo dataset

    Returns:
        Tuple[np.ndarray, np.ndarray]: train, test dataset
    """    
    cols = ['uid', 'pid', 'rating']
    dir_path = './data/raw/yahoo'
    train_df = pd.read_csv(os.path.join(dir_path, 'ydata-ymusic-rating-study-v1_0-train.txt'),\
                    delimiter='\t', names=cols)
    test_dir_path = './data/processed/'
    test_df = pd.read_csv(os.path.join(test_dir_path, 'eval_split.dat'),\
                    delimiter='\t', names=cols)
    prop_df = pd.read_csv(os.path.join(test_dir_path, 'prop_split.dat'),\
                    delimiter='\t', names=cols)

    # train_raw_matrix = np.loadtxt(os.path.join('./data/raw/coat', 'train.ascii'))
    # test_raw_matrix = np.loadtxt(os.path.join('./data/raw/coat', 'test.ascii'))
    item_feats = np.loadtxt('./data/processed/item_attr_matrix.txt')
    # train_df = pd.DataFrame(
    #     {'uid': np.nonzero(train_raw_matrix)[0].tolist(),
    #     'pid': np.nonzero(train_raw_matrix)[1].tolist(),
    #     'rating': train_raw_matrix[np.nonzero(train_raw_matrix)]
    #     })
    
    # test_df = pd.DataFrame(
    #     {'uid': np.nonzero(test_raw_matrix)[0].tolist(),
    #     'pid': np.nonzero(test_raw_matrix)[1].tolist(),
    #     'rating': test_raw_matrix[np.nonzero(test_raw_matrix)]
    #     })

    col_names = []
    n_clusts = item_feats.shape[1]
    n_users = np.unique(train_df['uid']).shape[0]
    for i in range(n_clusts):
        col_names.append('f'  + str(i))
    
    col_name_df = {}
    for i, col_name in enumerate(col_names):
        col_name_df[col_name] = item_feats[:, i]
        col_name_df['pid'] = range(0, item_feats.shape[0])
    
    item_feats_df = pd.DataFrame(col_name_df)
    train_df_item = train_df.merge(item_feats_df, on='pid')
    test_df_item = test_df.merge(item_feats_df, on='pid')
    prop_df_item = prop_df.merge(item_feats_df, on='pid')

    # Generating user * attribute matrix
    col_names_idx = dict(zip(col_names, range(len(col_names))))
    user_attr_train = np.zeros((n_users, len(col_names)))
    user_attr_train_count = np.zeros((n_users, len(col_names)))

    user_attr_test = np.zeros((n_users, len(col_names)))
    user_attr_test_count = np.zeros((n_users, len(col_names)))

    user_attr_prop = np.zeros((n_users, len(col_names)))
    user_attr_prop_count = np.zeros((n_users, len(col_names)))

    
    for j, row in train_df_item.iterrows():
        for col_name, idx in col_names_idx.items():
            if int(row[col_name]) == 1:
                user_attr_train[int(row['uid'])-1, idx] += row['rating']
    
    for j, row in train_df_item.iterrows():
        for col_name, idx in col_names_idx.items():
            if int(row[col_name]) == 1:
                user_attr_train_count[int(row['uid'])-1, idx] += 1
            
    for j, row in test_df_item.iterrows():
        for col_name, idx in col_names_idx.items():
            if int(row[col_name]) == 1:
                user_attr_test[int(row['uid'])-1, idx] += row['rating']
    
    for j, row in test_df_item.iterrows():
        for col_name, idx in col_names_idx.items():
            if int(row[col_name]) == 1:
                user_attr_test_count[int(row['uid'])-1, idx] += 1

    for j, row in prop_df_item.iterrows():
        for col_name, idx in col_names_idx.items():
            if int(row[col_name]) == 1:
                user_attr_prop[int(row['uid'])-1, idx] += row['rating']
    
    for j, row in prop_df_item.iterrows():
        for col_name, idx in col_names_idx.items():
            if int(row[col_name]) == 1:
                user_attr_prop_count[int(row['uid'])-1, idx] += 1


    # replace count of ratings with average rating
    user_attr_train /= user_attr_train_count
    user_attr_test /= user_attr_test_count
    user_attr_prop /= user_attr_prop_count
    # replace division with 0 with -1
    user_attr_train[np.isnan(user_attr_train)] = 0
    user_attr_test[np.isnan(user_attr_test)] = 0
    user_attr_prop[np.isnan(user_attr_prop)] = 0
    # Convert float values to nearest int
    user_attr_train = np.ceil(user_attr_train)
    user_attr_test = np.ceil(user_attr_test)
    user_attr_prop = np.ceil(user_attr_prop)

    # train & test user * attribute dataframe
    train_ua_df = pd.DataFrame(
    {'uid': np.nonzero(user_attr_train)[0].tolist(),
     'pid': np.nonzero(user_attr_train)[1].tolist(),
     'rating': user_attr_train[np.nonzero(user_attr_train)]
    })

    # np.random.seed(12345)
    # msk = np.random.rand(len(train_ua_df)) < 0.9
    
    # train_split_df = train_ua_df[msk]
    # valid_split_df = train_ua_df[~msk]


    test_ua_df = pd.DataFrame(
    {'uid': np.nonzero(user_attr_test)[0].tolist(),
     'pid': np.nonzero(user_attr_test)[1].tolist(),
     'rating': user_attr_test[np.nonzero(user_attr_test)]
    })

    prop_ua_df = pd.DataFrame(
    {'uid': np.nonzero(user_attr_prop)[0].tolist(),
     'pid': np.nonzero(user_attr_prop)[1].tolist(),
     'rating': user_attr_prop[np.nonzero(user_attr_prop)]
    })
    
    return (train_df_item, test_df_item, train_ua_df, test_ua_df, prop_ua_df, item_feats_df)



if __name__ == '__main__':
    res = load_coat()
    res = load_yahoo()



    

