import pandas as pd
import seaborn as sns
import numpy as np
import os, sys, pdb
from scipy.special import expit
import matplotlib.pyplot as plt
import random, pickle
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from collections import Counter
from scipy import sparse
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--alpha', '-u', type=float, required=True)
args = parser.parse_args()

def get_dist(ratingM):
    counts = []
    for i in range(1, 6):
        counts.append(np.round(np.mean(ratingM==i), 3))
    return counts

n_dim = 15
num_users, num_items = 8000, 150
np.random.seed(42)
user_vec = np.random.randn(num_users, n_dim)
item_vec = np.random.randn(num_items, n_dim)

def bins(scores): # map score into ratings (1,2,3,4,5) with given prob
    scores_ = scores.flatten()
    prob = [0.526, 0.242, 0.144, 0.062, 0.026]
    assert sum(prob) == 1.0
    indices = np.argsort(scores_)
    indicators = np.cumsum(prob) * indices.shape[0]
    indicators = np.hstack(([0], indicators)).astype('int')
    # print(indicators)
    for i in range(1, indicators.shape[0]):
        scores_[indices[indicators[i-1]:indicators[i]]] = i
    return scores_.reshape(scores.shape)
ratingM = bins(np.dot(user_vec, item_vec.T)).astype('int')

p_r = get_dist(ratingM)
p_r_o = np.array([0.3, 0.1, 0.1, 0.2, 0.3], dtype=float)
p_o = 5.0e-2
print("p_r_o is ", p_r_o)
p_o_r = p_r_o * p_o / p_r
print("p_o_r is", p_o_r)
p_o_r = np.concatenate([[0.0], p_o_r], axis=0)
print(p_o_r)
probM = p_o_r[ratingM.astype(int)]
print(probM)

alpha = args.alpha
probM = alpha * probM + (1.0 - alpha) * p_o
observM = np.less(np.random.uniform(0, 1.0, (num_users, num_items)), probM)
print(np.sum(observM), np.sum(probM))
observRM = np.ma.masked_array(ratingM, mask=np.invert(observM), fill_value=0)
print(observRM)
print(observRM.filled())

counts = []
for i in range(1, 6):
    counts.append(np.round(np.sum(observRM==i) * 1.0 / np.sum(observM), 3))
print("the rating distribution of observed rating matrix is : \n", counts)

path = '/home/sgupta/projects/ConvRecSys/ultr_prefelicit/data/raw/simulated/'
path += 'alpha' + str(alpha) + '/'
if not os.path.exists(path):
    os.makedirs(path)
np.savetxt(path+"train.ascii", observRM.filled(), fmt='%d')
np.savetxt(path+"groundTruth.ascii", ratingM, fmt='%d')
np.savetxt(path+"propensities.ascii", probM, fmt='%.8f')

sampleP = np.random.uniform(size=(num_users, num_items))
K = 10
indices_y = (np.flip(np.argsort(sampleP, axis=1), axis=1)[:, :K]).flatten()
indices_x = np.repeat(np.arange(num_users), K).flatten()
sampleInd = np.zeros((num_users, num_items))
sampleInd[indices_x, indices_y] = 1
sampleInd = sampleInd.astype(bool)
testRM = np.where(sampleInd, ratingM, 0)
np.savetxt(path+"test.ascii", testRM, fmt='%d')

