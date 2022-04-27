

import numpy as np
import scipy
from sklearn.mixture import BayesianGaussianMixture as GMM
import pdb

def cluster_items(n_clusters=150) -> None:
    """
    cluster_items Generate item clusters and corresponding item attribute matrix 

    Args:
        n_clusters (int, optional): [description]. Defaults to 35.
    """    
    item_vec_obj = np.loadtxt('./models/embeddings/vectors_v.dat', dtype=str)
    item_indices = list(map(lambda a:int(a[1:]),  item_vec_obj[:, 0]))
    item_vecs = np.zeros((item_vec_obj.shape[0], item_vec_obj.shape[1]-1))
    item_attr_matrix = np.zeros((item_vec_obj.shape[0], n_clusters), dtype=np.int32)
    for i, idx in enumerate(item_indices):
        item_vecs[idx-1, :] =  np.asarray(item_vec_obj[i, 1:], dtype=np.float64)
    gm = GMM(n_components=n_clusters, random_state=0).fit(item_vecs)
    item_clusts = gm.predict(item_vecs)
    for i, clus in enumerate(item_clusts):
        item_attr_matrix[i, clus] = 1
    np.savetxt('./data/processed/item_attr_matrix.txt', item_attr_matrix)
    #pdb.set_trace()

if __name__ == '__main__':
    cluster_items()
    
