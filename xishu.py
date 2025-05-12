# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 16:18:17 2024

@author: 20808
"""
from sklearn.decomposition import SparseCoder
import numpy as np
import os


def SR(self, data, lam, save_path, Bi_threshold=None, Sp_Ratio=None):
    data = np.transpose(data, (0, 2, 1))
    Network = []
    for i in range(len(data)):
        signal = data[i].T
        sparse_coder = SparseCoder(dictionary=signal, transform_algorithm='lasso_lars', transform_alpha=lam)
        network = sparse_coder.transform(signal)
        network = (network + network.T) / 2
        Network.append(network)
    SR_matrices = np.array(Network)
    if Sp_Ratio is not None:
        SR_matrices = self.K_Sparsity(SR_matrices, Sp_Ratio)
        np.save(os.path.join(save_path, 'SR_Adjacency_Matrix_sparse_' + str(lam) + '_Sparsity_' + str(Sp_Ratio) + '%.npy'), SR_matrices)
    if Bi_threshold is not None:
        SR_matrices = self.T_Binarization(SR_matrices, Bi_threshold)
        np.save(os.path.join(save_path, 'SR_Adjacency_Matrix_sparse_' + str(lam) + '_Binarization_' + str(Bi_threshold) + '%.npy'), SR_matrices)
    else: np.save(os.path.join(save_path, 'SR_Adjacency_Matrix_sparse_' + str(lam) + '.npy'), SR_matrices)
    return SR_matrices