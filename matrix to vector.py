# -*- coding: utf-8 -*-
"""
Created on Mon May 12 19:46:42 2025

@author: 20808
"""


import numpy as np
import scipy.io
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

if __name__ == '__main__':
    data_path = './Data/NYU116.mat'
    mat_data = scipy.io.loadmat(data_path)
    
    raw_data = np.array(mat_data['AAL'])[0]
    data = []
    for item in raw_data:
        data.append(item)
        
    data_origin = np.array(data)
    data_transposed = np.transpose(data_origin, (0, 2, 1))
    data_transposed = np.array(np.stack(data_transposed, axis=0), np.float32)
    
    labels = np.array(mat_data['lab'][0])
    
    # Convert to Torch tensor
    data = torch.tensor(data_transposed)
    
    num_samples, height, width = data.shape
    result = torch.zeros(num_samples, height, height)
    
    BFN_one = []
    for i in range(num_samples):
        r = np.corrcoef(data[i])
        BFN_one.append(r)
        

        
   
    BFN_two = []
    
    for i in range(len(BFN_one)):
        
        r = BFN_one[i]
    
        
        new_r = np.corrcoef(r)
    
        
        BFN_two.append(new_r)
        



n = 13  


BFN_high = []


for matrix in BFN_two:
    
    current_matrix = matrix
    
    
    for _ in range(n):
        current_matrix = np.corrcoef(current_matrix)
    
    
    BFN_high.append(current_matrix)
    
    
def upper_triangle_elements_without_diagonal(x):
    
    x = np.array(x)
    
    
    upper_triangle = np.triu(x)
    
    
    upper_triangle_without_diag = upper_triangle - np.diag(np.diag(upper_triangle))
    
    
    feature = upper_triangle_without_diag[upper_triangle_without_diag != 0]
    
    return feature




yinyanren = upper_triangle_elements_without_diagonal(BFN_one[0])
yinyanren = upper_triangle_elements_without_diagonal(BFN_one[1])
yinyanren = upper_triangle_elements_without_diagonal(BFN_one[2])
yinyanren = upper_triangle_elements_without_diagonal(BFN_one[3])


upper_triangle_features = [upper_triangle_elements_without_diagonal(matrix) for matrix in BFN_high]

np.save('BFN_fifteen_features.npy', upper_triangle_features)

