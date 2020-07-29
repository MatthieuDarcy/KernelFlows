# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:40:34 2020

@author: matth
"""

import numpy as np

from matrix_operations import norm_matrix
from matrix_operations import inner_matrix


#%%
    
# This function computes the pairwise difference between elements of X (used for the nabla of any kernel with a norm)
def pairwise_diff(X):
    
    diff_matrix = np.zeros((X.shape[0], X.shape[1], X.shape[0]))
    for element in range(X.shape[0]):
        for dimension in range(X.shape[1]):    
            diff_matrix[element, dimension] =  X[element, dimension] - X[:, dimension] 
    return diff_matrix

"""
For nabla, must return a matrix of dimension (n, d,n) where n i s the number of points, 
d is the dimension of each point.
The first dimension corresponds to different points. The third dimension correponds to 
a specific point, the derivative wrt to every other point.
"""
# Nabla of the RBF kernel    
def nabla_RBF(X,  parameters):
    # Computing the Kernel matrix
    matrix_norm = norm_matrix(X, X)
    sigma = parameters[0]
    batch_matrix =  np.exp(-matrix_norm /(2 *sigma**2))
     
    
    # Stack of matrices
    K_matrix = np.zeros((X.shape[0], X.shape[1], X.shape[0]))
    for i in range(X.shape[1]):
        K_matrix[:, i, :] = batch_matrix
        
    # The pairwize differences between elements    
    matrix_diff = pairwise_diff(X)
    derivative_matrix = -np.multiply(matrix_diff, K_matrix)/(sigma**2)
    #print(derivative_matrix[0])
    return derivative_matrix, batch_matrix

def nabla_inverse_power(X, parameters):
    # Computing the Kernel matrix
    matrix_norm = norm_matrix(X, X)
    alpha = parameters[0]
    beta = parameters[1]
    first_derivative_matrix = (beta + matrix_norm)**(-alpha-1)
    batch_matrix = (beta + matrix_norm)**(-alpha)
    
    # Stack of matrices
    K_matrix = np.zeros((X.shape[0], X.shape[1], X.shape[0]))
    for i in range(X.shape[1]):
        K_matrix[:, i, :] = first_derivative_matrix
    
    # The pairwize differences between elements    
    matrix_diff = pairwise_diff(X)
    derivative_matrix = -2*alpha*np.multiply(matrix_diff, K_matrix)

    return derivative_matrix, batch_matrix

def nabla_polynomial(X,parameters):
    d = parameters[0]
    c = parameters[1]
    
    matrix_inner = inner_matrix(X,X)
    batch_matrix = (matrix_inner + c)**d
    
    # Stack of matrices
    K_matrix = np.zeros((X.shape[0], X.shape[1], X.shape[0]))
    for i in range(X.shape[1]):
        K_matrix[:, i, :] = batch_matrix
    
    
    derivative_matrix = d*np.multiply(np.transpose(X), K_matrix)
    
    for element in range(X.shape[0]):
        derivative_matrix[element, :, element] *= 2
    return derivative_matrix, batch_matrix
           
nabla_dic = {"RBF" : nabla_RBF, "inverse power" : nabla_inverse_power, "polynomial" : nabla_polynomial}
