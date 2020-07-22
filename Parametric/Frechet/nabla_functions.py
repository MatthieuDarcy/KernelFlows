# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:07:43 2020

@author: matth
"""


from matrix_operations import norm_matrix
from matrix_operations import inner_matrix

import numpy as np
#%%
def nabla_RBF(X, parameters): 
    sigma = parameters[0]
    
    matrix_norm = norm_matrix(X, X)
    theta = np.exp(-matrix_norm/(2*sigma**2))
    grad_matrix = (np.multiply(matrix_norm, theta))/(sigma**3)
    return grad_matrix, theta

def nabla_inverse_power(X, parameters):
    alpha = parameters[0]
    beta = parameters[1]
    matrix_norm = norm_matrix(X, X)
    epsilon = 0.00001
    theta = (beta**2 + matrix_norm)**(-(alpha + epsilon))
    
    grad_matrix = np.zeros((2, X.shape[0], X.shape[0]))
    
    grad_1 = - np.multiply(theta, np.log(beta**2 + matrix_norm))
    grad_2 = -2*(alpha+epsilon)*beta*(beta**2 + matrix_norm)**(-(alpha+epsilon)-1)
    
    grad_matrix[0] = grad_1
    grad_matrix[1] = grad_2
    
    return grad_matrix, theta
    
nabla_dic = {"RBF": nabla_RBF, "inverse power" : nabla_inverse_power}
