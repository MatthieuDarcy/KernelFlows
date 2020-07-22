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

def nabla_rational_quadratic(X, parameters):
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

def nabla_linear_gaussian(X, parameters):
    theta = 0
    matrix_diff_norm = norm_matrix(X, X)
    for i in range(parameters.shape[1]):
        # print("beta", parameters[1, i])
        # print("sigma", parameters[0, i])
        theta = theta + parameters[1, i]**2*np.exp(-matrix_diff_norm / (2* parameters[0, i]**2))
    
    grad_matrix = np.zeros((2, parameters.shape[1], X.shape[0], X.shape[0]))
    
    for i in range(parameters.shape[1]):
        sigma = parameters[0, i]
        beta = parameters[1, i]
        grad_matrix[0, i] = beta**2*np.multiply(matrix_diff_norm, np.exp(-matrix_diff_norm / (2* sigma**2)))/(sigma**3)
        grad_matrix[1, i] = 2*beta*np.exp(-matrix_diff_norm/ (2* sigma**2))
        
    
    return grad_matrix, theta

def nabla_sigmoid(X, parameters):
    alpha = parameters[0]
    beta = parameters[1]
    matrix_inner = inner_matrix(X, X)
    theta = np.tanh(alpha *matrix_inner + beta)
    
    grad_matrix = np.zeros((2, X.shape[0], X.shape[0]))
    
    grad_1 = np.multiply(matrix_inner, 1/(np.cosh(alpha* matrix_inner + beta)**2))
    grad_2 = 1/(np.cosh(alpha* matrix_inner + beta)**2)
    
    grad_matrix[0] = grad_1
    grad_matrix[1] = grad_2
    
    return grad_matrix, theta
                           
nabla_dic = {"RBF": nabla_RBF, "rational quadratic" : nabla_rational_quadratic, 
             "gaussian multi": nabla_linear_gaussian, "sigmoid": nabla_sigmoid}
