# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:32:37 2020

@author: matth
"""


import autograd.numpy as np
from matrix_operations_autograd import norm_matrix
from matrix_operations_autograd import inner_matrix


#%%

""" In this section we define various kernels. Warning, not all of them work 
at the moment, the most reliable one is the RBF kernel. Note that currently the 
laplacian kernel does not work"""
        

# Define the RBF Kernel. Takes an array of parameters, returns a value
def kernel_RBF(matrix_1, matrix_2, parameters):
    matrix = norm_matrix(matrix_1, matrix_2)
    sigma = parameters[0]
    K =  np.exp(-matrix/ (2* sigma**2))
    
    return K


# do not use right now
def kernel_laplacian(matrix_1, matrix_2, parameters):
    gamma = parameters[0]
    matrix = norm_matrix(matrix_1, matrix_2)
    K =  np.exp(-matrix * gamma)
    return K

def kernel_sigmoid(matrix_1, matrix_2, parameters):
    alpha = parameters[0]
    beta = parameters[1]
    matrix = inner_matrix(matrix_1, matrix_2)
    K = np.tanh(alpha *matrix + beta)
    return K

def kernel_inverse_power(matrix_1, matrix_2, parameters):
    alpha = parameters[0]
    beta = parameters[1]
    epsilon = 0.0001
    matrix = norm_matrix(matrix_1, matrix_2)
    return (beta**2 + matrix)**(-(alpha+ epsilon))

def kernel_inverse_power_alpha(matrix_1, matrix_2, parameters):
    alpha = parameters[0]
    beta = 1.0
    epsilon = 0.0001
    matrix = norm_matrix(matrix_1, matrix_2)
    return (beta**2 + matrix)**(-(alpha+ epsilon))

def kernel_inverse_multiquad(matrix_1, matrix_2, parameters):
    beta = parameters[0]
    gamma = parameters[1]
    matrix = norm_matrix(matrix_1, matrix_2)
    return (beta**2 + gamma*matrix)**(-1/2)

def kernel_cauchy(matrix_1, matrix_2, parameters):
    sigma = parameters[0]
    matrix = norm_matrix(matrix_1, matrix_2)
    return 1/(1 + matrix/sigma**2)

def kernel_quad(matrix_1, matrix_2, parameters):
    c = parameters[0]
    matrix = inner_matrix(matrix_1, matrix_2)
    K = (matrix+c) ** 2
    return K 

def kernel_poly(matrix_1, matrix_2, parameters):
    a = parameters[0]
    b = parameters[1]
    d = parameters[2]
    matrix = inner_matrix(matrix_1, matrix_2)
    K = (a * matrix + b) ** d
    return K 


def kernel_gaussian_linear(matrix_1, matrix_2, parameters):
    K = 0
    matrix = norm_matrix(matrix_1, matrix_2)
    for i in range(parameters.shape[1]):
        # print("beta", parameters[1, i])
        # print("sigma", parameters[0, i])
        K = K + parameters[1, i]**2*np.exp(-matrix / (2* parameters[0, i]**2))
    return K

"""A dictionnary containing the different kernels. If you wish to build a custom 
 kernel, add the function to the dictionnary.
"""
kernels_dic = {"RBF" : kernel_RBF,"poly": kernel_poly, "Laplacian": kernel_laplacian, 
               "Sigmoid": kernel_sigmoid, "inverse_power": kernel_inverse_power,
               "inverse_multiquad": kernel_inverse_multiquad, "quadratic" : kernel_quad,
               "poly": kernel_poly, "inverse_power_alpha": kernel_inverse_power_alpha,
               "gaussian multi": kernel_gaussian_linear}
