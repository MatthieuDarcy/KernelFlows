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
    K =  np.exp(-matrix/ (sigma**2))
    
    return K


# do not use right now
def kernel_laplacian(matrix_1, matrix_2, parameters):
    gamma = parameters[0]
    matrix = np.sqrt(np.absolute(norm_matrix(matrix_1, matrix_2)))
    K =  np.exp(-matrix * gamma)
    return K

def kernel_sigmoid(matrix_1, matrix_2, parameters):
    alpha = parameters[0]
    beta = parameters[1]
    matrix = inner_matrix(matrix_1, matrix_2)
    K = np.tanh(alpha *matrix + beta)
    return K

def kernel_rational_quadratic(matrix_1, matrix_2, parameters):
    alpha = parameters[0]
    beta = parameters[1]
    epsilon = 1e-8
    matrix = norm_matrix(matrix_1, matrix_2)
    return (beta**2 + matrix)**(-(alpha**2+ epsilon))

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




def kernel_gaussian_linear(matrix_1, matrix_2, parameters):
    K = 0
    matrix = norm_matrix(matrix_1, matrix_2)
    for i in range(parameters.shape[1]):
        # print("beta", parameters[1, i])
        # print("sigma", parameters[0, i])
        K = K + parameters[1, i]**2*np.exp(-matrix / (2* parameters[0, i]**2))
    return K

def kernel_bernoulli(matrix_1, matrix_2, parameters):
    matrix = norm_matrix(matrix_1, matrix_2)
    alpha_0 = parameters[0]
    sigma_0 = parameters[1]
    
    alpha_1 = parameters[2]
    sigma_1 = parameters[3]
    
    K =  alpha_0**2*np.maximum(0, 1-matrix/(sigma_0**2))+ alpha_1**2*np.exp(-matrix/ (2* sigma_1**2))
    
    return K
    
def kernel_local_periodic(matrix_1, matrix_2, parameters):
    p = parameters[0]
    l = parameters[1]
    sigma = parameters[2]
    matrix = norm_matrix(matrix_1, matrix_2)
    
    K = np.exp(-np.sin(matrix*np.pi/p)**2/l**2)*np.exp(-matrix/sigma**2)
    return K

def kernel_rational_quadratic_gaussian(matrix_1, matrix_2, parameters):
    alpha = parameters[0]
    beta = parameters[1]
    sigma = parameters[2]
    delta_1 = parameters[3]
    delta_2 = parameters[4]
    c = parameters[5]
    
    epsilon = 1e-8
    matrix = norm_matrix(matrix_1, matrix_2)
    
    K = c + delta_1**2*((beta**2 + matrix)**(-(alpha**2 + epsilon))) + delta_2**2*np.exp(-matrix/ (2* sigma**2))
    
    return K
    
def kernel_periodic(matrix_1, matrix_2, parameters):
    p = parameters[0]
    l = parameters[1]
    matrix = norm_matrix(matrix_1, matrix_2)
    K = np.exp(-np.sin(matrix*np.pi/p)**2/l**2)
    
    return K



def kernel_constant(matrix_1, matrix_2, parameters):
    c = parameters[0]
    matrix = c**2*np.ones(shape = (matrix_1.shape[0], matrix_2.shape[0]))
    return matrix


def kernel_RBF_sigma(matrix_1, matrix_2, parameters):
    matrix = norm_matrix(matrix_1, matrix_2)
    sigma = parameters[0]
    var = parameters[1]
    K =  var**2*np.exp(-matrix/ (sigma**2))
    
    return K



def kernel_matern_25(matrix_1, matrix_2, param):
    
    l = param[0]
    sigma = param[1]
    
    r = np.sqrt(np.abs(norm_matrix(matrix_1, matrix_2)))
    
    exp = np.exp(-np.sqrt(5)*r/l)
    
    
    return sigma**2*(1 + np.sqrt(5)*r/l + 5/3*(r/l)**2)*exp

def kernel_rbf_poly(matrix_1, matrix_2, parameters):
    
    matrix_norm = norm_matrix(matrix_1, matrix_2)
    matrix_inner = inner_matrix(matrix_1, matrix_2)
    
    l = parameters[0]
    var = parameters[1]
    K_1 =  var**2*np.exp(-matrix_norm/ (l**2))
    
    d = np.floor(parameters[2])
    b = parameters[3]
    a = parameters[4]
    
    K_2 = a*(matrix_inner + b) ** d
    
    return K_1 + K_2

def kernel_rbf_linear(matrix_1, matrix_2, parameters):
    
    matrix_norm = norm_matrix(matrix_1, matrix_2)
    
    
    l = parameters[0]
    var = parameters[1]
    K_1 =  var*np.exp(-matrix_norm/ (l**2))
    
    c = parameters[2]
    b = parameters[3]
    a = parameters[4]
    matrix_inner = inner_matrix(matrix_1 +c, matrix_2 +c)
    
    K_2 = a*(matrix_inner + b)
    
    return K_1 + K_2
# def kernel_linear(matrix_1, matrix_2, parameters):
#     c = parameters[0]
#     d = parameters[1]
#     a = parameters[2]
#     matrix = inner_matrix(matrix_1 +c, matrix_2 +c) 
    
#     return a*(matrix +  d)

def kernel_linear(matrix_1, matrix_2, parameters):
    c = parameters[0]
    #a = parameters[1]
    matrix = inner_matrix(matrix_1, matrix_2) 
    
    return matrix + c

def kernel_poly(matrix_1, matrix_2, parameters):
    d = np.floor(parameters[0])

    c = parameters[1]
    a = parameters[2]
    
    b =  parameters[3]
    matrix = inner_matrix(matrix_1 + c, matrix_2 + c)
    K = a * matrix ** d + b
    return K 


def kernel_BM(matrix_1, matrix_2):
    
    return np.minimum(matrix_1[:, None], matrix_1[:, None].T)
    
    
"""A dictionnary containing the different kernels. If you wish to build a custom 
 kernel, add the function to the dictionnary.
"""
kernels_dic = {"RBF" : kernel_RBF,"poly": kernel_poly, "Laplacian": kernel_laplacian, 
               "sigmoid": kernel_sigmoid, "Rational Quadratic": kernel_rational_quadratic,
               "inverse_multiquad": kernel_inverse_multiquad, "quadratic" : kernel_quad,
               "poly": kernel_poly, "inverse_power_alpha": kernel_inverse_power_alpha,
               "gaussian multi": kernel_gaussian_linear, "Bernoulli": kernel_bernoulli,
               "Local Periodic" : kernel_local_periodic, "Rational Quadratic and Gaussian": kernel_rational_quadratic_gaussian,
               "Periodic": kernel_periodic, "Linear": kernel_linear, "Constant": kernel_constant, 
               "RBF var" : kernel_RBF_sigma, "Matern 2.5": kernel_matern_25, "RBF and poly": kernel_rbf_poly,
               "RBF and linear": kernel_rbf_linear, "BM": kernel_BM
               }

