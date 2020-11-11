
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:26:44 2020

@author: matth
"""

import numpy as np
import math

from kernel_functions_hybrid import kernels_dic_hybrid
from nabla_functions_hybrid import nabla_dic_hybrid

default_lambda = 0.000001


#%%

# The pi or selection matrix
def pi_matrix(sample_indices, dimension):
    pi = np.zeros(dimension)
    
    for i in range(dimension[0]):
        pi[i][sample_indices[i]] = 1
    
    return pi

# The rho function
def rho(parameters, matrix_data, Y_data, sample_indices,  kernel_keyword= "RBF"):
    kernel = kernels_dic_hybrid[kernel_keyword]    
    kernel_matrix = kernel(matrix_data, matrix_data, parameters)
    
    pi = pi_matrix(sample_indices, (sample_indices.shape[0], matrix_data.shape[0]))   
    
    sample_matrix = np.matmul(pi, np.matmul(kernel_matrix, np.transpose(pi)))
    
    Y_sample = Y_data[sample_indices]
    
    lambda_term = 0.000001
    inverse_data = np.linalg.inv(kernel_matrix + lambda_term * np.identity(kernel_matrix.shape[0]))
    inverse_sample = np.linalg.inv(sample_matrix + lambda_term * np.identity(sample_matrix.shape[0]))
    
    top = np.dot(Y_sample, np.matmul(inverse_sample, Y_sample))
    bottom = np.dot(Y_data, np.matmul(inverse_data, Y_data))
    
    return 1 - top/bottom

# Computes the frechet derivative for KF (equation 6.5 of the original paper)
def frechet(parameters, X, Y, sample_indices, kernel_keyword = "RBF", regu_lambda = default_lambda):
    Y_sample = Y[sample_indices]

    pi = pi_matrix(sample_indices, (sample_indices.shape[0], X.shape[0])) 
    lambda_term = regu_lambda
    
    # Setting the kernel functions
    nabla = nabla_dic_hybrid[kernel_keyword]
    
    
    # Computing the nabla matrix and the regular matrix (points)
    derivative_matrix, batch_matrix, derivative_matrix_parameters = nabla(X, parameters)
    
    # # Computing the nabla matrix and the regular matrix (parameters)
    # nabla_matrix_parameters, theta = nabla_parameters(X, parameters)
    

    # Computing the Kernel matrix Inverses
    sample_matrix = np.matmul(pi, np.matmul(batch_matrix, np.transpose(pi)))
    batch_inv = np.linalg.inv(batch_matrix + lambda_term * np.identity(batch_matrix.shape[0]))
    sample_inv = np.linalg.inv(sample_matrix + lambda_term * np.identity(sample_matrix.shape[0]))
    
    
    # Computing the top and bottom terms
    top = np.matmul(np.transpose(Y_sample), np.matmul(sample_inv, Y_sample))
    bottom = np.matmul(np.transpose(Y), np.matmul(batch_inv, Y))
    
    # Computing z_hat and y_hat (see original paper)
    Z_hat = np.matmul(np.transpose(pi), np.matmul(sample_inv, np.matmul(pi, Y)))
    Y_hat = np.matmul(batch_inv, Y)
    #Computing rho   
    rho = 1- top/bottom
    

    # Computing the Frechet derivative (points)
    K_y = np.squeeze(np.matmul(derivative_matrix, Y_hat), axis = 2)
    K_z = np.squeeze(np.matmul(derivative_matrix, Z_hat), axis = 2)
    
    g = 2*((1-rho)* Y_hat * K_y - Z_hat * K_z) 
    g = g/bottom
    
    # Computing the Frechet derivative (parameters)
    gradient = ((1-rho)*np.matmul(np.transpose(Y_hat), np.matmul(derivative_matrix_parameters,Y_hat)) - np.matmul(np.transpose(Z_hat), np.matmul(derivative_matrix_parameters, Z_hat)))
    gradient = -gradient/bottom
    gradient = np.squeeze(gradient)
    
    return g, rho, gradient

#%% 
"""We define several useful functions"""
    
# Returns a random sample of the data, as a numpy array
def sample_selection(data, size):
    indices = np.arange(data.shape[0])
    sample_indices = np.sort(np.random.choice(indices, size, replace= False))
    
    return sample_indices

# This function creates a batch and associated sample
def batch_creation(data, batch_size, sample_proportion = 0.5):
    # If False, the whole data set is the mini-batch, otherwise either a 
    # percentage or explicit quantity.
    if batch_size == False:
        data_batch = data
        batch_indices = np.arange(data.shape[0])
    elif 0 < batch_size <= 1:
        batch_size = int(data.shape[0] * batch_size)
        batch_indices = sample_selection(data, batch_size)
        data_batch = data[batch_indices]
    else:
        batch_indices = sample_selection(data, batch_size)
        data_batch = data[batch_indices]
        

    # Sample from the mini-batch
    sample_size = math.ceil(data_batch.shape[0]*sample_proportion)
    sample_indices = sample_selection(data_batch, sample_size)
    
    return sample_indices, batch_indices
    

# Splits the data into the target and predictor variables.
def split(data):
    X = data[:, :-1]
    Y = data[:, -1]
    
    return X, Y


# Generate a prediction
def kernel_regression(X_train, X_test, Y_train, param, kernel_keyword = "RBF", regu_lambda = default_lambda):
    kernel = kernels_dic_hybrid[kernel_keyword]

    # The data matrix (theta in the original paper)
    k_matrix = kernel(X_train, X_train, param)
    k_matrix += regu_lambda * np.identity(k_matrix.shape[0])
    
    # The test matrix 
    t_matrix = kernel(X_test, X_train, param)
    
    # Regression coefficients in feature space
    coeff = np.matmul(np.linalg.inv(k_matrix), Y_train)
    
    prediction = np.matmul(t_matrix, coeff)
    
    return prediction, coeff

def kernel_regression_coeff(X_train, Y_train, param, kernel_keyword = "RBF", regu_lambda = default_lambda):
    kernel = kernels_dic_hybrid[kernel_keyword]

    # The data matrix (theta in the original paper)
    k_matrix = kernel(X_train, X_train, param)
    k_matrix += regu_lambda * np.identity(k_matrix.shape[0])
    
    # Regression coefficients in feature space
    coeff = np.matmul(np.linalg.inv(k_matrix), Y_train)
    
    return coeff
    
# This function ccomputes epsilon (relative: max relative transaltion = rate, absolute: max translation = rate)
def compute_LR(rate, old_points, g_pert, type_epsilon = "relative"):
    if type_epsilon == "relative":
        norm_old = np.linalg.norm(old_points, axis = 1)
        norm_pert = np.linalg.norm(g_pert, axis = 1)


        #Replace all tiny values by 1
        norm_pert[norm_pert < 0.000001] = 1
        ratio = norm_old/norm_pert
        
        epsilon = rate * np.amin(ratio)
    elif type_epsilon == "absolute":
        norm_pert = np.linalg.norm(g_pert, axis = 1)
        norm_pert[norm_pert < 0.000001] = 1
        epsilon = rate / np.amax(norm_pert)
    
    elif type_epsilon == "usual":
        epsilon = rate
    else:
        print("Error type of epsilon")
    return epsilon
#%%
class KernelFlowsHybrid():
    
    def __init__(self, kernel_keyword, parameters, regression_type = "single"):
        self.kernel_keyword = kernel_keyword
        self.parameters = np.copy(parameters)
        
        # Lists that keep track of the history of the algorithm
        self.rho_values = []
        self.coeff = []
        self.points_hist = []
        self.batch_hist = []
        self.epsilon = []
        self.perturbation = []
        self.para_hist = []
        self.para_hist.append(parameters)
    
    
    def fit(self, X, Y, iterations, batch_size = False,learning_rate = 0.1, type_epsilon = "relative", show_it = 100, record_hist = True, reg = default_lambda, LR_parameters = 0.1):
        # Create a copy of the parameters (so the original parameters aren't modified)
        self.LR = learning_rate
        self.type_epsilon = type_epsilon
        self.X_train = np.copy(X)
        self.Y_train = np.copy(Y)
        self.iteration = iterations
        self.points_hist.append(np.copy(X))

        
        parameters = self.parameters
        
        if batch_size == False:
            self.regression = False
        
        X = np.copy(X)
        self.X = X
        data_set_ind = np.arange(X.shape[0])
        perturbation = np.zeros(X.shape)
        for i in range(iterations):
            if i % show_it == 0:
                print("Iteration ", i)
            
            
            # Create a batch and a sample
            sample_indices, batch_indices = batch_creation(X, batch_size)
            X_batch = X[batch_indices]
            Y_batch = Y[batch_indices]
            self.batch_hist.append(np.copy(X_batch))
            # The indices of all the elements not in the batch
            not_batch = np.setdiff1d(data_set_ind, batch_indices)
            

            # Compute the gradient
            g, rho, gradient = frechet(parameters, X_batch, Y_batch, sample_indices, kernel_keyword = self.kernel_keyword)
            
            if rho >1.0001 or rho <-0.00001:
                    print ("Rho outside allowed bounds", rho)
            
            # Compute the perturbations by interpolation
            if batch_size == False:
                perturbation = g
                coeff = kernel_regression_coeff(X_batch, g, parameters, self.kernel_keyword, regu_lambda = reg)
            else:
                g_interpolate, coeff = kernel_regression(X_batch, X[not_batch], g, parameters, self.kernel_keyword, regu_lambda = reg)                
                perturbation[batch_indices] = g
                perturbation[not_batch] = g_interpolate

            #print(perturbation.shape)
            #Find epsilon
            epsilon = compute_LR(learning_rate, X, perturbation, type_epsilon = type_epsilon)
            
                
            #Update the points
            X += epsilon * perturbation
            # Update the parameters
            parameters -= LR_parameters * gradient
            # Recording the regression coefficients
            self.coeff.append(coeff)
            # Update the history
            self.rho_values.append(rho)
            self.epsilon.append(epsilon)
            self.perturbation.append(perturbation)
            self.para_hist.append(np.copy(parameters))
            
            if record_hist == True:
                self.points_hist.append(np.copy(X))
                                

        
        return X
                
    def flow_transform(self, X_test, iterations, show_it = 1000, epsilon_choice = "combination"):
        kernel = kernels_dic_hybrid[self.kernel_keyword]
        X_test = np.copy(X_test)

        # Keeping track of the perturbations
        self.test_history = []
        self.test_history.append(np.copy(X_test))
        for i in range(iterations):
            if i % show_it == 0:
                print("Iterations: ", i)
                
            # Fetching the regression coefficients and the batch used
            coeff = self.coeff[i]
            X_batch = self.batch_hist[i]
            parameters = self.para_hist[i]
                
            # Computing the regression matrix
            test_matrix= kernel(X_test, X_batch, parameters)
            
            # Prediction and perturbation
            perturbation = np.dot(test_matrix, coeff)
            
            if epsilon_choice == "historic":
                epsilon = self.epsilon[i]
            elif epsilon_choice == "new":
                epsilon = compute_LR(self.LR, X_test, perturbation, type_epsilon = self.type_epsilon)
            elif epsilon_choice == "combination":
                epsilon_1 = self.epsilon[i]
                epsilon_2 = compute_LR(self.LR, X_test, perturbation, type_epsilon = self.type_epsilon)
                epsilon = min(epsilon_1, epsilon_2)
            else:
                print("Error epsilon type ")
            
            X_test += epsilon * perturbation
                
            # Updating the test history
            self.test_history.append(np.copy(X_test))
        return X_test
    
    def predict(self, X_test, show_it = 1000, regu = default_lambda, kernel_it = -1, epsilon_choice = "combination"):

        # Transforming using the flow
        if kernel_it > 0:
           flow_test = self.flow_transform(X_test, kernel_it, show_it = show_it, epsilon_choice = epsilon_choice)  
        elif kernel_it == -1:
            flow_test = self.flow_transform(X_test, self.iteration, show_it = show_it, epsilon_choice = epsilon_choice) 
        else: 
            print("Error, Kernel iteration not understood")
            
        # Fetching the train set transformed
        if kernel_it == -1: 
            X_train = self.X
        elif kernel_it >0:
            X_train = self.points_hist[kernel_it]
        else:
            print("Error kernel it")
            
        Y_train = self.Y_train
            
        prediction, coeff = kernel_regression(X_train, flow_test, Y_train, self.parameters, self.kernel_keyword, regu_lambda = regu) 

        return prediction
    
    def predict_train(self, regu = default_lambda):
        X_train = self.X
        Y_train = self.Y_train
        prediction, coeff = kernel_regression(X_train, X_train, Y_train, self.parameters, self.kernel_keyword, regu_lambda = regu) 

        return prediction

#%%
        
 
if __name__ == "__main__":
    
    """ Generating the swiss roll dataset"""
    from numpy.random import normal
    import matplotlib.pyplot as plt
    def spirals(N, start, end, a ,b, noise):
        interval  =  np.linspace(start, end , N)
        d1= []
        d2 = []
        d3 = []
        d4 = []
        for element in interval:
            d1.append((a* element -b) * np.sin(element) + normal(0, noise))
            d2.append((a * element - b) * np.cos(element)+ normal(0, noise))
            d3.append((-a* element +b) * np.sin(element)+ normal(0, noise))
            d4.append((-a * element +b) * np.cos(element)+ normal(0, noise))
        
        return np.array([d1, d2]).T, np.array([d3, d4]).T
    N = 60
    a = 1.5
    b = -2 
    start = 0
    end = 2 * math.pi
    noise = 0
    
    y1 = np.expand_dims(np.zeros(N) - 1, axis = 1)
    y2 = np.expand_dims(np.zeros(N) + 1, axis = 1)
    d1,d2 = spirals(N , start, end, a,b, noise)
    spiral_1 = np.concatenate((d1, y1), axis = 1)
    spiral_2 = np.concatenate((d2, y2), axis = 1)
    data = np.concatenate((spiral_1, spiral_2), axis = 0)
    
    X = data[:, :-1]
    Y = data[:, -1]
    Y = np.expand_dims(Y, axis = 1)
    labels = data[:, -1]
    
    
    plt.scatter(X[:, 0], X[:, 1], c= labels)
    plt.title("Swiss Roll Data set")
#%%
    """ Fitting Kernel Flows"""
    mu = np.array([2.0])
    iterations = 10000
    KF = KernelFlowsHybrid("RBF", mu)
    train_transformed = KF.fit(X, Y, iterations, batch_size = False, show_it = 2000, learning_rate = 0.2)
    #%%    
    """ Plotting the results"""
    # Change this parameter to view different iterations
    it = 1000
    plt.figure()
    plt.title("Train at iteration {}".format(it))
    plt.scatter(KF.points_hist[it][:, 0], KF.points_hist[it][:, 1], c= labels)
    
    it = 5000
    plt.figure()
    plt.title("Train at iteration {}".format(it))
    plt.scatter(KF.points_hist[it][:, 0], KF.points_hist[it][:, 1], c= labels)
    
    it = 10000
    plt.figure()
    plt.title("Train at iteration {}".format(it))
    plt.scatter(KF.points_hist[it][:, 0], KF.points_hist[it][:, 1], c= labels)

    
    it = iterations
    plt.figure()
    plt.title("Train at iteration {}".format(it))
    plt.scatter(KF.points_hist[it][:, 0], KF.points_hist[it][:, 1], c= labels)
