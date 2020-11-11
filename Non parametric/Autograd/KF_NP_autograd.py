
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:26:44 2020

@author: matth
"""

import autograd.numpy as np
from autograd import value_and_grad 
import math
import random

from kernel_functions_autograd import kernels_dic

default_lambda = 1e-5


#%%

# The pi or selection matrix
def pi_matrix(sample_indices, dimension):
    pi = np.zeros(dimension)
    
    for i in range(dimension[0]):
        pi[i][sample_indices[i]] = 1
    
    return pi

# The rho function
def rho(parameters, matrix_data, Y_data, sample_indices,  kernel_keyword= "RBF", reg = 0.000001):
    kernel = kernels_dic[kernel_keyword]    
    kernel_matrix = kernel(matrix_data, matrix_data, parameters)
    
    pi = pi_matrix(sample_indices, (sample_indices.shape[0], matrix_data.shape[0]))   
    
    sample_matrix = np.matmul(pi, np.matmul(kernel_matrix, np.transpose(pi)))
    
    Y_sample = Y_data[sample_indices]
    
    lambda_term = reg
    inverse_data = np.linalg.inv(kernel_matrix + lambda_term * np.identity(kernel_matrix.shape[0]))
    inverse_sample = np.linalg.inv(sample_matrix + lambda_term * np.identity(sample_matrix.shape[0]))
    

    top = np.matmul(Y_sample.T, np.matmul(inverse_sample, Y_sample))
    bottom = np.matmul(Y_data.T, np.matmul(inverse_data, Y_data))
    return 1 - top/bottom


def mmd(parameters, matrix_data, Y_data, sample_indices, kernel_keyword= "RBF", reg = 0.000001):
    matrix_1 = matrix_data
    matrix_2 = matrix_data[sample_indices]
    
    kernel = kernels_dic[kernel_keyword]
    
    kernel_matrix_1 = kernel(matrix_1, matrix_1, parameters)
    m = kernel_matrix_1.shape[0]
    kernel_matrix_2 = kernel(matrix_2, matrix_2, parameters)
    n = kernel_matrix_2.shape[0]
    
    kernel_matrix_3 = kernel(matrix_1, matrix_2, parameters)
    
    mean_1 = np.sum(kernel_matrix_1)/(m^2)
    mean_2 = np.sum(kernel_matrix_2)/(n^2)
    cov = np.sum(kernel_matrix_3)/(n*m)
    
    return mean_1 + mean_2 - 2*cov


#%% 
"""We define several useful functions"""
    
# Returns a random sample of the data, as a numpy array
def sample_selection(data, size):
    indices = np.arange(data.shape[0])
    sample_indices = np.sort(np.random.choice(indices, size, replace= False))
    
    return sample_indices

# This function creates a batch and associated sample
def batch_creation(data, batch_size = False, sample_proportion = 0.5):
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
    
def batch_creation_uniform(data, sample_proportion = 0.5, batch_size = False):
    outcomes = [0, 1]
    weights = [sample_proportion, 1-sample_proportion]
    batch_size = 0
    sample_size = 0
    
    while batch_size == 0:
        batch_rv  = random.choices(outcomes, weights, k = data.shape[0])
        # print(batch_rv)
        batch_idx = np.where(np.array(batch_rv) == 0)[0]
        batch_size = batch_idx.shape[0]
        # print("batch", batch_size, batch_idx)
    
    while sample_size == 0:
        sample_rv = random.choices(outcomes, weights, k = batch_idx.shape[0])
        # print(sample_rv)
        sample_idx = np.where(np.array(sample_rv) == 0)[0]
        sample_size = sample_idx.shape[0]
        # print("sample", sample_size, sample_idx)
    
    return sample_idx, batch_idx
    
sampling_dic = {"default": batch_creation, "uniform": batch_creation_uniform}

#%%
    
# Splits the data into the target and predictor variables.
def split(data):
    X = data[:, :-1]
    Y = data[:, -1]
    
    return X, Y


# Generate a prediction
def kernel_regression(X_train, X_test, Y_train, param, kernel_keyword = "RBF", reg = default_lambda):
    kernel = kernels_dic[kernel_keyword]

    # The data matrix (theta in the original paper)
    k_matrix = kernel(X_train, X_train, param)
    k_matrix += reg * np.identity(k_matrix.shape[0])
    
    # The test matrix 
    t_matrix = kernel(X_test, X_train, param)
    
    # Regression coefficients in feature space
    coeff = np.matmul(np.linalg.inv(k_matrix), Y_train)
    
    prediction = np.matmul(t_matrix, coeff)
    
    return prediction, coeff

def kernel_regression_coeff(X_train, Y_train, param, kernel_keyword = "RBF", reg = default_lambda):
    kernel = kernels_dic[kernel_keyword]

    # The data matrix (theta in the original paper)
    k_matrix = kernel(X_train, X_train, param)
    k_matrix += reg * np.identity(k_matrix.shape[0])
    
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

def adjust_LR(iteration, LR, parameters):
    if parameters["adjust_type"] == False:
        return parameters["LR"]
    else:
        if parameters["adjust_type"] == "linear":
            return LR-parameters["rate"]
        elif parameters["adjust_type"] == "threshold":
            if iteration in parameters["LR_threshold"][0]:
                return parameters["LR_threshold"][1][np.where(parameters["LR_threshold"][0]==iteration)[0][0]]
            else:
                return LR
        
#%% Grad functions

""" We define the gradient calculator function.Like rho, the gradient 
calculator function accesses the gradfunctions via a keyword"""
# Gradient calculator function. Returns an array
def grad_rho(parameters, X_data, Y_data, sample_indices, kernel_keyword= "RBF", reg = 0.000001):
    grad_K = value_and_grad(rho, 1)
    rho_value, gradient = grad_K(parameters, X_data, Y_data, sample_indices, kernel_keyword, reg = reg)
    return rho_value, gradient

def grad_mmd(parameters, X_data, Y_data, sample_indices, kernel_keyword= "RBF", reg = 0.000001):
    grad_K = value_and_grad(mmd, 1)
    rho_value, gradient = grad_K(parameters, X_data, Y_data, sample_indices, kernel_keyword, reg = reg)
    return rho_value, gradient

grad_dic = {"rho" : grad_rho, "mmd": grad_mmd}       
    
#%%
                
default_parameters = {"LR" : 0.1, "type_epsilon": "relative", 
                      "adjust_type" : False, "rate": 0.0, "LR_threshold" :[],
                      "reg" : default_lambda, "loss": "rho", "sampling": "default"}

class KernelFlowsNPAutograd():
    
    def __init__(self, kernel_keyword, parameters):
        self.kernel_keyword = kernel_keyword
        self.parameters = np.copy(parameters)
        
        # Lists that keep track of the history of the algorithm
        self.rho_values = []
        self.coeff = []
        self.points_hist = []
        self.batch_hist = []
        self.epsilon = []
        self.perturbation = []
    
    
    def fit(self, X, Y, iterations, batch_size = False, learning_parameters = default_parameters, 
            show_it = 100, record_hist = False):
        
        learning_parameters = {**default_parameters, **learning_parameters}
        learning_rate = learning_parameters["LR"]
        type_epsilon = learning_parameters["type_epsilon"]
        reg = learning_parameters["reg"]
        learning_parameters["rate"] = learning_rate/iterations
        loss_name = learning_parameters["loss"]
        grad = grad_dic[loss_name]      
        batch_creation = sampling_dic[learning_parameters["sampling"]]
        
        # Create a copy of the parameters (so the original parameters aren't modified)
        self.LR = [learning_rate]
        self.type_epsilon = type_epsilon
        self.X_train = np.copy(X)
        self.Y_train = np.copy(Y)
        self.iteration = iterations
        self.points_hist.append(np.copy(X))
        parameters = self.parameters
        
        
        X = np.copy(X)
        self.X = X
        data_set_ind = np.arange(X.shape[0])
        perturbation = np.zeros(X.shape)
        for i in range(iterations):
            if type(show_it) == int and i % show_it == 0:
                print("Iteration ", i)

            # Create a batch and a sample
            sample_indices, batch_indices = batch_creation(X, batch_size)
            X_batch = X[batch_indices]
            Y_batch = Y[batch_indices]
            self.batch_hist.append(np.copy(X_batch))
            
            # The indices of all the elements not in the batch
            not_batch = np.setdiff1d(data_set_ind, batch_indices)
            
        
            # Compute the gradient
            rho, g = grad(self.parameters, X_batch, Y_batch, 
                                           sample_indices, self.kernel_keyword, reg = reg)
            g = -g
            if loss_name == "rho" and (rho >1.0001 or rho <-0.00001):
                    print ("Rho outside allowed bounds", rho)
            # Compute the perturbations by interpolation
            if batch_size == False:
                perturbation = g
                coeff = kernel_regression_coeff(X_batch, g, parameters, self.kernel_keyword, reg = reg)

            else:
                g_interpolate, coeff = kernel_regression(X_batch, X[not_batch], g, parameters, self.kernel_keyword, reg = reg)                
                perturbation[batch_indices] = g
                perturbation[not_batch] = g_interpolate

            #Find epsilon
            epsilon = compute_LR(learning_rate, X, perturbation, type_epsilon = type_epsilon)
            # Adjust the learning rate based on the learning parameters
            learning_rate = adjust_LR(i, learning_rate, learning_parameters)
            self.LR.append(learning_rate)

            
                
            #Update the points
            X += epsilon * perturbation
            
            # Recording the regression coefficients
            self.coeff.append(coeff)
            # Update the history
            self.rho_values.append(rho)
            self.epsilon.append(epsilon)
            self.perturbation.append(perturbation)

            
            if record_hist == True:
                self.points_hist.append(np.copy(X))
                                
        
        return X
                
    def flow_transform(self, X_test, iterations, show_it = False, epsilon_choice = "combination"):
        kernel = kernels_dic[self.kernel_keyword]
        X_test = np.copy(X_test)
        # Keeping track of the perturbations
        self.test_history = []
        self.test_history.append(X_test)
        # First case: mini-batch was used, hence the regression coefficients have already been computed
        for i in range(iterations):
            
            if type(show_it) == int and i % show_it == 0:
                print("Iterations: ", i)
                
            # Fetching the regression coefficients, the batch and the learning rate
            coeff = self.coeff[i]
            X_batch = self.batch_hist[i]
            learning_rate = self.LR[i]
                
            # Computing the regression matrix
            test_matrix= kernel(X_test, X_batch, self.parameters)
            
            # Prediction and perturbation
            perturbation = np.dot(test_matrix, coeff)
            
            if epsilon_choice == "historic":
                epsilon = self.epsilon[i]
            elif epsilon_choice == "new":
                epsilon = compute_LR(learning_rate, X_test, perturbation, type_epsilon = self.type_epsilon)
            elif epsilon_choice == "combination":
                epsilon_1 = self.epsilon[i]
                epsilon_2 = compute_LR(learning_rate, X_test, perturbation, type_epsilon = self.type_epsilon)
                epsilon = min(epsilon_1, epsilon_2)
            else:
                print("Error epsilon type ")
            X_test += epsilon * perturbation
                
            # Updating the test history
            self.test_history.append(X_test)
        return X_test
    
    def predict(self, X_test, show_it = False, reg = default_lambda, kernel_it = -1, epsilon_choice = "combination"):

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
            
        prediction, coeff = kernel_regression(X_train, flow_test, Y_train, self.parameters, self.kernel_keyword, reg= reg) 

        return prediction
    
    def fit_predict(self, X_train, X_test, Y_train, reg = default_lambda, kernel_it = -1,  show_it = False, epsilon_choice = "historic"):
        # Transforming using the flow
        if kernel_it > 0:
           flow_test = self.flow_transform(X_test, kernel_it, show_it = show_it, epsilon_choice = epsilon_choice)
           flow_train = self.flow_transform(X_train, kernel_it, show_it = show_it, epsilon_choice = epsilon_choice)
        elif kernel_it == -1:
            flow_test = self.flow_transform(X_test, self.iteration, show_it = show_it, epsilon_choice = epsilon_choice) 
            flow_train = self.flow_transform(X_train,self.iteration, show_it = show_it, epsilon_choice = epsilon_choice)
        else: 
            print("Error, Kernel iteration not understood")
        
        prediction, coeff = kernel_regression(flow_train, flow_test, Y_train, self.parameters, self.kernel_keyword, reg = reg) 
        
        return prediction
    def predict_train(self, reg = default_lambda):
        X_train = self.X
        Y_train = self.Y_train
        prediction, coeff = kernel_regression(X_train, X_train, Y_train, self.parameters, self.kernel_keyword, reg = reg) 

        return prediction

#%%
class KernelRegression():
    
    def __init__(self, kernel_keyword, parameters):
        self.kernel_keyword = kernel_keyword
        self.parameters = np.copy(parameters)
        
    def fit(self, X_train, Y_train, reg = default_lambda):
        self.coeff = kernel_regression_coeff(X_train, Y_train, self.parameters, kernel_keyword = self.kernel_keyword, reg = reg)
        self.X_train = np.copy(X_train)
        
    def predict(self, X_test, reg = default_lambda):  
        kernel = kernels_dic[self.kernel_keyword]
        test_matrix = kernel(X_test, self.X_train, self.parameters)
        pred = np.dot(test_matrix, self.coeff)

        return pred
    
    def fit_predict(self, X_train, X_test, Y_train, reg = default_lambda):
        self.fit(X_train, Y_train, reg = reg)
        pred = self.predict(X_test)
        
        return pred

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
    parameters = {"LR" : 0.2}
    mu = np.array([2.0])
    iterations = 5000
    KF = KernelFlowsNPAutograd("RBF", mu)
    train_transformed = KF.fit(X, Y, iterations, batch_size = False, show_it = 2000, learning_parameters = parameters, record_hist = True)
    #%%    
    # Change this parameter to view different iterations
    it = 2000
    plt.figure()
    plt.title("Train at iteration {}".format(it))
    plt.scatter(KF.points_hist[it][:, 0], KF.points_hist[it][:, 1], c= labels)
    
    it = 5000
    plt.figure()
    plt.title("Train at iteration {}".format(it))
    plt.scatter(KF.points_hist[it][:, 0], KF.points_hist[it][:, 1], c= labels)
        
    it = -1
    plt.figure()
    plt.title("Train at iteration {}".format(it))
    plt.scatter(KF.points_hist[it][:, 0], KF.points_hist[it][:, 1], c= labels)

#%%
    

