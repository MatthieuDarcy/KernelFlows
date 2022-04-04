# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 16:29:31 2022

@author: User
"""

import autograd.numpy as np
from autograd import value_and_grad 



from sklearn.model_selection import train_test_split



class GPRegressor():
    
    def __init__(self, kernel_function, parameters):
        
        
        self.kernel_function = kernel_function
        self.parameters = np.copy(parameters)
        self.reg = 1e-5
        
    
    def rho(self, parameters, X_data, Y_data, sample_indices,  reg = 1e-5):
        
        # Construct the kernel matrix
        kernel_matrix = self.kernel_function(X_data, X_data, parameters)
        
        # Extract a submatrix from the sample 
        kernel_sample = kernel_matrix[np.ix_(sample_indices, sample_indices)]
        
        Y_sample = Y_data[sample_indices]
        
        

        top = np.matmul(Y_sample.T, np.linalg.solve(kernel_sample + reg*np.eye(kernel_sample.shape[0]), Y_sample))
        bottom = np.matmul(Y_data.T, np.linalg.solve(kernel_matrix + reg*np.eye(kernel_matrix.shape[0]), Y_data))
        return 1 - top/bottom

    # This function creates a batch and associated sample
    def batch_creation(self, X, batch_proportion = 1.0, sample_proportion = 0.5):
        """
        

        Parameters
        ----------
        X : numpy array 
        batch_size : TYPE
            DESCRIPTION.
        sample_proportion : TYPE, optional
            DESCRIPTION. The default is 0.5.

        Returns
        -------
        sample_indices : TYPE
            DESCRIPTION.
        batch_indices : TYPE
            DESCRIPTION.

        """

        N = X.shape[0]

        N_batch = int(np.ceil(N*batch_proportion))
        N_sample = int(np.ceil(N_batch*sample_proportion))
        

        
        
        idx = np.arange(N)
        

        np.random.shuffle(idx)

        
        idx_batch = idx[:N_batch]
        
        idx_sample = idx_batch[: N_sample]
        
        return idx_sample, idx_batch
    
    def fit(self, X, Y, optimize = False):
        
        self.X_train = np.copy(X)
        
        k_matrix = self.kernel_function(self.X_train,self.X_train, self.parameters)
        
        self.weights = np.linalg.solve(k_matrix + self.reg*np.eye(k_matrix.shape[0]), Y)
        
    
    def predict(self, X_test):
        
        k_test = self.kernel_function(X_test, self.X_train, self.parameters)
        
        return k_test@self.weights
        
        
        
    
    def optimize_parameters(self,  X, Y, iterations, batch_proportion = 1.0, sample_proportion = 0.5, optimizer = "SGD",
                            learning_rate = 0.1, beta = 0., reg = 1e-5, copy= True, n_runs = 1, parameter_range = None):
        
        if copy:
            self.X_train = np.copy(X)
            self.Y_train = np.copy(Y)
        
        # Lists that keep track of the history of the algorithm
        self.rho_hist = []
        self.grad_hist = []
        self.para_hist = []
        self.rho_running_mean = []
            
        
        if parameter_range is None:
            parameter_range = np.array([[1e-5,1e5] for i in range(self.parameters.shape[0])])

            

        
        grad_rho = value_and_grad(self.rho)

        best_score = np.inf
        
        optimized_parameters = []
        
        for k in range(n_runs):
            
            if k ==0:
                parameters = np.copy(self.parameters)
                
            else:
                parameters = (parameter_range[:, 1] - parameter_range[:, 0])*np.random.random(size = (self.parameters.shape[0])) + parameter_range[:, 0]
                
            momentum = np.zeros(parameters.shape, dtype = "float")
            
            rho_values = []
            para_values = []
            
            for i in range(iterations):
    
                    
                # Create a batch and a sample
                

                sample_indices, batch_indices = self.batch_creation(X, batch_proportion = batch_proportion, sample_proportion = sample_proportion)
                
                #print(sample_indices, batch_indices)
                X_data = X[batch_indices]
                Y_data = Y[batch_indices]
                
    
                    
                # Changes parameters according to SGD rules
                if optimizer == "SGD":
                    
                    rho, grad_mu = grad_rho(parameters, X_data, Y_data, 
                                               sample_indices, reg = self.reg)
                    
                    if  rho > 1 + 1e-5 or rho < 0 - 1e-5:
                        print("Warning, rho outside [0,1]: ", rho)
                    else:
                        parameters -= learning_rate * grad_mu
                        
                
                # Changes parameters according to Nesterov Momentum rules     
                elif optimizer == "Nesterov":
                    rho, grad_mu = grad_rho(parameters  - learning_rate * beta * momentum, X_data, Y_data, 
                                               sample_indices, reg = self.reg)

                    if  rho > 1 + 1e-5 or rho < 0 - 1e-5:
                        print("Warning, rho outside [0,1]: ", rho)
                    else:
                        momentum = beta * momentum + grad_mu
                        parameters -= learning_rate * momentum
                
                else:
                    print("Error optimizer, name not recognized")
                
                rho_values.append(rho)
                para_values.append(np.copy(parameters))
            
            
            self.rho_hist.append(rho_values)
            self.para_hist.append(para_values)
            
            # Select the best parameter based on the running mean
            running_loss = np.convolve(rho_values, np.ones(int(len(rho_values)*0.1))/int(len(rho_values)*0.1), mode='valid')
        
            best_loss = np.argmin(running_loss)
            score = running_loss 

            optimized_parameters.append(np.copy(parameters))
            
            
            if score < best_score:
                best_score = score
                best_run = k
                
        
        self.rho_hist = np.array(self.rho_hist)
        self.para_hist = np.array(self.para_hist)
        return optimized_parameters[best_run]
        
        
if __name__ == "__main__": 
    
    import matplotlib.pyplot as plt
    
    noise_level = 0.2


    def f(x, noise_level=noise_level):
        return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2))\
               + np.random.randn() * noise_level
    
    
    x = np.linspace(-2, 2, 200).reshape(-1, 1)
    fx = np.array([f(x_i, noise_level=0.0) for x_i in x])
    y = np.array([f(x_i, noise_level=noise_level) for x_i in x])
    
    plt.figure()
    plt.plot(x, fx, label = "True function")
    plt.scatter(x, y, label = "Observations")
    plt.legend()
    
    #%%
    
    from sklearn.model_selection import train_test_split

    X, X_test, Y, Y_test = train_test_split(x,y, test_size = 0.2, random_state=2022, shuffle = True) 
    
    #%%
    
    from kernel_functions_autograd import kernel_RBF
    
    sigma = np.array([1.0])
    GP =  GPRegressor(kernel_RBF, sigma)
    
    
    #%%
    from sklearn.metrics import mean_squared_error as mse
    GP.fit(X, Y)
    
    pred = GP.predict(X_test)

    print("The mean squared error on the test set is", np.round(mse(pred, Y_test), 3))
    
    
    #%%
    
    fx_pred = GP.predict(x)
    
    plt.figure()
    plt.plot(x, fx, label = "True function")
    plt.plot(x, fx_pred, label = "Predicted function")
    plt.legend()
    
    
    #%% Optimize
    iterations = 1000
    optimized_para = GP.optimize_parameters(X, Y, iterations)
    #%%
    running_loss = np.convolve(GP.rho_hist[0], np.ones(100)/100, mode='valid')
    
    
    rho_values = GP.rho_hist[0]
    
    run = np.convolve(rho_values, np.ones(int(len(rho_values)*0.1))/int(len(rho_values)*0.1), mode='valid')
    
    
    #%%
    
    plt.figure()
    plt.plot(running_loss)
    
    
    
    
    
    
                
            
                
                
                
                
        