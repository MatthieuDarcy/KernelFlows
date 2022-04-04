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
    
    def fit(self, X, Y, optimize = False, optimizer = "SGD", iterations = 1000):
        
        self.X_train = np.copy(X)
        self.Y_train = np.copy(Y)
        
        if optimize == True:
            optimized_para = self.optimize_parameters(self.X_train, self.Y_train, iterations = iterations, optimizer = optimizer)
            self.parameters = optimized_para
        
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


        
        parameters = np.copy(self.parameters)
        momentum = np.zeros(parameters.shape, dtype = "float")
            

            
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
                
                self.rho_hist.append(rho)
                self.para_hist.append(np.copy(parameters))
    
            
        # Select the best parameter based on the running mean
        running_loss = np.convolve(self.rho_hist, np.ones(int(len(self.rho_hist)*0.1))/int(len(self.rho_hist)*0.1), mode='valid')
        
        
        
        best_loss = np.argmin(running_loss)
        
        best_parameter = self.para_hist[best_loss + iterations - running_loss.shape[0]]
        
        
        
        
        self.rho_hist = np.array(self.rho_hist)
        self.para_hist = np.array(self.para_hist)
        self.rho_running_mean = np.array(running_loss)
        return best_parameter
        
        
if __name__ == "__main__": 
    
    import matplotlib.pyplot as plt
    
    noise_level = 0.1


    def f(x, noise_level=noise_level):
        return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2))\
               + np.random.randn() * noise_level
    
    
    x = np.linspace(-2, 2, 400).reshape(-1, 1)
    fx = np.array([f(x_i, noise_level=0.0) for x_i in x])
    
    X = np.random.uniform(-2, 2, size = (200,1))
    Y = np.array([f(x_i, noise_level=noise_level) for x_i in X])
    
    plt.figure()
    plt.plot(x, fx, label = "True function")
    plt.scatter(X, Y, label = "Observations")
    plt.legend()
    
    

    
    #%%
    
    from kernel_functions_autograd import kernel_RBF
    
    sigma = np.array([1.0])
    GP =  GPRegressor(kernel_RBF, sigma)
    
    
    #%%
    from sklearn.metrics import mean_squared_error as mse
    GP.fit(X, Y)
    
    pred = GP.predict(x)

    print("The mean squared error on the test set is", np.round(mse(pred, fx), 3))
    
    
    #%%
    
    fx_pred = GP.predict(x)
    
    plt.figure()
    plt.plot(x, fx, label = "True function")
    plt.plot(x, fx_pred, label = "Predicted function")
    plt.legend()
    
    
    #%% Use optimized parameters
    
    
    GP.fit(X, Y, optimize = True)
    
    pred = GP.predict(x)
    
    
    print("The optimized paramters are ", GP.parameters)
    print("The mean squared error on the test set is", np.round(mse(pred, fx), 3))
    
    
    #%%
    
    fx_pred = GP.predict(x)
    
    plt.figure()
    plt.plot(x, fx, label = "True function")
    plt.plot(x, fx_pred, label = "Predicted function")
    plt.legend()
    
    #%%
    
    
    plt.figure()
    plt.plot(GP.rho_running_mean, label = "Running mean")
    plt.legend()
    
    plt.figure()
    plt.plot(GP.rho_hist, label = "Rho")
    plt.legend()
    


    
    #%%
    
    
    GP2 = GPRegressor(kernel_RBF, GP.para_hist[-1])
    
    GP2.fit(X, Y)
    
    
    pred = GP2.predict(x)
    
    
    print("The optimized paramters are ", GP.parameters)
    print("The mean squared error on the test set is", np.round(mse(pred, fx), 3))
    
    
    #%%
    
    fx_pred = GP2.predict(x)
    
    plt.figure()
    plt.plot(x, fx, label = "True function")
    plt.plot(x, fx_pred, label = "Predicted function")
    plt.legend()
    
    #%%


    
    
    #%%
    
    
    
    
    
    
    
                
            
                
                
                
                
        