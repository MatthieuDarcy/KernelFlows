# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 16:29:31 2022

@author: User
"""

import autograd.numpy as np
from autograd import value_and_grad 




#from sklearn.model_selection import train_test_split



class GPRegressor():
    
    def __init__(self, kernel_function, parameters, reg= 1e-5):
        
        
        self.kernel_function = kernel_function
        self.parameters = np.copy(parameters)
        self.reg = reg
        
    def rho(self, parameters, X_data, Y_data, sample_indices,  reg = 1e-5):
        
        # Construct the kernel matrix
        
        kernel_matrix = self.kernel_function(X_data, X_data, parameters)
        
        # Extract a submatrix from the sample 
        kernel_sample = kernel_matrix[np.ix_(sample_indices, sample_indices)]
        kernel_cross = kernel_matrix[np.ix_(np.arange(kernel_matrix.shape[0]), sample_indices)]
        
        #print(kernel_cross.shape)
        
        Y_sample = Y_data[sample_indices]
        
        
        w_full = np.linalg.solve(kernel_matrix + reg*np.eye(kernel_matrix.shape[0]), Y_data)
        
        w_sample = np.linalg.solve(kernel_sample + reg*np.eye(kernel_sample.shape[0]), Y_sample)
        
        #w_mod = np.copy(w_full)
        #w_mod[sample_indices] -= w_sample
        
        full_norm = w_full@kernel_matrix@w_full
        sample_norm = w_sample@kernel_sample@w_sample

        top =  sample_norm - 2*w_full@kernel_cross@w_sample
        
        return 1 + top/full_norm
    # def rho(self, parameters, X_data, Y_data, sample_indices,  reg = 1e-5):
        
    #     # Construct the kernel matrix
        
    #     kernel_matrix = self.kernel_function(X_data, X_data, parameters)
        
    #     # Extract a submatrix from the sample 
    #     kernel_sample = kernel_matrix[np.ix_(sample_indices, sample_indices)]
        
    #     Y_sample = Y_data[sample_indices]
        
        

    #     top = np.matmul(Y_sample.T, np.linalg.solve(kernel_sample + reg*np.eye(kernel_sample.shape[0]), Y_sample))
    #     bottom = np.matmul(Y_data.T, np.linalg.solve(kernel_matrix + reg*np.eye(kernel_matrix.shape[0]), Y_data))
    #     return 1 - top/bottom
    
    def rho_average(self,  parameters, X, Y,  reg = 1e-5, n= 1, batch_proportion = 1.0, sample_proportion = 0.5):
        """
        Computes the (sample average) rho loss function

        Parameters
        ----------
        parameters : TYPE
            DESCRIPTION.
        X_data : TYPE
            DESCRIPTION.
        Y_data : TYPE
            DESCRIPTION.
        sample_indices : TYPE
            DESCRIPTION.
        reg : TYPE, optional
            DESCRIPTION. The default is 1e-5.
        n : TYPE, optional
            DESCRIPTION. The default is 1.
        batch_proportion : TYPE, optional
            DESCRIPTION. The default is 1.0.
        sample_proportion : TYPE, optional
            DESCRIPTION. The default is 0.5.

        Returns
        -------
        None.

        """
        
        #print(parameters)
        
        # Select a batch
        batch_indices = self.sample(X, batch_proportion) 
        batch_indices = np.sort(batch_indices)
        
        #print(batch_indices)
        X_data = X[batch_indices]
        Y_data = Y[batch_indices]
        
        # print(X_data)
        # print(Y_data)
        
        

        
        kernel_matrix = self.kernel_function(X_data, X_data, parameters)
        
        #print(kernel_matrix)
        
        loss = 0
        
        for i in range(n):

            # Select a sub-sample
            sample_indices = self.sample(X_data, sample_proportion) 
            sample_indices = np.sort(sample_indices)
            
            Y_sample = Y_data[sample_indices]
            
            # Extract a submatrix from the sample 
            kernel_sample = kernel_matrix[np.ix_(sample_indices, sample_indices)]
            kernel_cross = kernel_matrix[np.ix_(np.arange(kernel_matrix.shape[0]), sample_indices)]
            
            # Compute rho using the polarization identity (appears to be more stable)
            
            
            w_full = np.linalg.solve(kernel_matrix + reg*np.eye(kernel_matrix.shape[0]), Y_data)
            w_sample = np.linalg.solve(kernel_sample + reg*np.eye(kernel_sample.shape[0]), Y_sample)
            
            
            full_norm = w_full@kernel_matrix@w_full
            sample_norm = w_sample@kernel_sample@w_sample

            top =  sample_norm - 2*w_full@kernel_cross@w_sample
            
            bottom = full_norm

            loss+= 1 + top/bottom
        return loss/n

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
    
    def sample(self, X, sample_proportion):
        
        N = X.shape[0]
        N_sample = int(np.ceil(N*sample_proportion))
        
        idx = np.arange(N)
        
        np.random.shuffle(idx)
        
        idx_sample = idx[: N_sample]
        
        return idx_sample
    
        
        
    
    def fit(self, X, Y, optimize = False, optimizer = "SGD", iterations = 1000, learning_rate = 0.1, parameters = None):
        
        self.X_train = np.copy(X)
        self.Y_train = np.copy(Y)
        
        if parameters is None:
            parameters = self.parameters
        else:
            self.parameters = np.copy(parameters)
        #print(parameters)
            
        
        if optimize == True:
            optimized_para = self.optimize_parameters(self.X_train, self.Y_train, parameters, iterations = iterations, optimizer = optimizer, learning_rate= learning_rate)
            self.parameters = optimized_para
            parameters = optimized_para
        
        k_matrix = self.kernel_function(self.X_train,self.X_train,parameters)
        
        self.weights = np.linalg.solve(k_matrix + self.reg*np.eye(k_matrix.shape[0]), Y)
        
    
    def predict(self, X_test):
        
        k_test = self.kernel_function(X_test, self.X_train, self.parameters)
        
        return k_test@self.weights
        
    
    def optimize_parameters(self,  X, Y, initial_parameters, iterations, batch_proportion = 1.0, sample_proportion = 0.5, optimizer = "SGD",
                            learning_rate = 0.1, beta = 0., reg = 1e-5, copy= True, n_samples = 1):
        
        if copy:
            self.X_train = np.copy(X)
            self.Y_train = np.copy(Y)
        
        # Lists that keep track of the history of the algorithm
        self.rho_hist = []
        self.grad_hist = []
        self.para_hist = []
        self.rho_running_mean = []
            
    
        grad_rho = value_and_grad(self.rho_average)


        
        parameters = np.copy(initial_parameters)
        momentum = np.zeros(parameters.shape, dtype = "float")
            

            
        for i in range(iterations):
    
                    

                
    
                    
                # Changes parameters according to SGD rules
                if optimizer == "SGD":
                    
                    rho, grad_mu = grad_rho(parameters, X, Y, reg = self.reg, n = n_samples, batch_proportion = batch_proportion)
                    
                    if  rho < 0 - 1e-5:
                        print("Warning, rho <0: ", rho)
                    else:
                        parameters -= learning_rate * grad_mu
                        
                
                # Changes parameters according to Nesterov Momentum rules     
                elif optimizer == "Nesterov":
                    rho, grad_mu = grad_rho(parameters  - learning_rate * beta * momentum, X, Y, reg = self.reg, n = n_samples, sample_proportion= sample_proportion)
                    
                    if  rho < 0 - 1e-5:
                       print("Warning, rho <0: ", rho)
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
        
        self.early_stopping_parameter = best_parameter
        
        
        
        
        self.rho_hist = np.array(self.rho_hist)
        self.para_hist = np.array(self.para_hist)
        self.rho_running_mean = np.array(running_loss)
        return parameters
        




#%%
if __name__ == "__main__": 
    
    import matplotlib.pyplot as plt
    
    noise_level = 0.2


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
    
    
    # #%%
    
    # X = np.arange(10)[:, None]
    
    # Y = np.arange(10)[:, None]+ 10
    
    # #%%
    
    # print(GP.rho_average(np.array([1e5]), X, Y, batch_proportion=1.0, n = 5))
    
    
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
    
    # #%%
    # K = GP.kernel_function(X, X, sigma)
    # print(Y.T@np.linalg.solve(K + 1e-1*np.eye(K.shape[0]), Y))
    
    
    # c = np.linalg.solve(K + 1e-1*np.eye(K.shape[0]), Y)
    # print(c.T@K@c)
    #%%
    
    # K = GP.kernel_function(X, X, sigma)
    # alpha = np.linalg.solve(K + 1e-10*np.eye(K.shape[0]), Y)
    
    # sample = np.arange(100)
    
    
    # K_sample =GP.kernel_function(X[sample], X[sample], sigma)
    # beta = np.linalg.solve(K_sample + 1e-10*np.eye(K_sample.shape[0]), Y[sample])
    
    
    # K_cross = GP.kernel_function(X, X[sample], sigma)
    
    # print(alpha@K_cross@beta)
    
    # print(Y[sample]@np.linalg.solve(K_sample + 1e-10*np.eye(K_sample.shape[0]), Y[sample]))
    
    #%%
    
    # sample_idx = GP.sample(X, 0.5)
    # sample_idx = np.sort(sample_idx)
    
    # #%%
    
    # print(GP.rho(sigma, X,Y, sample_idx, reg = 1e-5))
    # print(GP.rho_true(sigma, X,Y, sample_idx, reg = 1e-5))
    
    # #%%
    # g = value_and_grad(GP.rho)
    
    # print(g(sigma, X, Y, sample_idx, reg = 1e-8))
    # #print(GP.rho(sigma, X,Y, sample_idx))
    
    # g_true = value_and_grad(GP.rho_true)
    
    # print(g_true(sigma, X, Y, sample_idx, reg = 1e-8))
    
    
    

    
    
    #%% Use optimized parameters
    
    
    sigma = np.array([1.0])
    GP =  GPRegressor(kernel_RBF, sigma)
    
    #GP.fit(X, Y, parameters=np.array([5.0]), optimize = True)
    
    
    opt_param = GP.optimize_parameters(X, Y, sigma, 1000, learning_rate = 0.1, optimizer = "SGD", n_samples= 1)
    GP.fit(X, Y, parameters=opt_param)
    
    
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
    


    

    
    
    
    
    
    
    
                
            
                
                
                
                
        