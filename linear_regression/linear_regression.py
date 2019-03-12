"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean squre error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    
    err = np.mean(np.square(np.dot(X,w)-y),dtype='float64')
    
    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here #
  w=np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
  #####################################################		
  
  
  return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    row,col=X.shape
    combo=np.dot(X.T,X)
    eig_value=np.linalg.eigvals(combo)
    eig_min=np.min(np.abs(eig_value) )
    
    while(eig_min<1.0e-5):
        combo=combo+0.1*np.identity(col)
        
        eig_value=np.linalg.eigvals(combo)
        eig_min=np.min(np.abs(eig_value) )
        
    w=np.dot(np.dot(np.linalg.inv(combo),X.T),y)
    
    #####################################################
   
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
    row,col=X.shape
    trans_matrix=np.dot(X.T,X)+lambd*np.identity(col)
    

 
    
    w=np.dot(  np.dot( np.linalg.inv( trans_matrix ),X.T),y   )    
  #####################################################		
    
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    
    minerr=float('inf')
    bestlambda=0
    
    for i in range(-19,20):
        
        w=regularized_linear_regression(Xtrain,ytrain,10**i)
        error=mean_square_error(w,Xval,yval)
        if(error<minerr):
            minerr=error
            bestlambda=10**i
  
    #####################################################		
    
    return bestlambda
    

###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    tempX=np.array(X)
    for i in range(2,power+1):
        newm=np.power(tempX,i)
        X=np.insert(X,len(X[0]),newm.T,axis=1)
        

    
    #####################################################		
    
    
    
    return X


