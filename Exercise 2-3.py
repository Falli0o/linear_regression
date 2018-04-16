
# coding: utf-8

# In[ ]:


from __future__ import division
get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import math
import multivarlinreg
import rmse


# In[ ]:


#Linear regression
red_train = np.loadtxt('redwine_training.txt')
red_test = np.loadtxt('redwine_testing.txt')
red_train_data = red_train[:, :11]
red_train_score = red_train[:, 11]
red_test_data = red_test[:, :11]
red_test_score = red_test[:, 11]
#red_train.shape


# In[ ]:


"""
def multivarlinreg(data, ground_truth):
    #data = full_data[:, :-1]
    X = np.hstack((data, np.repeat(1, data.shape[0]).reshape(-1, 1)))
    X_T_X = np.dot(X.T, X)
    # if full-rank matrix or positive definite matrix:
    #check if it invertible
    if np.linalg.det(X_T_X) != 0:
        inverse = np.linalg.inv(X_T_X)
        w = np.dot(np.dot(inverse, X.T), ground_truth) #w0 at the last column
        #print w
        return w
    else:
        print "use other method"
        """


# In[ ]:


#only contains the first feature (fixed acidity)
train_fixed_acidity = red_train_data[:, 0].reshape(-1, 1)
train_w_acidity = multivarlinreg.multivarlinreg(train_fixed_acidity, red_train_score)
train_w_acidity
#the propotion of acidity is not very high; bias is very large for it???
#actually we can not use it to predivt the wine's quality very well
#array([0.05035934, 5.2057261 ])


# In[ ]:


#physiochemical
w_all = multivarlinreg.multivarlinreg(red_train_data, red_train_score)
w_all.shape
np.set_printoptions(suppress=True)
w_all
#positive relate negative relation
#the first weight for acidity is changed 
#Some features play important roles in wine's quality. Some features are negatively related.


# In[ ]:


"""#Exercise 3 (Evaluating Linear Regression).
def rmse(predicted_value, ground_truth):
    diff = ground_truth - predicted_value
    diff_square = np.dot(diff, diff)
    #rmse = np.sqrt(np.divide(diff_square, ground_truth.shape[0]))
    rmse = np.sqrt(diff_square/ground_truth.shape[0])
    return rmse
    """


# In[ ]:


#1-dimensional input variables using the training set
#first feature for the test set
test_fixed_acidity = red_test_data[:, 0].reshape(-1, 1)
test_X_acidity = np.hstack((test_fixed_acidity, np.repeat(1, test_fixed_acidity.shape[0]).reshape(-1, 1)))
predicted_score_acidity = np.dot(test_X_acidity, train_w_acidity.T)
#predicted_score_acidity = predicted_value(train_fixed_acidity, test_fixed_acidity, red_test_score)
rmse.rmse(predicted_score_acidity, red_test_score)
#0.7860892754162216


# In[ ]:


#full 11-dimensional input variables
test_X = np.hstack((red_test_data, np.repeat(1, red_test_data.shape[0]).reshape(-1, 1)))
predicted_score = np.dot(test_X, w_all.T)
rmse.rmse(predicted_score, red_test_score)
#0.644717277241364

