# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 14:33:14 2021

@author: yhkim
"""

import numpy as np
import pandas as pd 
from collections import Counter
import FunLib_stock as FL
import multiprocessing as mp


# Load data (X & y) 2(permno, date) + 94(firm-level) + 74(sic_dummy)+ 752(interact) +1(y) = 923 columns
# X(922) : 201601 ~ 201612, y(1) :201602 ~ 201701
# X-201601 & y-201602 are in the same row. 

data_1year = pd.read_csv('data_1year.csv')

date_1_col = data_1year['DATE']
result_1 = Counter(date_1_col)
date_1 = list(result_1.keys())
num_1 = list(result_1.values())



# Here we use only 1-year data, training-data(8months), cross-validation(1month, if needed), test-data(3months)
# Because the authors predict 1 year(12months) excess-return after training just 1 time,
# we only train 1 time here(using 8 months data)

# ===========================================================================
#     Setting for estimation (according to above period)  
# ===========================================================================
X = data_1year.iloc[:,2:-1]
X_no_inter = data_1year.iloc[:,2:170]     # without intersect terms
y = data_1year.iloc[:,-1]


num_est = 1  # We estimate parameter 1 time, here not use

# if we estimate parameters more than 1 time(i.e using longer data), we should set below # recursively
num_train = sum(num_1[0:8])
num_val = num_1[8]
num_test = sum(num_1[9:])

num_t_v = [num_train, num_val]

y_true = y.iloc[-num_test:].to_numpy().reshape(-1,1)  # for caluclating R2oos

# Computational Ressources: Determine Number of available cores
ncpus = mp.cpu_count()
print("CPU count is: "+str(ncpus))



# ===========================================================================
#     OLS, Loss: MSE, 94 + dummy variable(no intersection term)
# ===========================================================================

from sklearn.linear_model import LinearRegression

X_train_ols = X_no_inter.iloc[:(num_train+num_val),:].to_numpy()
y_train_ols = y.iloc[:(num_train+num_val)].to_numpy()

reg = LinearRegression().fit(X_train_ols, y_train_ols)

Y_pred_ols = np.full((num_test,1),np.nan)

for i in range(num_test):
    Y_pred_ols[i,0] = reg.predict(X_no_inter.iloc[(num_train+num_val+i),:].to_numpy().reshape(1,-1))
    
print('R2OOS, MSE error - Linear regression without intersection terms : ', FL.R2OOS(y_true, Y_pred_ols))    
    



# ===========================================================================
#     OLS, Loss: Huber Loss, 94 + dummy variable(no intersection term)
# ===========================================================================

from sklearn.linear_model import HuberRegressor

reg_huber = HuberRegressor(max_iter=500, alpha=0).fit(X_train_ols, y_train_ols)

Y_pred_ols_huber = np.full((num_test,1),np.nan)

for i in range(num_test):
    Y_pred_ols_huber[i,0] = reg_huber.predict(X_no_inter.iloc[(num_train+num_val+i),:].to_numpy().reshape(1,-1))

print('R2OOS, Huber Loss - Linear regression without intersection terms : ', FL.R2OOS(y_true, Y_pred_ols_huber))




# ===========================================================================
#    PCR, 94 + dummy variable(no intersection term), Use cross-validation to select the number of PCA components
# ===========================================================================

numpc = [3,6,9,12,15,20,25,30]


Y_pred_pca, argmin_numpc = FL.Pca_regression(X_no_inter.to_numpy(), y.to_numpy().reshape(-1,1), numpc, num_t_v)

print('R2OOS, Principal Components Regression - without intersection terms : ', FL.R2OOS(y_true, Y_pred_pca)) 
print('# of principal components : ', argmin_numpc)




# ===========================================================================
#    PLS, 94 + dummy variable(no intersection term), Use cross-validation to select the number of components
# ===========================================================================

numpls = [3,6,9,12,15,20,25,30]

Y_pred_pls, argmin_numpls = FL.Pls_regression(X_no_inter.to_numpy(), y.to_numpy().reshape(-1,1), numpls, num_t_v)

print('R2OOS, Partial Least Square - without intersection terms : ', FL.R2OOS(y_true, Y_pred_pls)) 
print('# of components : ', argmin_numpls)




# =========================================================================
#  elastic-net, Loss : mse + penalty, 94 + dummy variable(no intersection term), hyperparameter tuning
# =========================================================================

Y_pred_elastic = FL.elastic_net(X_no_inter.to_numpy(), y.to_numpy().reshape(-1,1), num_t_v)

print('R2OOS, Elastic-net - without intersection terms : ', FL.R2OOS(y_true, Y_pred_elastic))




# =========================================================================
#   Generalized-linear, 94 + dummy variable(no intersection term), Use cross-validation to select the number of PCA components  
# =========================================================================
# Loss ftn : MSE
# We use Lasso (Not group Lass) 
# include spline series of order 2 
# number of knots = [3,5,7...] and we choose the only one that minimize cross-validation MSE 
# we set knots by using linspace(col.mean-2*col.std, col.mean+2*col.std, # knots)
# for example if we use 3 knots, the # of variables is 94(order1) + 94*3(order 2) + dummy(74) = 450 

num_knots = [3]

Y_pred_general_lin = FL.general_linear(X_no_inter.to_numpy(), y.to_numpy().reshape(-1,1), num_t_v, num_knots)

print('R2OOS, generalized linear - without intersection terms / with knots : ', FL.R2OOS(y_true, Y_pred_general_lin))



   
# =========================================================================
#                   Random-forest,  intersection term  (hyperparameter tuning)
# =========================================================================
# use intersection 
# Here we only consider 1(max_depth), 2(max_features) for hyper-parameter tuning
# for detail, refer to algorithms in details 'ML_supp' file
# we set hyper-parameter following Table A.5

Y_pred_RF = FL.Random_Forest(X.to_numpy(), y.to_numpy().reshape(-1,1), num_t_v)

print('R2OOS, Random-forest - with intersection terms : ', FL.R2OOS(y_true, Y_pred_RF))




#GBRT-with hyperparameter tuning은 Random-forest-with hyperparameter tuning과 다르게 매우 오래걸림 (2시간 이상)



# =========================================================================
#                    GBRT  intersection term - huber loss  (hyperparameter tuning)
# =========================================================================
# Huber-Loss ftn , use intersection 
# Here we only consider 'max_depth', 'n_estimators', 'learning_rate' for hyper-parameter tuning
# for detail, refer to algorithms in details 'ML_supp' file
# we set hyper-parameter following Table A.5

Y_pred_GBR = FL.Gradient_boosting(X.to_numpy(), y.to_numpy().reshape(-1,1), num_t_v)

print('R2OOS, Gradient Boosting Regressor - with intersection terms : ', FL.R2OOS(y_true, Y_pred_GBR))





# =========================================================================
#                  Neural Net, No hyper parameter tuning, intersection term
# =========================================================================
# archi : # of neurons in hyden layer 
# Use mini-batch, MSE Loss 
# Linear > Relu > Batch-normalization > linear > Relu > BN .... > linear 
# Adam optimizer, Learning decay, Early-stopping

archi = [int(X.shape[1]), int(X.shape[1]/2), int(X.shape[1]/4)]
epoch = 500

Y_pred_NN = FL.Neural_net(X.to_numpy(), y.to_numpy().reshape(-1,1), num_t_v, archi, epoch)

print('R2OOS, Neural Net - with intersection terms : ', FL.R2OOS(y_true, Y_pred_NN))



