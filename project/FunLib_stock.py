# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 18:58:20 2021

@author: yhkim
@co-author : Seongdeok Ko
"""

# =========================================================================
#   R2oos 
# =========================================================================

def R2OOS(y_true, y_forecast):
    
    import numpy as np
   
    SSres = np.nansum(np.square(y_true-y_forecast))
    SStot = np.nansum(np.square(y_true))

    return 1-SSres/SStot



# =========================================================================
#   PCR, 94 + dummy variable(no intersection term), Use cross-validation to select the number of PCA components  
# =========================================================================

def Pca_regression(X,Y,numpc,num_t_v):
    # numpc (list) : # of principal component ex[3,4,5,6,7]
    # num_t_v (list) : # of training set & cross-val set   ex[100, 10]
    # X consists of Traing, Val and Test set
    
    import numpy as np 
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    
    num_train = num_t_v[0]
    num_val = num_t_v[1]
    num_test = X.shape[0] - (num_train + num_val)
    
    # Split data into training and test
    X_train = X[:num_train,:]
    Y_train = Y[:num_train,:]
    
    X_val = X[num_train:(num_train+num_val),:]
    Y_val = Y[num_train:(num_train+num_val),:]
    
    X_test = X[(num_train+num_val):,:]
    
       
    # Scale Inputs for Training
    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    
    X_val_scaled = X_scaler.transform(X_val)
    X_test_scaled = X_scaler.transform(X_test)
    
    
    # use cross-validation mean-squared-error to determine the number of component 
    mse = np.full((len(numpc),1),np.nan)

    for i in range(len(numpc)):
        pca = PCA(n_components = numpc[i])
        principalComponents = pca.fit_transform(X_train_scaled)
        
        X_val_weighted = pca.transform(X_val_scaled)
        
        line_fitter = LinearRegression()
        line_fitter.fit(principalComponents, Y_train)
        
        Ypred_val = np.full((num_val,1),np.nan, dtype = np.float32)
        for j in range(num_val):
            Ypred_val[j,0] = line_fitter.predict(X_val_weighted[j,:].reshape(1,-1))
                   
        mse[i,0] = mean_squared_error(Y_val.reshape(-1), Ypred_val.reshape(-1))
    
    
    argmin_numpc = numpc[np.argmin(mse)]
    
    pca = PCA(n_components = argmin_numpc)
    principalComponents = pca.fit_transform(X_train_scaled)
    
    X_test_weighted = pca.transform(X_test_scaled)
    
    line_fitter = LinearRegression()
    line_fitter.fit(principalComponents, Y_train)
        
    Ypred_test = np.full((num_test,1),np.nan, dtype = np.float32)
    for j in range(num_test):
        Ypred_test[j,0]=line_fitter.predict(X_test_weighted[j,:].reshape(1,-1))
        
          
    return Ypred_test, argmin_numpc




# =========================================================================
#   PLS, 94 + dummy variable(no intersection term), Use cross-validation to select the number of components  
# =========================================================================

def Pls_regression(X,Y,numpls,num_t_v):
    # numpls (list) : # of component ex[3,4,5,6,7]
    # num_t_v (list) : # of training set & cross-val set   ex[100, 10]
    # X consists of Traing, Val and Test set
    
    import numpy as np 
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    
    num_train = num_t_v[0]
    num_val = num_t_v[1]
    num_test = X.shape[0] - (num_train + num_val)
    
    # Split data into training and test
    X_train = X[:num_train,:]
    Y_train = Y[:num_train,:]
    
    X_val = X[num_train:(num_train+num_val),:]
    Y_val = Y[num_train:(num_train+num_val),:]
    
    X_test = X[(num_train+num_val):,:]
    
       
    # Scale Inputs for Training
    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    
    X_val_scaled = X_scaler.transform(X_val)
    X_test_scaled = X_scaler.transform(X_test)
    
    
    # use cross-validation mean-squared-error to determine the number of component 
    mse = np.full((len(numpls),1),np.nan)

    for i in range(len(numpls)):
        pls = PLSRegression(n_components = numpls[i])
        pls.fit(X_train_scaled, Y_train)
                
        Ypred_val = np.full((num_val,1),np.nan)
        for j in range(num_val):
            Ypred_val[j,0]=pls.predict(X_val_scaled[j,:].reshape(1,-1))          
        
        mse[i,0] = mean_squared_error(Y_val.reshape(-1), Ypred_val.reshape(-1))
    
    
    argmin_numpls = numpls[np.argmin(mse)]
    
    pls = PLSRegression(n_components = argmin_numpls)
    pls.fit(X_train_scaled, Y_train)
                
    Ypred_test = np.full((num_test,1),np.nan)
    for j in range(num_test):
        Ypred_test[j,0]=pls.predict(X_test_scaled[j,:].reshape(1,-1))          
              
    
    return Ypred_test, argmin_numpls



# =========================================================================
#  elastic-net, Loss : mse + penalty, 94 + dummy variable(no intersection term), hyperparameter tuning
# ========================================================================= 

def elastic_net(X,Y,num_t_v):
    # num_t_v (list) : # of training set & cross-val set   ex[100, 10]
    # X consists of Traing, Val and Test set
    
    import numpy as np
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import PredefinedSplit
    
    num_train = num_t_v[0]
    num_val = num_t_v[1]
    num_test = X.shape[0] - (num_train + num_val)
    
    # Split data into training and test
    X_train = X[:(num_train+num_val),:]   # train + validation
    Y_train = Y[:(num_train+num_val),:]   # train + validation
    
    X_test = X[(num_train+num_val):,:]
    
       
    # Scale Inputs for Training
    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)

    X_test_scaled = X_scaler.transform(X_test)
    
    # pre-define validation 
    test_fold =  np.concatenate(((np.full((num_train),-1),np.full((num_val),0))))
    ps = PredefinedSplit(test_fold.tolist())
    
    # fit & predict 
    model = ElasticNetCV(cv=ps, max_iter=5000, n_jobs=-1, l1_ratio=[.1, .3, .5, .7, .9], 
                         random_state=42)
    model = model.fit(X_train_scaled, Y_train.reshape(-1))
    
    Ypred_test = np.full((num_test,1),np.nan)
    for j in range(num_test):
        Ypred_test[j,0]=model.predict(X_test_scaled[j,:].reshape(1,-1))
        
    
    return Ypred_test



# =========================================================================
#   Generalized-linear, 94 + dummy variable(no intersection term), Use cross-validation to select the number of PCA components  
# =========================================================================
# Loss ftn : MSE
# We use Lasso (Not group Lass) 
# include spline series of order 2 
# number of knots = [3,5,7...] and we choose the only one that minimize cross-validation MSE 
# we set knots by using linspace(col.mean-2*col.std, col.mean+2*col.std, # knots)
# for example if we use 3 knots, the # of variables is 94(order1) + 94*3(order 2) + dummy(74) = 450 

def general_linear(X,Y,num_t_v, num_knots):
    # num_t_v (list) : # of training set & cross-val set   ex[100, 10]
    # X consists of Traing, Val and Test set
    
    import numpy as np 
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import PredefinedSplit
    from sklearn.linear_model import LassoCV
    
    num_train = num_t_v[0]
    num_val = num_t_v[1]
    num_test = X.shape[0] - (num_train + num_val)
       
    mse = np.full((len(num_knots),1),np.nan)
    Ypred_test = np.full((len(num_knots),num_test,1),np.nan)
    
    for i in range(len(num_knots)):
        
        X_temp = X
        
        # 94 variables > make spline series of order 2
        for j in range(94):
            
            # make knots
            std_train = np.std(X[:num_train,j])
            mean_train = np.mean(X[:num_train,j])           
            
            knots = np.linspace(mean_train-2*std_train, mean_train+2*std_train, num_knots[i])
            
            # add (variable - knots)**2 column
            for k in knots:
                add_col = ((X[:,j]-k)**2).reshape(-1,1)
                X_temp = np.concatenate((X_temp, add_col), axis=1)
        
        print(X_temp.shape)
        
        # Split data into training and test
        X_train = X_temp[:(num_train+num_val),:]   # train + validation
        Y_train = Y[:(num_train+num_val),:]   # train + validation
        
        X_test = X_temp[(num_train+num_val):,:]
        
        # Scale Inputs for Training
        X_scaler = StandardScaler()
        X_train_scaled = X_scaler.fit_transform(X_train)
        
        X_test_scaled = X_scaler.transform(X_test)
        
        # pre-define validation 
        test_fold =  np.concatenate(((np.full((num_train),-1),np.full((num_val),0))))
        ps = PredefinedSplit(test_fold.tolist())
        
        # we use cross-val to find best 'alpha'(penalty term in loss function)
        model = LassoCV(cv=ps, max_iter=3000, n_jobs=-1, random_state=42)
        model = model.fit(X_train_scaled, Y_train.reshape(-1))

        
        # to choose # of knots, calculate mse of validation set
        Ypred_val = np.full((num_val,1),np.nan)
        for j in range(num_val):
            Ypred_val[j,0]=model.predict(X_train_scaled[num_train+j,:].reshape(1,-1))
            
        mse[i,0] = mean_squared_error(Y[num_train:(num_train+num_val),:].reshape(-1), Ypred_val.reshape(-1))
        
        # predic test set 
        for j in range(num_test):
            Ypred_test[i,j,0]=model.predict(X_test_scaled[j,:].reshape(1,-1))
    
    
    # choose knots that minimize mse in validation
    argmin_index = np.argmin(mse)
    
    print(argmin_index)
    
    Ypred_test_final = Ypred_test[argmin_index,:,:].reshape(-1,1)
    
    return Ypred_test_final
    
    
    
# =========================================================================
#                   Random-forest  intersection term  (hyperparameter tuning)
# =========================================================================

# in random forest, no need to scale x-variable 
def Random_Forest(X,Y,num_t_v):
    
    # use intersection 
    # We can set many hyper-parameter in Random forest model. 
    # 1. the depth of the individual trees(max_depth),  
    # 2. the size of the randomly selected sub-set of predictors (max_features) 
    # 3. the number of trees(n_estimators)  
    # 4. min_samples_split   5. min_samples_leaf  6. max_samples  ..........etc....
    # 
    # Here we only consider 1(max_depth), 2(max_features) for hyper-parameter tuning
    
    # for detail, refer to algorithms in details 'ML_supp' file
    # we set hyper-parameter following Table A.5
    
    
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import PredefinedSplit
    
    num_train = num_t_v[0]
    num_val = num_t_v[1]
    num_test = X.shape[0] - (num_train + num_val)
    
    # Split data into training and test
    X_train = X[:(num_train+num_val),:]   # train + validation
    Y_train = Y[:(num_train+num_val),:]   # train + validation
    
    X_test = X[(num_train+num_val):,:]
    
    # pre-define validation 
    test_fold =  np.concatenate(((np.full((num_train),-1),np.full((num_val),0))))
    ps = PredefinedSplit(test_fold.tolist())
    
    
    # Set hyper-parameter candidate 
    max_depth = [3,4,5]
    max_features = [10,20,50,100]
    
    grid_param = {'max_depth':max_depth, 'max_features':max_features}     

    RFR = RandomForestRegressor(n_estimators=300, bootstrap = True, n_jobs=-1, random_state=42, max_samples = 0.5)
        
    RFR_grid = GridSearchCV(estimator=RFR, param_grid=grid_param, n_jobs=-1, cv=ps)
    RFR_grid.fit(X_train, np.ravel(Y_train))
        
    Ypred_test = np.full((num_test,1),np.nan)
    for j in range(num_test):
        Ypred_test[j,0]=RFR_grid.predict(X_test[j,:].reshape(1,-1))
        
    return Ypred_test




 
# =========================================================================
#                  GBRT  intersection term - huber loss  (hyperparameter tuning)
# =========================================================================

# in GBRT, no need to scale x-variable 
def Gradient_boosting(X,Y,num_t_v):
    # Huber-Loss ftn, use intersection 
    # We can set many hyper-parameter in GBRT model.
    # Here we only consider 'max_depth', 'n_estimators', 'learning_rate' for hyper-parameter tuning
    # for detail, refer to algorithms in details 'ML_supp' file
    # we set hyper-parameter following Table A.5
    
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import PredefinedSplit
    
    num_train = num_t_v[0]
    num_val = num_t_v[1]
    num_test = X.shape[0] - (num_train + num_val)
    
    # Split data into training and test
    X_train = X[:(num_train+num_val),:]   # train + validation
    Y_train = Y[:(num_train+num_val),:]   # train + validation
    
    X_test = X[(num_train+num_val):,:]
    
    # pre-define validation 
    test_fold =  np.concatenate(((np.full((num_train),-1),np.full((num_val),0))))
    ps = PredefinedSplit(test_fold.tolist())
    
    # Set hyper-parameter candidate 
    max_depth = [4,5]
    n_estimators=[100,300]
    learning_rate = [0.1, 0.01]
    
    grid_param = {'max_depth':max_depth, 'n_estimators':n_estimators, 'learning_rate':learning_rate}
    
    GBR = GradientBoostingRegressor(loss='huber', random_state=42, 
                                    max_features = 'auto', n_iter_no_change = None)
    
    GBR_grid = GridSearchCV(estimator=GBR, param_grid=grid_param, n_jobs=-1, cv=ps)
    GBR_grid.fit(X_train, np.ravel(Y_train))
    
    Ypred_test = np.full((num_test,1),np.nan)
    for j in range(num_test):
        Ypred_test[j,0]=GBR_grid.predict(X_test[j,:].reshape(1,-1))
        
    return Ypred_test





# =========================================================================
#             Neural Net, No hyper parameter tuning, intersection term
# =========================================================================

def Neural_net(X, Y, num_t_v, archi, epoch):
    # archi : # of neurons in hyden layer 
    # Use mini-batch, MSE Loss 
    # Linear > Relu > Batch-normalization > linear > Relu > BN .... > linear 
    # Adam optimizer, Learning decay, Early-stopping
    
    import torch 
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np  
    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader
    
    
    #seed 
    torch.manual_seed(1597)
    np.random.seed(1597)
    
    num_train = num_t_v[0]
    num_val = num_t_v[1]
    num_test = X.shape[0] - (num_train + num_val)
    
    # Split data into training and test
    X_train = X[:num_train,:]
    Y_train = Y[:num_train,:]
    
    X_val = X[num_train:(num_train+num_val),:]
    Y_val = Y[num_train:(num_train+num_val),:]
    
    X_test = X[(num_train+num_val):,:]
    
       
    # Scale Inputs for Training
    X_scaler = MinMaxScaler(feature_range=(-1,1))
    X_train_scaled = X_scaler.fit_transform(X_train)
    
    X_val_scaled = X_scaler.transform(X_val)
    X_test_scaled = X_scaler.transform(X_test)
    
    # from np.array > torch tensor 
    X_train_scaled = torch.tensor(X_train_scaled)    
    X_val_scaled = torch.tensor(X_val_scaled)
    X_test_scaled = torch.tensor(X_test_scaled)
    
    Y_train = torch.tensor(Y_train)
    Y_val = torch.tensor(Y_val)
    
    
    # dataset
    train_dataset = TensorDataset(X_train_scaled, Y_train)   
    valid_dataset = TensorDataset(X_val_scaled, Y_val)
    
    trainloader = DataLoader(train_dataset, batch_size=2048, shuffle=True, drop_last=True)
    validloader = DataLoader(valid_dataset, batch_size=2048, shuffle=True, drop_last=True)
    
    # define Network 
    class NN_fwd_model(nn.Module):
        
        def __init__(self, X_dim, Y_dim, archi):
            
            super(NN_fwd_model, self).__init__()
            
            n = len(archi)
            self.nn_module = torch.nn.Sequential()
            
            for i in range(n):
                if i==0:                 
                    self.nn_module.add_module('linear'+str(i+1), nn.Linear(X_dim, archi[i]))
                    self.nn_module.add_module('Relu'+str(i+1), nn.ReLU())
                    self.nn_module.add_module('BN'+str(i+1), nn.BatchNorm1d(archi[i]))
                    
                else:                  
                    self.nn_module.add_module('linear'+str(i+1), nn.Linear(archi[i-1], archi[i]))
                    self.nn_module.add_module('Relu'+str(i+1), nn.ReLU())
                    self.nn_module.add_module('BN'+str(i+1), nn.BatchNorm1d(archi[i]))                    
            
            # for output layer
            self.lastlinear = nn.Linear(archi[-1], Y_dim)
                    
            # Using He-initilization 
            for m in self.nn_module:
                if isinstance(m,nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            
            nn.init.kaiming_normal_(self.lastlinear.weight, nonlinearity="relu")
                    
         
        def forward(self, X_train_scaled):
           y_hat = self.nn_module(X_train_scaled)
           y_hat = self.lastlinear(y_hat)
           
           return y_hat
       
        
    model = NN_fwd_model(X_train_scaled.shape[1], Y_train.shape[1], archi)
    print(model)
    
    
    # define loss ftn 
    loss_ftn = torch.nn.MSELoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                lr_lambda=lambda epoch: 0.99 ** epoch)


    
    min_val_loss = np.Inf
    epochs_no_improve = np.nan

    for i in range(epoch):       
        
        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []
  
        model.train()
        for (batch_X, batch_Y) in trainloader:
            
            optimizer.zero_grad()
           
            # compute the model output
            trained_y = model(batch_X.float())            
            
            # calculate loss

            regularization_loss = 0
            for param in model.parameters():
                regularization_loss += torch.sum(abs(param))

            loss = loss_ftn(trained_y, batch_Y.float())        
            loss = loss + 0.005 * regularization_loss    # 0.005 : L1 Penalty
            # credit assignment
            loss.backward()
            
            # update model weights
            optimizer.step()
            scheduler.step()
            
            train_losses.append(loss.item())
        
        model.eval()
        for (batch_X_val, batch_Y_val) in validloader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(batch_X_val.float())
            # calculate the loss
            loss = loss_ftn(output, batch_Y_val.float())
            # record validation loss
            valid_losses.append(loss.item())         
            
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)        
        
        if i % 5 ==0:
            print('the epoch number ' + str(i) + ' (train_loss) : ' + str(train_loss))
            print('the epoch number ' + str(i) + ' (valid_loss) : ' + str(valid_loss))
        
        # Early-stopping
        if valid_loss < min_val_loss:
             epochs_no_improve = 0
             min_val_loss = valid_loss
             torch.save(model.state_dict(), 'best_model_NN.pt')
  
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve > 20:
            print('Early stopping!' )
            break
        else:
            continue
        
    
    model.load_state_dict(torch.load('best_model_NN.pt'))
    model.eval()
    Ypred_test = np.full((num_test,1),np.nan)
    for j in range(num_test):
        Ypred_test[j,0]=model(X_test_scaled[j,:].float().unsqueeze(0))
        
        
    return Ypred_test
