
def Neural_net(X, Y, num_tr_cv_te, archi, epoch):
    # archi : # of neurons in hyden layer 
    # Number of Neurons in Hidden Layers follows that of Gu Kelly Xiu
    # Pyramid Structure
    # Use mini-batch, MSE Loss + L2 penalty term (Which is inside Adam)
    # Linear > Relu > Batch-normalization > linear > Relu > BN .... > linear 
    # Adam optimizer, Learning decay, Early-stopping
    # We did not use Ensenble 
    
    import torch 
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np  
    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader
    
    device = torch.device('cuda:0' if USE_CUDA else 'cpu') # To use GPU in GOOGLE COLAB

    #seed 
    torch.manual_seed(1597) 
    np.random.seed(1597)
    
    num_train = num_tr_cv_te[0]
    num_val = num_tr_cv_te[1]
    num_test = num_tr_cv_te[2]
    
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
    
    # from np.array > torch tensor > To GPU
    X_train_scaled = torch.tensor(X_train_scaled).to(device)    
    X_val_scaled = torch.tensor(X_val_scaled).to(device)
    X_test_scaled = torch.tensor(X_test_scaled).to(device)
    
    Y_train = torch.tensor(Y_train).to(device)
    Y_val = torch.tensor(Y_val).to(device)
        
    # dataset
    train_dataset = TensorDataset(X_train_scaled, Y_train)   
    valid_dataset = TensorDataset(X_val_scaled, Y_val)
    
    trainloader = DataLoader(train_dataset, batch_size=4096, shuffle=True, drop_last=True)
    validloader = DataLoader(valid_dataset, batch_size=4096, shuffle=True, drop_last=True)
    


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
    model = model.to(device)
    print(model)
    
    
    # define loss ftn 
    loss_ftn = torch.nn.MSELoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0075, weight_decay = 0.005)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                lr_lambda=lambda epoch: 0.95 ** epoch)
    
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

            loss = loss_ftn(trained_y, batch_Y.float())        
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