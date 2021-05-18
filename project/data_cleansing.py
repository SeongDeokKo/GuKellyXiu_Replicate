# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:21:22 2021

@author: yhkim
@co-author : Seongdeok Ko
"""
#%% import modules 
import numpy as np
import pandas as pd 
from collections import Counter
#%% 
# Read Data 
# ==============================================================================
#   Dacheng Xiu Web-site, size : 3,760,208*97,  97 = permno, date, 94variables, SIC-code
# ==============================================================================
raw_data = pd.read_csv('datashare.csv')

# sorting with 'date' value
raw_data[raw_data.columns[2:]] = raw_data[raw_data.columns[2:]].astype('float32')
raw_data = raw_data.sort_values(by = ['DATE', 'permno'], ascending = True) 
#%%
## Step 1

# for index and concatenate, extract date 
date_raw_col = raw_data['DATE']
result_raw = Counter(date_raw_col)
date_raw = list(result_raw.keys())  # unique Date
num_raw = list(result_raw.values()) # 각 Date에 따른 sample 숫

# Extract 10 years (120 months). 2007.1~2016.12
# you can change the all_sample_period
#all_sample_months = 120 
#sum_months = 0
#for i in range(1,all_sample_months+1):
#    sum_months += num_raw[-i]
    
#firm_level_data = raw_data.iloc[-sum_months:,:]    # (707,576 * 97)

#del raw_data
#del date_raw_col
#del result_raw
#del date_raw
#del num_raw


# ==============================================================================
#   Using All data years, size : 707,576*97,  97 = permno, date, 94variables, SIC-code 
#   Missing values: use cross-sectional median at each month 
# ==============================================================================
firm_level_data = raw_data 
date = firm_level_data['DATE']
result = Counter(date)
date_firm = list(result.keys())
date_num_firm = list(result.values())


# for missing values 
for i in range(len(date_raw)):
    if i==0:
        sum_firm = 0
    else:
        sum_firm = sum(date_num_firm[:i])
    
    data = firm_level_data.iloc[sum_firm:(sum_firm + date_num_firm[i]), :]
    
    # except (permno, Date, Sic2), replace 'nan' with median   
    for j in range(2,96):
        replacing = data.iloc[:,j].fillna(data.iloc[:,j].median())
        
        if j==2:
            new_data = pd.concat([data.iloc[:,:2], replacing], axis=1)
            #print(new_data.shape)
        else:
            new_data = pd.concat([new_data, replacing], axis=1)
    
    # concat sic2 column
    new_data = pd.concat([new_data, data.iloc[:,-1]], axis=1)
    #print(new_data.shape)
    
    if i==0:
        new_firm_data = new_data
    else:
        new_firm_data = pd.concat([new_firm_data, new_data], axis=0)



del firm_level_data  
del raw_data
num_firm_charac = 94
new_firm_data[new_firm_data.columns[0:2]] = new_firm_data[new_firm_data.columns[0:2]].astype(np.int32)
#%%

# ==============================================================================
#   last column 'sic2' : 74 SIC number > change to dummy variable  (707,576 * 74) 
# ==============================================================================
sic2 = new_firm_data['sic2'].to_numpy()

sic_dummy = np.full((new_firm_data.shape[0],74),0)  # 74 sic code

for i in range(new_firm_data.shape[0]):
    for j in range(1,75):
        if sic2[i]==j:
            sic_dummy[i,j-1]=1

# for naming columns
sic_col = [] 
for i in range(1,75):
    sic_col.append('sic_dummy'+str(i))

sic_dummy = pd.DataFrame(data = sic_dummy, columns = sic_col)
sic_dummy = sic_dummy.astype('int8')
#%%
w_o_sic = new_firm_data.iloc[:,:-1]

all_X = pd.concat([w_o_sic.reset_index(drop=True), sic_dummy.reset_index(drop=True)], axis=1)


del sic_dummy 
del w_o_sic
del sic2

#%%

# ==============================================================================
#   Welch & Goyal -  make 8 variables we need
# ==============================================================================

goyal = pd.read_excel('PredictorData2019.xlsx',sheet_name='Monthly')

goyal = goyal.iloc[1034:1752,:]   #From 195703~ 201612
num_macro = 8

macro_data = np.full((goyal.shape[0],num_macro),np.nan)

macro_data[:,0] = np.log(goyal['D12'] / goyal['Index'])  # d/p
macro_data[:,1] = np.log(goyal['E12'] / goyal['Index'])  # e/p
macro_data[:,2] = goyal['b/m']                   # B/M
macro_data[:,3] = goyal['ntis']                  # net equity expansion
macro_data[:,4] = goyal['tbl']                   # Treasury-bill rate
macro_data[:,5] = goyal['lty'] - goyal['tbl']    # term-spread
macro_data[:,6] = goyal['BAA'] - goyal['AAA']    # default-spread
macro_data[:,7] = goyal['svar']                  # stock variance

col_macro = ['d/p', 'e/p', 'b/m', 'ntis', 'tbl', 'tms', 'dfy', 'svar']

macro_data = pd.DataFrame(data = macro_data, columns = col_macro) # size : 120*8
goyal[goyal.columns[0]] = goyal[goyal.columns[0]].astype('int32')
goyal[goyal.columns[1:]]  = goyal[goyal.columns[1:]].astype('float32')

#%%


# ==============================================================================
#   Make Interaction term (707,576 * (94*8) ) = 707,576 * 752
# ==============================================================================

# num_firm_charac = 94, num_macro = 8

interact = np.full((new_firm_data.shape[0], num_firm_charac*8), 0,dtype = np.float32)
       
#%%
for i in range(new_firm_data.shape[0]):
    if i%1000 ==0:
        print(i)
   
    for j in range(len(date_firm)):
        if new_firm_data.iloc[i,1] == date_firm[j]:    # 1 means : 2nd columnd 'DATE'
            interact[i,0:94] = new_firm_data.iloc[i,2:-1].to_numpy().reshape(1,-1) * macro_data.iloc[j,0]
            interact[i,94:188] = new_firm_data.iloc[i,2:-1].to_numpy().reshape(1,-1) * macro_data.iloc[j,1]
            interact[i,188:282] = new_firm_data.iloc[i,2:-1].to_numpy().reshape(1,-1) * macro_data.iloc[j,2]
            interact[i,282:376] = new_firm_data.iloc[i,2:-1].to_numpy().reshape(1,-1) * macro_data.iloc[j,3]
            interact[i,376:470] = new_firm_data.iloc[i,2:-1].to_numpy().reshape(1,-1) * macro_data.iloc[j,4]
            interact[i,470:564] = new_firm_data.iloc[i,2:-1].to_numpy().reshape(1,-1) * macro_data.iloc[j,5]
            interact[i,564:658] = new_firm_data.iloc[i,2:-1].to_numpy().reshape(1,-1) * macro_data.iloc[j,6]
            interact[i,658:] = new_firm_data.iloc[i,2:-1].to_numpy().reshape(1,-1) * macro_data.iloc[j,7]
            break

del new_firm_data
#%%

# make column names for interact term
col_stock = list(new_firm_data.columns) 
col_stock = col_stock[2:-1] # for 94 variables

col_interact = []
for i in col_macro:
    for j in col_stock:
        col_interact.append(str(j)+'_'+str(i))
        
interact = pd.DataFrame(data = interact, columns = col_interact)

del macro_data 
del col_stock

#%%
# ==============================================================================
#   All_X  = 707,576 * { 2(permno, date) + 94(firm-level) + 74(sic_dummy)+ 752(interact)  } = 707,576 * 922 
# ==============================================================================
all_X = pd.concat([all_X.reset_index(drop=True), interact.reset_index(drop=True)], axis=1)
# 여기서 데이터가 너무 커서 안돌아감. 
# 21.2 기가가 필요하다고 나옴. 
del interact

# all_X.to_csv('all_x_10years.csv', index = False)

# 처음 ~ 여기까지 35분정도 걸림.


# ==============================================================================
#   Return (Y variable) from  CRSP - HPR monthly (2007.2 ~ 2017.1) 
#   match Y variable to X
#   if X : 2007.01, search the return of firm in 2007.02   (using for-loop to find permno)
#   ...
#   ...
#   if X : 2016.12, search the return of firm in 2017.01 
# ==============================================================================
ret = pd.read_csv('hpr_10years.csv')             # size : 840,547 * 3 
ret = ret.sort_values(by=['date','PERMNO'])
#print(ret.shape)

# delete rows that have 'nan' / 'B' / 'C' in 'RET' column
ret = ret.dropna()
ret = ret[ret.RET != 'B']
ret = ret[ret.RET != 'C']
#print(ret.shape)                                 # size : 816,857 * 3

date_= ret['date']
result_= Counter(date_)
date_ret = list(result_.keys())
date_num_ret = list(result_.values())


# to change excess return 
tbl_rate = goyal['tbl']
tbl_rate_concat = []

for i in range(len(date_ret)):
    for j in range(date_num_ret[i]):
        tbl_rate_concat.append(tbl_rate.iloc[i]*(30/365))   # monthly

tbl_rate_concat = np.asarray(tbl_rate_concat).reshape(-1,1)
ret_concat = ret['RET'].to_numpy(dtype=float).reshape(-1,1)   

excess_ret = ret_concat - tbl_rate_concat
excess_ret = pd.DataFrame(data = excess_ret, columns = ['excess_ret'] )

w_o_ret = ret.iloc[:,:2]
ret = pd.concat([w_o_ret.reset_index(drop=True), excess_ret.reset_index(drop=True)], axis=1)  # size : 816,857 * 3

del goyal
del w_o_ret




# Want to make 707,576*1 Y value 
# Since data is too long, it takes a lot of time 
for i in range(len(date_firm)):   
    match_ret = np.full((date_num_firm[i],1),np.nan)   
    
    if i==0:
        sum_stock = 0
        sum_ret = 0
    else: 
        sum_stock = sum(date_num_firm[:i])
        sum_ret = sum(date_num_ret[:i])  
    
    last_index = 0
    for j in range(date_num_firm[i]):
        print(i,j)
        for k in range(last_index, date_num_ret[i]):
            if new_firm_data.iloc[sum_stock+j,0] == ret.iloc[sum_ret+k,0]:
                match_ret[j,0] = ret.iloc[sum_ret+k,2]
                last_index = k+1 
                break
    
    if i ==0:
        matched_ret = match_ret
    else: 
        matched_ret = np.concatenate((matched_ret, match_ret), axis = 0)
        
      
matched_ret = pd.DataFrame(data = matched_ret, columns = ['excess_ret'])

# matched_ret.to_csv('all_y_10years.csv', index = False)



# ==============================================================================
#   Make full data X, y  (10 year)
# ==============================================================================


all_data = pd.concat([all_X.reset_index(drop=True), matched_ret.reset_index(drop=True)], axis=1)  # 707,576 * 923 
print(all_data.shape)

all_data = all_data.dropna(subset = ['excess_ret'])      #
print(all_data.shape)

all_data.to_csv('all_data_10years.csv', index = False)

del all_X
del matched_ret

"""
# ==============================================================================
#   Make full data X, y  (1 year)
# ==============================================================================

date_10_col = all_data['DATE']
result_10 = Counter(date_10_col)
date_10 = list(result_10.keys())
num_10 = list(result_10.values())

data_1year = all_data.iloc[-sum(num_10[-12:]):,:]

data_1year.to_csv('data_1year.csv', index = False)
"""