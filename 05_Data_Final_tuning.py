# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:51:50 2019

@author: in-gyunAhn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
#Xtrain with birth data and its moving average 
#To find out the relationship between successive years
#and Target data containing birth rate each year


Xtrain=pd.read_csv("https://drive.google.com/uc?export=download&id=1lqSop-lRGpvDHp-vGZ5cjzBLd0AhwkVP",encoding="CP949")
target_data = pd.read_csv('https://drive.google.com/uc?export=download&id=1tuLsS2krwwcaPv6aicjMGmnqdMeRlAfw',encoding="CP949")

#%%
#Merging some data with similar things
real_estate_transaction = Xtrain['apt_transaction']+Xtrain['housing_transaction']
real_estate_transaction = real_estate_transaction.rename('real_estate_transaction')
edu_env = Xtrain['childcare']+Xtrain['education']
edu_env=edu_env.rename('edu_env')

Xtrain = pd.concat([Xtrain,real_estate_transaction],axis=1)
Xtrain = pd.concat([Xtrain,edu_env],axis=1)

#%%
#Scaling some data dividing by population
varlist=['employment','marriage','industry_1','industry_2','industry_3'
         ,'real_estate_transaction']

for i in varlist:
    a = Xtrain[i]/Xtrain['population']
    a = a.rename(i+'_per_person')
    Xtrain=pd.concat([Xtrain,a],axis=1)
    
#%%
# Categorical industry_1 is added
Xtrain['cat_industry_1']=Xtrain['industry_1']
Xtrain.cat_industry_1[Xtrain.cat_industry_1 != 0]=1

#%%
    #EDA
    
#%% 
# Calculate Correlation
Xtrain.columns
varlist = ['apt_midprice','real_estate_transaction_per_person','employment_per_person','edu_env'
           , 'marriage_per_person', 'cat_industry_1', 'industry_2_per_person', 'industry_3_per_person']

corr = Xtrain[varlist].corr()
fig=plt.figure(figsize=(12,8))
cax=plt.imshow(corr, vmin=-1, vmax=1, cmap=plt.cm.RdBu)
ax=plt.gca()
ax.set_xticks(range(len(corr)))
ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr,fontsize=10,rotation='vertical')
ax.set_yticklabels(corr,fontsize=10)
plt.colorbar(cax)

bigCorr = corr[corr > 0.5]
#%%
# Calculate VIF
from sklearn.linear_model import LinearRegression

varlist = ['edu_env', 'apt_midprice','real_estate_transaction_per_person','employment_per_person'
           , 'marriage_per_person', 'cat_industry_1', 'industry_2_per_person', 'industry_3_per_person']

reg=LinearRegression()
for i in varlist:
    y=Xtrain[i]
    X=Xtrain[np.setdiff1d(varlist,[i])]
    
    reg.fit(X,y)
    
    reg.coef_
    reg.intercept_
    
    reg.score(X,y)
    
    if(reg.score(X,y) == 1):
        print(i,"INF")
        continue
    print(i, 1/(1-reg.score(X,y)),'\n')
#%%
#Draw plot figures of each factor

varlist = ['edu_env', 'apt_midprice','real_estate_transaction_per_person','employment_per_person'
           , 'marriage_per_person', 'cat_industry_1', 'industry_2_per_person', 'industry_3_per_person']

for v in varlist :
   plt.figure()
   plt.xlabel(v, fontsize=16)
   Xtrain[v].plot()
#%%
   #Model Learning
   
#%%
# Birth_data for target(next) year
varlist1=['birth_data','apt_midprice', 'real_estate_transaction_per_person','employment_per_person',
         'cat_industry_1','marriage_per_person', 'industry_2_per_person', 'industry_3_per_person', 'edu_env']

# Moving average(2 years) for target(next) year
varlist2=['birth_data_mov_av','apt_midprice', 'real_estate_transaction_per_person','employment_per_person',
         'cat_industry_1','marriage_per_person', 'industry_2_per_person', 'industry_3_per_person', 'edu_env']

# Both birth_data and moving average
varlist3=['birth_data','birth_data_mov_av','apt_midprice', 'real_estate_transaction_per_person','employment_per_person',
          'cat_industry_1','marriage_per_person', 'industry_2_per_person', 'industry_3_per_person', 'edu_env']

#%%
# Select learning methods to predict birth_rate
# Using varlist1
from sklearn.linear_model import LinearRegression,RANSACRegressor

# 1)LinearRegression
X=Xtrain[varlist1]
y=target_data['birth_rate']

reg=LinearRegression()
reg.fit(X,y)
reg.score(X,y)

# 2)
# For outlying birth_rate
X=Xtrain[varlist1]

reg = RANSACRegressor(random_state=4).fit(X, target_data['birth_rate'])
reg.score(X, target_data['birth_rate'])
#%%
# Using varlist2
from sklearn.linear_model import RANSACRegressor

# 1)LinearRegression
X=Xtrain[varlist2]
y=target_data['birth_rate']

reg=LinearRegression()
reg.fit(X,y)
reg.score(X,y)

# 2)
# For outlying birth_rate
X=Xtrain[varlist2]

reg = RANSACRegressor(random_state=4).fit(X, target_data['birth_rate'])
reg.score(X, target_data['birth_rate'])
#%%
# Using varlist3
from sklearn.linear_model import RANSACRegressor

# 1)LinearRegression
X=Xtrain[varlist3]
y=target_data['birth_rate']

reg=LinearRegression()
reg.fit(X,y)
reg.score(X,y)

# 2)
# For outlying birth_rate
X=Xtrain[varlist3]

reg = RANSACRegressor(random_state=4).fit(X, target_data['birth_rate'])
reg.score(X, target_data['birth_rate'])
#%%
    #Model Validation

#%%
# Since each year has 150 rows, 150*7 = 1050
# Divide train, test set
trainX = Xtrain[:1050]
testX = Xtrain[1050:]
trainY = target_data[:1050] 
testY = target_data[1050:]

#Extract birth rate
trainY=trainY['birth_rate']
testY=testY['birth_rate']

#%%
# Select validation method with varlist1
# Test1 for validation
# KFold cross validation
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

trainX1=trainX[varlist1]
testX1=testX[varlist1]

temp=[]
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(trainX1):
    train_x = trainX1.iloc[train_index]
    train_y = trainY.iloc[train_index]
    test_x = trainX1.iloc[test_index]
    test_y = trainY.iloc[test_index]
    
    reg = LinearRegression()
    reg.fit(train_x,train_y)
    predict_y = reg.predict(test_x)
    temp.append(r2_score(test_y,predict_y))
print("KFold ",np.mean(temp))

# Test2 for validation
# StratifiedKFold cross validation
trainX2=trainX[varlist1]
testX2=testX[varlist1]

temp=[]

skf=StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(trainX, trainX['year']):
    train_x = trainX2.iloc[train_index]
    train_y = trainY.iloc[train_index]
    test_x = trainX2.iloc[test_index]
    test_y = trainY.iloc[test_index]
    
    reg = LinearRegression()
    reg.fit(train_x,train_y)
    predict_y = reg.predict(test_x)
    temp.append(r2_score(test_y,predict_y))
print("StratifiedKFold ",np.mean(temp))

# Test3 for validation
# TimeSeriesSplit cross validation
trainX3=trainX[varlist1]
testX3=testX[varlist1]

temp=[]
tscv = TimeSeriesSplit(n_splits=6)
for train_index, test_index in tscv.split(trainX3):
   train_x = trainX3.iloc[train_index]
   train_y = trainY.iloc[train_index]
   test_x = trainX3.iloc[test_index]
   test_y = trainY.iloc[test_index]
   
   reg = LinearRegression()
   reg.fit(train_x,train_y)
   predict_y = reg.predict(test_x)
   temp.append(r2_score(test_y,predict_y))
print("TimeSeriesSplit ",np.mean(temp))
#%%
# Model varlist selection - using TimeSeriesSplit validation

print("LinearRegression")
'''
varlist1 case
'''
trainX1=trainX[varlist1]
testX1=testX[varlist1]

temp=[]
for train_index, test_index in tscv.split(trainX1):
   train_x = trainX1.iloc[train_index]
   train_y = trainY.iloc[train_index]
   test_x = trainX1.iloc[test_index]
   test_y = trainY.iloc[test_index]
   
   reg = LinearRegression()
   reg.fit(train_x,train_y)
   predict_y = reg.predict(test_x)
   temp.append(r2_score(test_y,predict_y))
   # print(r2_score(test_y,predict_y))
print("varlist1 ", np.mean(temp))

'''
varlist2 case
'''
trainX2=trainX[varlist2]
testX2=testX[varlist2]

temp=[]
for train_index, test_index in tscv.split(trainX2):
   train_x = trainX2.iloc[train_index]
   train_y = trainY.iloc[train_index]
   test_x = trainX2.iloc[test_index]
   test_y = trainY.iloc[test_index]
   
   reg = LinearRegression()
   reg.fit(train_x,train_y)
   predict_y = reg.predict(test_x)
   temp.append(r2_score(test_y,predict_y))
   # print(r2_score(test_y,predict_y))
print("varlist2 " ,np.mean(temp))

'''
varlist3 case
'''
trainX3=trainX[varlist3]
testX3=testX[varlist3]

temp=[]
for train_index, test_index in tscv.split(trainX3):
   train_x = trainX3.iloc[train_index]
   train_y = trainY.iloc[train_index]
   test_x = trainX3.iloc[test_index]
   test_y = trainY.iloc[test_index]
   
   reg = LinearRegression()
   reg.fit(train_x,train_y)
   predict_y = reg.predict(test_x)
   temp.append(r2_score(test_y,predict_y))
   # print(r2_score(test_y,predict_y))
print("varlist3 ", np.mean(temp))

#%%
# Model parameter selection - using StratifiedKFold validation
# Lasso, Ridge with valist1
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score

alpha=[0.01,0.05,0.1,0.3,0.5,0.7,1]

# Lasso regression
temp1 =[]
print("Lasso")
for k in alpha:
    for train_index, test_index in tscv.split(trainX1):
        train_x = trainX1.iloc[train_index]
        train_y = trainY.iloc[train_index]
        test_x = trainX1.iloc[test_index]
        test_y = trainY.iloc[test_index]
        
        lasso = Lasso(alpha = k)
        lasso.fit(train_x,train_y)
        predict_y = lasso.predict(test_x)
        temp1.append(r2_score(test_y,predict_y))    
        # print(r2_score(test_y,predict_y))
    print("alpha:",k,"Lasso mean:",np.mean(temp1))
   
# Ridge regression
print("Ridge")
temp2 =[]
for k in alpha:
    for train_index, test_index in tscv.split(trainX1):
        train_x = trainX1.iloc[train_index]
        train_y = trainY.iloc[train_index]
        test_x = trainX1.iloc[test_index]
        test_y = trainY.iloc[test_index]
        
        ridge = Ridge(alpha = k)
        ridge.fit(train_x,train_y)
        predict_y = ridge.predict(test_x)
        temp2.append(r2_score(test_y,predict_y))    
        # print(r2_score(test_y,predict_y))
    print("alpha:",k,"Ridge mean:", np.mean(temp2))
#%%
# Training Result
varlist1=['birth_data','apt_midprice','employment_per_person','cat_industry_1','real_estate_transaction_per_person',
         'marriage_per_person', 'industry_2_per_person', 'industry_3_per_person', 'edu_env']

train_x=trainX[varlist1]
train_y= target_data[:1050] ['birth_rate']
test_x=testX[varlist1]
test_y= target_data[1050:] ['birth_rate']

reg = LinearRegression()
reg.fit(train_x,train_y)
predict_y = reg.predict(test_x)
print("LinearRegression: ", (r2_score(test_y,predict_y)))

lasso = Lasso(alpha = 0.01)
lasso.fit(train_x,train_y)
predict_y = lasso.predict(test_x)
print("Lasso(alpha = 0.01): ", (r2_score(test_y,predict_y)))

ridge = Ridge(alpha = 1)
ridge.fit(train_x,train_y)
predict_y = ridge.predict(test_x)
print("Ridge(alpha = 1): ", (r2_score(test_y,predict_y)))

#Final result
# OLS
import statsmodels.api as sm

X=train_x
X=sm.add_constant(X)

model=sm.OLS(train_y, X)
result=model.fit()
result.summary()

