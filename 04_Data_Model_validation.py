# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 11:21:59 2019

@author: in-gyunAhn
"""
import pandas as pd
import numpy as np
#%%
# Import original collected data  
Xtrain=pd.read_csv("https://drive.google.com/uc?export=download&id=1lqSop-lRGpvDHp-vGZ5cjzBLd0AhwkVP",encoding="CP949")
target_data = pd.read_csv('https://drive.google.com/uc?export=download&id=1tuLsS2krwwcaPv6aicjMGmnqdMeRlAfw',encoding="CP949")

# Merge variales
real_estate_transaction = Xtrain['apt_transaction']+Xtrain['housing_transaction']
real_estate_transaction = real_estate_transaction.rename('real_estate_transaction')
edu_env = Xtrain['childcare']+Xtrain['education']
edu_env = edu_env.rename('edu_env')

Xtrain = pd.concat([Xtrain,real_estate_transaction],axis=1)
Xtrain = pd.concat([Xtrain,edu_env],axis=1)

varlist=['employment','marriage','industry_1','industry_2','edu_env','industry_3','real_estate_transaction']

for i in varlist:
    a = Xtrain[i]/Xtrain['population']
    a = a.rename(i+'_per_person')
    Xtrain = pd.concat([Xtrain,a],axis=1)

#%%
# Categorical industry_1 is added
Xtrain['cat_industry_1']=Xtrain['industry_1']
Xtrain.cat_industry_1[Xtrain.cat_industry_1 != 0]=1

#%%
# Since each year has 150 rows, 150*7 = 1050
# Divide train, test set
trainX = Xtrain[:1050]
testX = Xtrain[1050:]
trainY = target_data[:1050] 
testY = target_data[1050:]

'''
Xtrain_bf = pd.DataFrame()
Xtrain2017 = pd.DataFrame()

for i in range (len(Xtrain)):
    if Xtrain.get_value(i,'year') != 2017:
        Xtrain_bf = Xtrain_bf.append(Xtrain.iloc[i])
    else:
        Xtrain2017 = Xtrain2017.append(Xtrain.iloc[i])
'''
#%%
# Birth_data for target(next) year
varlist1=['birth_data','apt_midprice', 'real_estate_transaction','employment_per_person','cat_industry_1',
         'marriage_per_person', 'industry_1_per_person', 'industry_2_per_person', 'industry_3_per_person', 'edu_env']
# Moving average(2 years) for target(next) year
varlist2=['birth_data_mov_av','apt_midprice', 'real_estate_transaction','employment_per_person','cat_industry_1',
         'marriage_per_person', 'industry_1_per_person', 'industry_2_per_person', 'industry_3_per_person', 'edu_env']

# Both birth_data and moving average
varlist3=['birth_data','birth_data_mov_av','apt_midprice', 'real_estate_transaction','employment_per_person','cat_industry_1',
         'marriage_per_person', 'industry_1_per_person', 'industry_2_per_person', 'industry_3_per_person', 'edu_env']

'''
mapping = {'서울': 1, '부산': 2,'대구':3,'인천':4,'광주':5,'대전':6,'울산':7,
           '경기':8,'강원':9 ,'충북':10 ,'충남':11 ,'전북':12 , 
           '전남':13 ,'경북':14 ,'경남':15 ,'제주':16}
    
trainX = trainX.replace({'지역':mapping})
'''

trainY=trainY['birth_rate']
testY=testY['birth_rate']
#%%
# Select validation method with varlist3
# Test1 for validation
# KFold cross validation
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

trainX3=trainX[varlist3]
testX3=testX[varlist3]

temp=[]
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(trainX3):
    train_x = trainX3.iloc[train_index]
    train_y = trainY.iloc[train_index]
    test_x = trainX3.iloc[test_index]
    test_y = trainY.iloc[test_index]
    
    reg = LinearRegression()
    reg.fit(train_x,train_y)
    predict_y = reg.predict(test_x)
    temp.append(r2_score(test_y,predict_y))
    # print(r2_score(test_y,predict_y))
print("KFold ",np.mean(temp))

# Test2 for validation
# StratifiedKFold cross validation
trainX3=trainX[varlist3]
testX3=testX[varlist3]

temp=[]

skf=StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(trainX, trainX['year']):
    train_x = trainX3.iloc[train_index]
    train_y = trainY.iloc[train_index]
    test_x = trainX3.iloc[test_index]
    test_y = trainY.iloc[test_index]
    
    reg = LinearRegression()
    reg.fit(train_x,train_y)
    predict_y = reg.predict(test_x)
    temp.append(r2_score(test_y,predict_y))
    # print(r2_score(test_y,predict_y))
print("StratifiedKFold ",np.mean(temp))

# Test3 for validation
# TimeSeriesSplit cross validation
trainX3=trainX[varlist3]
testX3=testX[varlist3]

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
   # print(r2_score(test_y,predict_y))
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
         'marriage_per_person', 'industry_2_per_person', 'industry_3_per_person', 'edu_env_per_person']

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

# OLS
import statsmodels.api as sm

X=train_x
X=sm.add_constant(X)

model=sm.OLS(train_y, X)
result=model.fit()
result.summary()
