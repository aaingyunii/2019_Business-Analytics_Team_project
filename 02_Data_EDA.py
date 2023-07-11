# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:36:55 2019

@author: in-gyunAhn

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
Xtrain=pd.read_csv("https://drive.google.com/uc?export=download&id=1jRpf7GPw1BQ9m5oygfTpzcUEs6yQVWaJ",encoding="CP949")
#2010년
target_data = pd.read_csv('https://drive.google.com/uc?export=download&id=1FkHqlZwG36Sc5Y5i6RO1iyx_ce7KmIK7',encoding="CP949") 

varlist=['employment','marriage','industry_1','industry_2','industry_3']

for i in varlist:
    a = Xtrain[i]/Xtrain['population']
    a = a.rename(i+'_per_person')
    Xtrain=pd.concat([Xtrain,a],axis=1)

#%%
real_estate_transaction = Xtrain['apt_transaction']+Xtrain['housing_transaction']
real_estate_transaction = real_estate_transaction.rename('real_estate_transaction')
edu_env = Xtrain['childcare']+Xtrain['education']
edu_env=edu_env.rename('edu_env')

Xtrain = pd.concat([Xtrain,real_estate_transaction],axis=1)
Xtrain = pd.concat([Xtrain,edu_env],axis=1)
#%% 
# Calculate Correlation
Xtrain.columns
varlist = ['edu_env', 'apt_midprice','real_estate_transaction','apt_transaction','housing_transaction',
           'employment_per_person', 'marriage_per_person', 'industry_1_per_person', 'industry_2_per_person', 'industry_3_per_person']

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
'''
2. Make dummy 필요 없음
from sklearn.linear_model import LinearRegression

reg=LinearRegression()
dummy=pd.get_dummies(Xtrain['시도별'],prefix='city',drop_first=True)
Xtrain = pd.concat((Xtrain,dummy),axis=1)
Xtrain=Xtrain.drop(['시도별'],axis=1)
print(Xtrain.columns)
'''
#%%
#  Calculate VIF
from sklearn.linear_model import LinearRegression

varlist = ['edu_env', 'apt_midprice','real_estate_transaction','apt_transaction','housing_transaction',
           'employment_per_person', 'marriage_per_person', 'industry_1_per_person', 'industry_2_per_person', 'industry_3_per_person']

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
    print(i, 1/(1-reg.score(X,y)))
#%%
    

