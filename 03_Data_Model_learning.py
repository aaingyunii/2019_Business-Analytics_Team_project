# -*- coding: utf-8 -*-
"""

Created on Nov 17 2019

@author: in-gyunAhn

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
Xtrain=pd.read_csv("https://drive.google.com/uc?export=download&id=1lqSop-lRGpvDHp-vGZ5cjzBLd0AhwkVP",encoding="CP949")
target_data = pd.read_csv('https://drive.google.com/uc?export=download&id=1tuLsS2krwwcaPv6aicjMGmnqdMeRlAfw',encoding="CP949")

varlist=['employment','marriage','industry_1','industry_2','industry_3']
for i in varlist:
    a = Xtrain[i]/Xtrain['population']
    a = a.rename(i+'_per_person')
    Xtrain=pd.concat([Xtrain,a],axis=1)
    
# Categorical industry_1 is added
Xtrain['cat_industry_1']=Xtrain['industry_1']
Xtrain.cat_industry_1[Xtrain.cat_industry_1 != 0]=1

# Merge variales
real_estate_transaction = Xtrain['apt_transaction']+Xtrain['housing_transaction']
real_estate_transaction = real_estate_transaction.rename('real_estate_transaction')
edu_env = Xtrain['childcare']+Xtrain['education']
edu_env=edu_env.rename('edu_env')

Xtrain = pd.concat([Xtrain,real_estate_transaction],axis=1)
Xtrain = pd.concat([Xtrain,edu_env],axis=1)


#%%
# Calculate Correlation
Xtrain.columns
varlist = ['edu_env', 'apt_midprice','real_estate_transaction',
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
# Calculate VIF
from sklearn.linear_model import LinearRegression

varlist = ['edu_env', 'apt_midprice','real_estate_transaction',
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
Xtrain.columns
varlist=[ 'apt_midprice', 'birth_data','real_estate_transaction',
       'birth_data_mov_av', 'employment_per_person', 'marriage_per_person', 'industry_1_per_person', 'industry_2_per_person', 'industry_3_per_person',  'edu_env']
for v in varlist :
   plt.figure()
   plt.xlabel(v, fontsize=16)
   Xtrain[v].plot()

#%%
   ## without Moving average only birth rate
varlist1=['apt_midprice', 'real_estate_transaction', 'birth_data',
      'employment_per_person', 'marriage_per_person', 'industry_1_per_person', 'industry_2_per_person', 'industry_3_per_person', 'edu_env']

# Select learning methods to predict birth_rate
import statsmodels.api as sm
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import r2_score
# 1)
X=Xtrain[varlist1]
X=sm.add_constant(X)

model=sm.OLS(target_data['birth_rate'], X)
result=model.fit()
result.summary()

# 2)
# For outlying birth_rate
X=Xtrain[varlist1]

reg = RANSACRegressor(random_state=4).fit(X, target_data['birth_rate'])
reg.score(X, target_data['birth_rate'])

#%%
## with moving average
varlist2=['birth_data_mov_av','apt_midprice', 'real_estate_transaction','employment_per_person', 
         'marriage_per_person', 'industry_1_per_person', 'industry_2_per_person', 'industry_3_per_person', 'edu_env']

X=Xtrain[varlist3]
X=sm.add_constant(X)

# Linear regression
model=sm.OLS(target_data['birth_rate'], X)
result=model.fit()
result.summary()

# For outlying birth_rate

reg = RANSACRegressor(random_state=4).fit(X, target_data['birth_rate'])
reg.score(X, target_data['birth_rate'])
