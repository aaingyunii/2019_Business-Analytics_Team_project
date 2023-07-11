# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 15:16:54 2019

@author: in-gyunAhn
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import statsmodels.api as sm
from pandas.plotting import scatter_matrix

#%%


apt_midprice_data = pd.read_csv("https://drive.google.com/uc?export=download&id=12zdo-dURXHVEjHV0wt-vjMWXAQRXnCvv",encoding='utf-8')
apt_transaction_data = pd.read_csv("https://drive.google.com/uc?export=download&id=1HSQ29WIlsdA1qEwqXRvTvI6IPhpYcVgm",encoding='utf-8')
childcare_data = pd.read_csv("https://drive.google.com/uc?export=download&id=1XwSdPfIP2S9eruerZdQr7JsRuiIcvn_n",encoding="CP949")
housing_transaction_data = pd.read_csv("https://drive.google.com/uc?export=download&id=1YBh9_JDnMbtjmxYsM0VeleBCO92p-43U",encoding='utf-8')

population_data = pd.read_csv("https://drive.google.com/uc?export=download&id=1fgzsNNND84RqBFdnQRFr3-d3soWQ7sfw", encoding='CP949')
marriage_data = pd.read_csv("https://drive.google.com/uc?export=download&id=11OmQzRRRhkfcqlZV2HcxFZEJxoJ_Bp4z",encoding='CP949')
employment_data = pd.read_csv("https://drive.google.com/uc?export=download&id=1b5w7ErddZBR55uRpBHRLJvOvSriK9h9D",encoding='CP949')
education_data = pd.read_csv("https://drive.google.com/uc?export=download&id=1N9kfw-d2aUEOrQsF0lwoMkzZ-O8aro-F",encoding="CP949")
target_data = pd.read_csv("https://drive.google.com/uc?export=download&id=1ICwkBq_-unOEWNN3D6UamDG4CUfYNLo-",encoding="CP949")

nwi1=pd.read_csv("https://drive.google.com/uc?export=download&id=17F8LtBpr652PwiHcmSiccNfvX4Xqsv8N",encoding='CP949')
nwi2=pd.read_csv("https://drive.google.com/uc?export=download&id=1la0jWRTdm3B82-7ac7xOQI-wiinmhxUL",encoding='CP949')
nwi3=pd.read_csv("https://drive.google.com/uc?export=download&id=18AMeHh95DPex0Z96Knk7prGte2qHhBxH",encoding='CP949')




#%%

year = ['2010','2011','2012','2013','2014','2015','2016','2017']

#%%
childpd = pd.DataFrame(columns=['지역','구역','year','childcare'])
for i in year:
    temp = childcare_data[['지역','구역',i]]
    temp = temp.rename(columns={i:'childcare'})
    temp['year'] = i
    
    childpd = pd.concat([childpd,temp],axis=0)
#%%
edu = pd.DataFrame(columns=['지역','구역','year','education'])
for i in (year):
    temp = education_data[['지역','구역',i]]
    temp = temp.rename(columns={i:'education'})
    temp['year'] = i
    
    edu = pd.concat([edu,temp],axis=0)
    
#%%    
apt_mid =  pd.DataFrame(columns=['지역','year','apt_midprice'])
for i in (year):
    temp = apt_midprice_data[['지역',i]]
    temp = temp.rename(columns={i:'apt_midprice'})
    temp['year'] = i
    
    apt_mid = pd.concat([apt_mid,temp],axis=0)
  
#%%
apt_trans = pd.DataFrame(columns=['지역','구역','year','apt_transaction'])

for i in (year):
    temp = apt_transaction_data[['지역','구역',i]]
    temp = temp.rename(columns={i:'apt_transaction'})
    temp['year'] = i
    
    apt_trans = pd.concat([apt_trans,temp],axis=0)
    
#%%
 
employ =  pd.DataFrame(columns=['지역','year','employment'])

for i in year:
    temp = employment_data[['지역',i]]
    temp = temp.rename(columns={i:'employment'})
    temp['year'] = i
    
    employ = pd.concat([employ,temp],axis=0)
   
#%%
marrpd =  pd.DataFrame(columns=['지역','구역','year','marriage'])


for i in year:
    temp = marriage_data[['지역','구역',i]]
    temp = temp.rename(columns={i:'marriage'})
    temp['year'] = i
    
    marrpd = pd.concat([marrpd,temp],axis=0)

#%%
poppd = pd.DataFrame(columns=['지역','구역','year','population'])


for i in year:
    temp = population_data[['지역','구역',i]]
    temp = temp.rename(columns={i:'population'})
    temp['year'] = i
    
    poppd = pd.concat([poppd,temp],axis=0)
#%%  
housing = pd.DataFrame(columns=['지역','구역','year','housing_transaction'])

for i in year:
    temp = housing_transaction_data[['지역','구역',i]]
    temp = temp.rename(columns={i:'housing_transaction'})
    temp['year'] = i
    
    housing = pd.concat([housing,temp],axis=0)    
    
    
#%%
    
niw1pd =  pd.DataFrame(columns=['지역','구역','year','industry_1'])

for i in year:
    temp = nwi1[['지역','구역',i]]
    temp = temp.rename(columns={i:'industry_1'})
    temp['year'] = i
    
    niw1pd = pd.concat([niw1pd,temp],axis=0)
    
#%%
niw2pd =  pd.DataFrame(columns=['지역','구역','year','industry_2'])

for i in year:

    temp = nwi2[['지역','구역',i]]
    temp = temp.rename(columns={i:'industry_2'})
    temp['year'] = i
    
    niw2pd = pd.concat([niw2pd,temp],axis=0)

#%%
niw3pd =  pd.DataFrame(columns=['지역','구역','year','industry_3'])

for i in year:

    temp = nwi3[['지역','구역',i]]
    temp = temp.rename(columns={i:'industry_3'})
    temp['year'] = i
    
    niw3pd = pd.concat([niw3pd,temp],axis=0)
                    
#%%


trial = pd.DataFrame(columns = ['지역','구역','year','childcare'])

trial[['지역','구역']] = childpd[['지역','구역']]

trial = pd.merge(trial,childpd, how='right')
trial = pd.merge(trial,edu, on=['지역','구역','year'])
trial = pd.merge(trial,apt_mid, on=['지역','year'])


trial = pd.merge(trial,apt_trans, on=['지역','구역','year'])

trial = pd.merge(trial,employ, on=['지역','year'])
trial = pd.merge(trial,marrpd, on=['지역','구역','year'])
trial = pd.merge(trial,poppd, on=['지역','구역','year'])
trial = pd.merge(trial,housing, on=['지역','구역','year'])
trial = pd.merge(trial,niw1pd, on=['지역','구역','year'],how = 'outer' )
trial = pd.merge(trial,niw2pd, on=['지역','구역','year'],how = 'outer' )
trial = pd.merge(trial,niw3pd, on=['지역','구역','year'],how = 'outer' )


#%%
trial.to_csv("Xtrain.csv",encoding="CP949")

Xtrain=pd.read_csv("https://drive.google.com/uc?export=download&id=1jRpf7GPw1BQ9m5oygfTpzcUEs6yQVWaJ",encoding="CP949")
target_data = pd.read_csv("https://drive.google.com/uc?export=download&id=1ICwkBq_-unOEWNN3D6UamDG4CUfYNLo-",encoding="CP949")

varlist=['employment','marriage','industry_1','industry_2','industry_3']

for i in varlist:
    a = Xtrain[i]/Xtrain['population']
    a = a.rename(i+'_per_person')
    Xtrain=pd.concat([Xtrain,a],axis=1)

