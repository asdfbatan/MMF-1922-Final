#!/usr/bin/env python
# coding: utf-8
# Reference: https://www.kaggle.com/code/fedi1996/house-prices-data-cleaning-viz-and-modeling/notebook
# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import sklearn.metrics as metrics
import math
import re


# In[2]:


# Importing **train** and **test** datasets
sample_submission = pd.read_csv("sample_submission.csv")
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")
# Creating a copy of the train and test datasets
c_test = test.copy()
c_train = train.copy()


# In[3]:


c_train.head()


# In[4]:


c_test.head()


# In[5]:


c_train['train']  = 1
c_test['train']  = 0
df = pd.concat([c_train, c_test], axis=0,sort=False)


# In[6]:


#Percentage of NAN Values 
NAN = [(c, df[c].isna().mean()*100) for c in df]
NAN = pd.DataFrame(NAN, columns=["column_name", "percentage"])


# In[7]:


NAN = NAN[NAN.percentage > 50]
NAN.sort_values("percentage", ascending=False)


# In[8]:


# Now we will select numerical and categorical features
object_columns_df = df.select_dtypes(include=['object'])
numerical_columns_df =df.select_dtypes(exclude=['object'])


# In[9]:


object_columns_df.dtypes


# In[10]:


numerical_columns_df.dtypes


# In[11]:


df['cut']=df['cut'].map({'Ideal':5,'Premium':4, 'Very Good':3,'Good':2,'Fair':1})
df['color']=df['color'].map({'G':4,'E':6,'F':5,'H':3,'D':7,'I':2,'J' : 1})
df['clarity']=df['clarity'].map({'SI1' : 3, 'VS2' : 4, 'SI2': 2,'VS1' : 5,'VVS2' : 6, 'VVS1': 7, 'IF' : 8,'I1' : 1})


# In[12]:


#Select categorical features
rest_object_columns = df.select_dtypes(include=['object'])
#Using One hot encoder
df = pd.get_dummies(df, columns=rest_object_columns.columns)  


# In[13]:


df.head()


# In[14]:


print(df)


# In[15]:


df_final = df


# In[21]:


#df_final = df_final.drop(['id',],axis=1)

df_train = df_final[df_final['train'] == 1]
df_train = df_train.drop(['train',],axis=1)


df_test = df_final[df_final['train'] == 0]
df_test = df_test.drop(['price'],axis=1)
df_test = df_test.drop(['train',],axis=1)


# In[24]:


target= df_train['price']
df_train = df_train.drop(['price'],axis=1)


# In[25]:


x_train,x_test,y_train,y_test = train_test_split(df_train,target,test_size=0.33,random_state=0)


# In[26]:


xgb =XGBRegressor( booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.6, gamma=0,
             importance_type='gain', learning_rate=0.01, max_delta_step=0,
             max_depth=4, min_child_weight=1.5, n_estimators=2400,
             n_jobs=1, nthread=None, objective='reg:linear',
             reg_alpha=0.6, reg_lambda=0.6, scale_pos_weight=1, 
             silent=None, subsample=0.8, verbosity=1)


lgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       learning_rate=0.01, 
                                       n_estimators=12000, 
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.4, 
                                       )


# In[27]:


#Fitting
xgb.fit(x_train, y_train)
lgbm.fit(x_train, y_train,eval_metric='rmse')


# In[28]:


predict1 = xgb.predict(x_test)
predict = lgbm.predict(x_test)


# In[29]:


print('Root Mean Square Error test = ' + str(math.sqrt(metrics.mean_squared_error(y_test, predict1))))
print('Root Mean Square Error test = ' + str(math.sqrt(metrics.mean_squared_error(y_test, predict))))


# In[30]:


xgb.fit(df_train, target)
lgbm.fit(df_train, target,eval_metric='rmse')


# In[31]:


predict4 = lgbm.predict(df_test)
predict3 = xgb.predict(df_test)
predict_y = ( predict3*0.45 + predict4 * 0.55)


# In[32]:


submission = pd.DataFrame({
        "id": test["id"],
        "price": predict_y
    })
submission.to_csv('submission.csv', index=False)


# In[ ]:




