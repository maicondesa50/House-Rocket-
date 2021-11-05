#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[29]:


import pandas as pd
import numpy as np
from sklearn import linear_model as lm


# # Loading Data

# In[6]:


df_raw = pd.read_csv( 'dataset/kc_house_data.csv' )


# In[26]:


df_raw.head()


# # Data Preparation

# In[10]:


#Features
x_train = df_raw.drop( ['price', 'date'], axis = 1)
#Response variable
y_train = df_raw['price'].copy()


# # Model Training 

# In[17]:


#model description
model_lr = lm.LinearRegression()

#model training
model_lr.fit(x_train, y_train)


# In[18]:


#Prediction
pred = model_lr.predict(x_train)


# In[19]:


pred[0: 100]


# # Performance Metrics

# In[22]:


df1 = df_raw.copy()


# In[23]:


df1['prediction'] = pred


# In[43]:


df1[['id', 'price', 'prediction', 'error', 'error_absolut', 'error_perc', 'error_perc_abs']].head()


# In[42]:


df1['error'] = df1['price'] - df1['prediction']
df1['error_absolut'] = np.abs( df1['error'] )

df1['error_perc'] = ( (df1['price'] - df1['prediction'] ) / df1['price'])
df1['error_perc_abs'] = np.abs(df1['error_perc'])


# In[33]:


np.sum( df1['error_absolut'] ) / len (df1 ['error_absolut'])


# In[46]:


#mean absolute error
mae = np.mean(df1['error_absolut'])
print('MAE: {}'.format(mae))


# In[48]:


#Mean absolut percentage error
mape = np.mean(df1['error_perc_abs'])
print('MAPE: {}'.format(mape))

