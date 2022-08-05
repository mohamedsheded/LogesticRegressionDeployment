#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Major Libraries for utils
import pandas as pd
import os
## sklearn -- for pipeline and preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector
import numpy as np


# In[2]:


## Read the CSV file using pandas







## Split the Dataset -- Taking only train to fit (the same the model was trained on)
X =  pd.read_csv('X.csv')  ## Features
y =  pd.read_csv('y.csv')   ## target

## the same Random_state (take care)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,train_size=0.8)




# In[7]:


## numerical pipeline
num_pipeline = Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                        ])



X_train = num_pipeline.fit_transform(X_train) ## fit 


# we must use .fit_transform to make the model train on the original data set

# In[9]:


def preprocess_new(X_new):
    '''this function is used for the preprocessing for 
    a new data before prediction
    arguments
    *********
    (X_new: 2D array) --> The Features in the same order
                ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 
                 'population', 'households', 'median_income', 'ocean_proximity']
        All Featutes are Numerical, except the last one is Categorical.
    returns
    *******
    Preprocessed Features ready to make inference by the Model
    '''


    return num_pipeline.transform(X_new)
    


# In[ ]:




