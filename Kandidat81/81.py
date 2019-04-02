#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: 81
"""

#Loads libraries
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# Loads dataset
dataset = pd.read_csv('Flaveria.csv')

# Makes new columns with float instead of string objects. Basicly converts the string to float so it possible to run the algorithem. 
# So you can both se the string  and the float. 
dataset['N_level_float'] = dataset['N level'].map({'L':0.0, 'M':1.0, 'H':2})
dataset['species_float'] = dataset['species'].map({'brownii':0.0, 'pringlei':1.0, 'trinervia':2.0, 'ramosissima':3.0, 'robusta': 4.0, 'bidentis':5.0})

# Sets  X = N_level_float','species_float and y = 'Plant Weight(g)'
X = dataset[['N_level_float','species_float']].values 
y = dataset[['Plant Weight(g)']].values 

# Splits the dataset into traing and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=88) 

# Uses the KNeighborsRegressor 
my_Regressor = KNeighborsRegressor(n_neighbors=3).fit(X_train, y_train)

#Prints score 
print ("Score : {:.3f}".format(my_Regressor.score(X_test, y_test)))





