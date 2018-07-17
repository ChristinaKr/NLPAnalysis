#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 16:49:29 2018

@author: christinakronser
"""
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


con = sqlite3.connect('databaseTest.db')
cur = con.cursor()

# y-variable in regression is funding gap  
cur.execute("SELECT GAP FROM dataset22")
y = cur.fetchall()
y = np.array([i[0] for i in y])     # list of int
print("y shape: ", y.shape)

# x1-variable in regression is the project description length ("WORD_COUNT")
cur.execute("SELECT NORM_WORDS FROM dataset22") # WORD_COUNT
x1 = cur.fetchall()
x1 = np.array([i[0] for i in x1])
x1 = x1.reshape(len(x1), 1)
print("x1 shape: ", x1.shape)
# x2-variable in regression is the description's sentiment score ("SENTIMENTSCORE")
cur.execute("SELECT NORM_SCORE FROM dataset22") # SENTIMENTSCORE
x2 = cur.fetchall()
x2 = np.array([i[0] for i in x2])
x2 = x2.reshape(len(x2), 1)
print("x2 shape: ", x2.shape)
# x3-variable in regression is the description's magnitude score ("MAGNITUDE")
cur.execute("SELECT NORM_MAGNITUDE FROM dataset22") # MAGNITUDE
x3 = cur.fetchall()
x3 = np.array([i[0] for i in x3])
x3 = x3.reshape(len(x3), 1)
print("x3 shape: ", x3.shape)

X = np.concatenate((x1,x2,x3), axis = 1)
print("X shape: ", X.shape)

# Using Skicit-learn to split data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);
R2 = rf.score(train_features, train_labels)
print("R^2: ", R2)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
##print("predictions[:10]: ", predictions[:10])
##print("test_targets[:10]: ", test_labels[:10])
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), '$ of funding gap')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_list = ["length", "sentiment score", "sentiment magnitude"]
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];