#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 09:34:47 2018

@author: christinakronser
"""

import numpy as np
#import scipy.stats as stats
#import matplotlib.pyplot as plt
import sklearn
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn import linear_model


def linear_regression_funding_speed():
    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()
    
    # y-variable in regression is funding speed ("DAYS_NEEDED")    
    cur.execute("SELECT DAYS_NEEDED FROM success")
    y = cur.fetchall()
    y = np.array([i[0] for i in y])     # list of int
    print("y shape: ", y.shape)

    # x-variable in regression is the project description length ("WORD_COUNT")
    cur.execute("SELECT WORD_COUNT FROM success")
    x = cur.fetchall()
    x = np.array([i[0] for i in x])     # list of int
    print("x shape: ", x.shape)

    # Get the train and test data split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print("x_train shape: ", x_train.shape)
    print("x_test shape: ", x_test.shape)
    print("y_train shape: ", y_train.shape)
    print("y_test shape: ", y_test.shape)
    
    # Fit a model
    lm = linear_model.LinearRegression()
    x_train = x_train.reshape(-1, 1)
#    print("x_train new shape: ", x_train.shape)
    y_train = y_train.reshape(-1, 1)
#    print("y_train new shape: ", y_train.shape)
    model = lm.fit(x_train, y_train)
    x_test = x_test.reshape(-1, 1)
#    print("x_test new shape: ", x_test.shape)
    predictions_test = lm.predict(x_test)
    predictions_train = lm.predict(x_train)
#    print("model : ", model)
#    print("predictins shape: ", predictions_test.shape)
#    print("y_test[5]: ", y_test[5])
#    print("predictions[5]: ", predictions_test[5])
    
    # Calculate the root mean square error (RMSE) for test and training data
    N = len(y_test)
    rmse_test = np.sqrt(np.sum((np.array(y_test).flatten() - np.array(predictions_test).flatten())**2)/N)
    print("RMSE TEST: ", rmse_test)
    
    N = len(y_train)
    rmse_train = np.sqrt(np.sum((np.array(y_train).flatten() - np.array(predictions_train).flatten())**2)/N)
    print("RMSE train: ", rmse_train)
    
    weight = model.coef_
    print("model coef: ", weight)
    print("model intercept: ", model.intercept_)
    
        
    r2_score = sklearn.metrics.r2_score(y_test, predictions_test)
    print(r2_score)
    




def main():
    linear_regression_funding_speed()
    
    
    
    
if __name__ == "__main__": main()

    