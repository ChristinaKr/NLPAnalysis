#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 09:03:47 2018

@author: christinakronser

Database to be found: https://drive.google.com/file/d/1KHmasvJFN4AWuflgicGeqvInMmNkKkio/view?usp=sharing
"""
import numpy as np
import sqlite3
import statsmodels.api as sm
import pandas as pd


def multicollinearity(cur, table):
    """
    Test for multicollinearity between variables in regression
    """
    # Retrieve variables from DB
    x = np.array(select(cur,"WORD_COUNT", table))
    x1 = x.reshape(len(x), 1)
    x2 = np.array(select(cur,"SENTIMENTSCORE", table))
    x2 = x2.reshape(len(x2), 1)
    x3 = np.array(select(cur,"MAGNITUDE", table))
    x3 = x3.reshape(len(x3), 1)
    humans = np.array(select(cur,"HUMANS_COUNT", table))
    family = np.array(select(cur,"FAMILY_COUNT", table))
    x4 = (humans + family)/x
    x4 = x4.reshape(len(x4), 1)
    x5 = np.array(select(cur,"HEALTH_COUNT", table))
    x5 = x5/x
    x5 = x5.reshape(len(x5), 1)
    work = np.array(select(cur,"WORK_COUNT", table))
    achieve = np.array(select(cur,"ACHIEVE_COUNT", table))
    x6 = (work + achieve)/x  
    x6 = x6.reshape(len(x6), 1)
    num = np.array(select(cur,"NUM_COUNT", table))
    quant = np.array(select(cur,"QUANT_COUNT", table))
    x7 = (num + quant)/x
    x7 = x7.reshape(len(x7), 1)
    x8 = np.array(select(cur,"PRONOUNS_COUNT", table)) 
    x8 = x8/x
    x8 = x8.reshape(len(x8), 1)
    x9 = np.array(select(cur,"INSIGHTS_COUNT", table))   
    x9 = x9/x
    x9 = x9.reshape(len(x9), 1)
    x10 = np.array(select(cur,"LOAN_AMOUNT", "data22"))
    x10 = x10/x
    x10 = x10.reshape(len(x10), 1)

    
    X = np.concatenate((x1,x2,x3, x4, x5, x6, x7, x8, x9, x10), axis = 1)
    corr = np.corrcoef(X, rowvar=0)
    w, v = np.linalg.eig(corr)
    print("Test for multicollinearity:")
    print(w)
    print(v[:,2])
    print(v[:,0])
    
    
def lin_reg(cur, table):
    """
    Linear Regression
    """
           
    # Retrieve NORMALISED variables from DB
    
    # y-variable in regression is funding gap  
    y = np.array(select(cur,"GAP", table))
    
    # x-variables    
    x1 = np.array(select(cur,"NORM_WORDS", table))
    x1 = x1.reshape(len(x1), 1)
    
    x2 = np.array(select(cur,"NORM_SCORE", table))
    x2 = x2.reshape(len(x2), 1)
    
    x3 = np.array(select(cur,"NORM_MAGNITUDE", table))
    x3 = x3.reshape(len(x3), 1)
    
    x = np.array(select(cur,"WORD_COUNT", table))
    humans = np.array(select(cur,"HUMANS_COUNT", table))
    family = np.array(select(cur,"FAMILY_COUNT", table))
    x4= (humans + family)/x
    x4 = (x4-np.average(x4))/np.std(x4)
    x4 = x4.reshape(len(x4), 1)
    
    x5 = np.array(select(cur,"HEALTH_COUNT", table))
    x5 = x5/x
    x5 = (x5-np.average(x5))/np.std(x5)
    x5 = x5.reshape(len(x5), 1)
    
    work = np.array(select(cur,"WORK_COUNT", table))
    achieve = np.array(select(cur,"ACHIEVE_COUNT", table))
    x6 = (work + achieve)/x
    x6 = (x6-np.average(x6))/np.std(x6)
    x6 = x6.reshape(len(x6), 1)
    
    num = np.array(select(cur,"NUM_COUNT", table))
    quant = np.array(select(cur,"QUANT_COUNT", table))
    x7 = (num + quant)/x
    x7 = (x7-np.average(x7))/np.std(x7)
    x7 = x7.reshape(len(x7), 1)
    
    x8 = np.array(select(cur,"PRONOUNS_COUNT", table)) 
    x8 = x8/x
    x8 = (x8-np.average(x8))/np.std(x8)
    x8 = x8.reshape(len(x8), 1)
    
    x9 = np.array(select(cur,"INSIGHTS_COUNT", table))   
    x9 = x9/x
    x9 = (x9-np.average(x9))/np.std(x9)
    x9 = x9.reshape(len(x9), 1)
    
    x10 = np.array(select(cur,"LOAN_AMOUNT", "data22"))
    x10 = x10/x
    x10 = (x10-np.average(x10))/np.std(x10)
    x10 = x10.reshape(len(x10), 1)

    X = np.concatenate((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10), axis = 1)
    
    X = pd.DataFrame({'Length':X[:,0], 'Sentiment':X[:,1], 'Magnitude':X[:,2], 'Family':X[:,3], 'Health':X[:,4], 'Work':X[:,5], 'Number':X[:,6], 'Pronouns':X[:,7], 'Insight':X[:,8], 'Loan amount':X[:,9]})
    
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X)
    results = model.fit()
    print(results.summary())    

def select(cur, variable, table):
    """
    Database function to retrieve a variable
    """
    cur.execute("SELECT {v} FROM {t}".format(v = variable, t = table))
    variable = cur.fetchall()
    variable = [i[0] for i in variable]
    return variable


def main():
    # Make a connection to the database
    con = sqlite3.connect('database.db')
    cur = con.cursor()
    
    # Testing our independent variables for multicollinearity
#    multicollinearity(cur, "data22")    
    
    # Linear Regression
    lin_reg(cur, "data22")

    
if __name__ == "__main__": main()



