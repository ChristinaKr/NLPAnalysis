#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 11:36:57 2018

@author: christinakronser
"""
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import collections
import matplotlib as mpl
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from scipy.stats.stats import kendalltau
from scipy import stats
from scipy.stats import shapiro


### 1a
def distribution_funding_speed_histogram(days):
    """
    Distribution of funding speed
    """
    
#    con = sqlite3.connect('databaseTest.db')
#    cur = con.cursor()
#
##    cur.execute("SELECT DAYS_NEEDED FROM funded WHERE BORROWER_GENDERS NOT LIKE '%female%' AND LOAN_AMOUNT > 4000 AND SECTOR_NAME = 'Agriculture' AND EUROPE = 0")
#    cur.execute("SELECT DAYS_NEEDED FROM success")
#    days = cur.fetchall()
    print("Number of entries: ", len(days))
    print("Maximum days: ", max(days))
    print("Minimum days: ", min(days))
    
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Funding speed in days")
    ax.set_ylabel("Number of loans")
    fig.suptitle('Histogram of Funding Speed')
    ax.hist(days, bins = 100, range = (0, 80))
    plt.show()


#def line_plot_gap():
#    fig = plt.figure()
#    ax = plt.axes()
#    x = np.linspace(0, 10, 1000)
#    ax.plot(, np.sin(x));

### 1b
    
def distribution_funding_gap_histogram(gap):
    """
    Distribution of funding gap
    """
    
#    con = sqlite3.connect('databaseTest.db')
#    cur = con.cursor()
#        
##    cur.execute("SELECT GAP FROM nosuccess WHERE LOAN_AMOUNT > 4000 AND SECTOR_NAME = 'Agriculture'")
#    cur.execute("SELECT GAP FROM nosuccess")
#    gap = cur.fetchall()
    print("Number of entries: ", len(gap))
    print("Maximum gap: ", max(gap))
    print("Minimum gap: ", min(gap))
    
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Funding gap in US$")
    ax.set_ylabel("Number of loans")
    fig.suptitle('Histogram of Funding Gap')
    gap = [i[0] for i in gap]
#    ax.hist(gap)
    ax.hist(gap, bins = 100, range = (0, 2000))
    plt.show()



### 2a
    
def distribution_loan_amount_days_histogram():
    """
    Distribution of loan amount
    """
    
    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()

#    cur.execute("SELECT DAYS_NEEDED FROM funded WHERE BORROWER_GENDERS NOT LIKE '%female%' AND LOAN_AMOUNT > 4000 AND SECTOR_NAME = 'Agriculture' AND EUROPE = 0")
    cur.execute("SELECT LOAN_AMOUNT FROM success")
    loan_amounts = cur.fetchall()
    print("Number of entries from success table: ", len(loan_amounts))
    print("Maximum loan amount: ", max(loan_amounts))
    print("Minimum loan amount: ", min(loan_amounts))
    
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Loan Amount")
    ax.set_ylabel("Number of loans")
    fig.suptitle('Histogram of Loan Amounts FUNDED')
    loan_amount = [i[0] for i in loan_amounts]
    ax.hist(loan_amount, bins = 100, range = (0, 5000))
    plt.show()

### 2b

def distribution_loan_amount_gap_histogram():
    """
    Distribution of loan amount
    """
    
    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()

#    cur.execute("SELECT DAYS_NEEDED FROM funded WHERE BORROWER_GENDERS NOT LIKE '%female%' AND LOAN_AMOUNT > 4000 AND SECTOR_NAME = 'Agriculture' AND EUROPE = 0")
    cur.execute("SELECT LOAN_AMOUNT FROM nosuccess")
    loan_amounts = cur.fetchall()
    print("Number of entries from nosuccess table: ", len(loan_amounts))
    print("Maximum loan amount: ", max(loan_amounts))
    print("Minimum loan amount: ", min(loan_amounts))
    
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Loan Amount")
    ax.set_ylabel("Number of loans")
    fig.suptitle('Histogram of Loan Amounts NOT FUNDED')
    loan_amount = [i[0] for i in loan_amounts]
    ax.hist(loan_amount, bins = 100, range = (0, 5000))
    plt.show()


### 2c (?)

def distribution_loan_amount_plot():
    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()
    cur.execute("SELECT LOAN_AMOUNT FROM success")
    loan_amount = cur.fetchall()
    loan_amount = [i[0] for i in loan_amount]
    loan_amount = np.array(loan_amount,dtype='int')
    
    x ,y  = np.unique(loan_amount, return_counts=True) # counting occurrence of each loan
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel("Loan amounts")
    ax.set_ylabel("Number of loans")
    fig.suptitle('Distribution of loan amounts')
    ax.set_ylim([0,50000])
    ax.set_xlim([0,1000])

#    plt.scatter(x,y)
    ax.plot(x, y)
    plt.show()    



### 3a
    
def scatter_loan_amount_vs_days(days, loan_amount_days):
    days = [i[0] for i in days]
    loan_amount_days = [i[0] for i in loan_amount_days]

    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Funding speed in days")
    ax.set_ylabel("Loan Amount")
    fig.suptitle('Scatter plot of funding speed and loan amount')
    x_val = np.array(days)
    ax.set_ylim([0,50000])
    ax.set_xlim([0,400])
    y_val = np.array(loan_amount_days)
    ax.plot(x_val, y_val, 'x')
    ax.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 1.0)
    
    plt.show()


#### 3b

def scatter_loan_amount_vs_gap(gap, loan_amount_gap):
    gap = [i[0] for i in gap]
    loan_amount_gap = [i[0] for i in loan_amount_gap]

    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Gap")
    ax.set_ylabel("Loan Amount")
    fig.suptitle('Scatter plot of funding gap and loan amount')
    x_val = np.array(gap)
    ax.set_ylim([0,50000])
    ax.set_xlim([0,10000])
    y_val = np.array(loan_amount_gap)
    ax.plot(x_val, y_val, 'x')
    ax.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 1.0)
    
    plt.show()

def plot_loan_amount_vs_gap(gap, loan_amount_gap):
    gap = [i[0] for i in gap]
    loan_amount_gap = [i[0] for i in loan_amount_gap]

    # create an empty figure object
    fig = plt.figure()
    # create an axis on the figure
    ax = fig.add_subplot(1,1,1)

    # plot these
    ax.plot(gap, loan_amount_gap, '-', linewidth=1)
    ax.set_xlabel("Gap")
    ax.set_ylabel("Loan Amount")
#    fig.savefig("old_faithful_evolution.pdf", fmt="pdf")

def check():

    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()
    
    cur.execute("SELECT LOAN_AMOUNT FROM success WHERE DAYS_NEEDED > 250")
    print("Loan amounts where days needed > 250: ", cur.fetchall())
    cur.execute("SELECT COUNT(*) FROM success WHERE DAYS_NEEDED > 250")
    print("Xount Days needed > 250: ", cur.fetchall())

def correlations_speed(days, length):
    x = days
    y = length
    
    
    # Scatterplot
    mpl.style.use('ggplot')
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel("Days raised")
    ax.set_ylabel("Length")
    fig.suptitle('Correlation funding speed and description length')
    plt.scatter(x, y)
    plt.show()

    # Pearson correlation and p-value
    p_corr_speed_length = pearsonr(x, y)
    print("Pearson: ", p_corr_speed_length)
    
    # Spearman correaltion and p-value    
    s_corr_speed_length = spearmanr(x,y)
    print("Spearman: ", s_corr_speed_length)
    
    # Kendall correlation and p-value
    k_corr_speed_length = kendalltau(x,y)
    print("Kendall: ", k_corr_speed_length)
    
def normality_tests(days, length):
    # D’Agostino’s K^2 Test
    k2, p = stats.normaltest(np.log(days))
    alpha = 0.05
    print("p = %.3f"%(p))
    if p < alpha:  # null hypothesis: x comes from a normal distribution
        print("The null hypothesis that the var. is normally distributed can be rejected")
    else:
        print("The null hypothesis cannot be rejected; data is normally distributed")

    # Shapiro-Wilk Test
    stat, p = shapiro(np.log(days))
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

def main():
    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()
        
#    cur.execute("SELECT GAP FROM nosuccess WHERE LOAN_AMOUNT < 2000 AND SECTOR_NAME = 'Agriculture'")
#    gap = cur.fetchall()
    
#    cur.execute("SELECT LOAN_AMOUNT FROM nosuccess")
#    loan_amount_gap = cur.fetchall()
#    cur.execute("SELECT LOAN_AMOUNT FROM success")
#    loan_amount_days = cur.fetchall()

#    cur.execute("SELECT DAYS_NEEDED FROM success")
    
    
    
    cur.execute("SELECT DAYS_NEEDED FROM success WHERE BORROWER_GENDERS NOT LIKE '%female%' AND LOAN_AMOUNT < 1000 AND SECTOR_NAME = 'Agriculture' AND EUROPE = 0")
    days = cur.fetchall()
    days = [i[0] for i in days]

    cur.execute("SELECT WORD_COUNT FROM success WHERE BORROWER_GENDERS NOT LIKE '%female%' AND LOAN_AMOUNT < 1000 AND SECTOR_NAME = 'Agriculture' AND EUROPE = 0")
    word_count = cur.fetchall()
    length = [i[0] for i in word_count]

#    correlations_speed(days, length)
#    
#    print("Distribution Funding Speed: ")
#    distribution_funding_speed_histogram(days)
    
    
    
#    print("Distribution Funding Speed LOG: ")
#    distribution_funding_speed_histogram(np.log(days))
#    scatter_loan_amount_vs_days(days, loan_amount_days)
    
    
#    print("Distribution Funding Gap: ")
#    distribution_funding_gap_histogram(gap)
#    print("Distribution Funding Gap LOG: ")
#    distribution_funding_gap_histogram(np.log(gap))
#    scatter_loan_amount_vs_gap(gap, loan_amount_gap)
#    plot_loan_amount_vs_gap(gap, loan_amount_gap)
    
#    distribution_loan_amount_plot()
#    distribution_loan_amount_days_histogram()
#    distribution_loan_amount_gap_histogram()
    
    normality_tests(days, length)
    
    
if __name__ == "__main__": main()

