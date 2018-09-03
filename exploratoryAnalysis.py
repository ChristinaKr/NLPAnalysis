#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 11:36:57 2018

@author: christinakronser
"""
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from scipy.stats.stats import kendalltau
from scipy import stats

def select(cur, variable, table):
    """
    Database function to retrieve a variable
    """
    cur.execute("SELECT {v} FROM {t}".format(v = variable, t = table))
    variable = cur.fetchall()
    variable = [i[0] for i in variable]
    return variable


def distribution_funding_speed_histogram(cur, variable, table):
    """
    Distribution of funding speed
    """
        
    days = select(cur,variable, table)
    
    print("Number of entries: ", len(days))
    print("Maximum days: ", max(days))
    print("Minimum days: ", min(days))
    
    plt.xlabel("Funding speed in days")
    plt.ylabel("Number of loans")
    plt.hist(days, bins=80, range =(0,80))

    plt.show()

    
def distribution_funding_gap_histogram(cur, variable, table):
    """
    Distribution of funding gap
    """

    gap = select(cur,variable, table)
    
    print("Number of entries: ", len(gap))
    print("Maximum gap: ", max(gap))
    print("Minimum gap: ", min(gap))
    
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Funding gap in $")
    ax.set_ylabel("Number of loans")
    fig.suptitle('Histogram of Funding Gap')
    ax.hist(gap, bins = 65, range = (0, 8000))
    plt.show()



def distribution_loan_amount_successfulloans_histogram(cur, variable, table):
    """
    Distribution of loan amount for successful loans
    """
    loan_amount = select(cur,variable, table)

    print("Number of entries from success table: ", len(loan_amount))
    print("Maximum loan amount: ", max(loan_amount))
    print("Minimum loan amount: ", min(loan_amount))
    
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Loan Amount")
    ax.set_ylabel("Number of loans")
    fig.suptitle('Histogram of Loan Amount for successful loans')
    ax.hist(loan_amount, bins = 80, range = (0, 5000))
    plt.show()


def distribution_loan_amount_unsuccessfulloans_histogram(cur,variable, table):
    """
    Distribution of loan amount for unsuccessful loans
    """
    
    loan_amount = select(cur,variable, table)

    print("Number of entries from nosuccess table: ", len(loan_amount))
    print("Maximum loan amount: ", max(loan_amount))
    print("Minimum loan amount: ", min(loan_amount))
    
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Loan Amount")
    ax.set_ylabel("Number of loans")
    fig.suptitle('Histogram of Loan Amount for unsuccessful loans')
    ax.hist(loan_amount, bins = 50, range = (0, 10000))
    plt.show()



def distribution_length_histogram(cur,variable, table):
    """
    Distribution of description length
    """
    length = select(cur,variable, table)

    print("Number of entries: ", len(length))
    print("Maximum loan amount: ", max(length))
    print("Minimum loan amount: ", min(length))
    
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Length")
    ax.set_ylabel("Number of loans")
    fig.suptitle('Distribution of description length for successful loans')
    ax.hist(length, bins = 100)
    plt.show()



#--------------- Scatter Plots ---------------
    
def scatter_loan_amount_vs_days(cur,variable1, variable2, table):
    """
    Scatter plot showing correlation of 2 variables
    """
    
    days = select(cur,variable1, table)
    loan_amount = select(cur,variable2, table)

    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data, label the axes and set limits
    ax.set_xlabel("Funding speed in days")
    ax.set_ylabel("Loan Amount")
    fig.suptitle('Scatter plot of funding speed and loan amount')
    x_val = np.array(days)
    y_val = np.array(loan_amount)
    ax.set_ylim([0,50000])
    ax.set_xlim([0,400])
    ax.plot(x_val, y_val, 'x')
    
    # draw least squares line
    ax.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 1.0)
    plt.show()



def scatter_loan_amount_vs_gap(cur,variable1, variable2, table):
    """
    Scatter plot showing correlation of 2 variables
    """    
    gap = select(cur,variable1, table)
    loan_amount = select(cur,variable2, table)

    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data, label the axes and set limits
    ax.set_xlabel("Gap")
    ax.set_ylabel("Loan Amount")
    fig.suptitle('Scatter plot of funding gap and loan amount')
    x_val = np.array(gap)
    ax.set_ylim([0,50000])
    ax.set_xlim([0,10000])
    y_val = np.array(loan_amount)
    ax.plot(x_val, y_val, 'x')
    
    # draw least squares line
    ax.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 1.0)
    plt.show()
#    fig.savefig("scatter_loanamount_gap.pdf", fmt="pdf")


#--------------- Correlations (and Scatter Plots) ---------------

def correlations_speed(cur,variable1, variable2, table):
    """
    Correlation of 2 variables (including scatter plot)
    """
    x = select(cur,variable1, table)
    y = select(cur,variable2, table)
          
    # Scatterplot
#    mpl.style.use('ggplot')
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel("Gap")
    ax.set_ylabel("Sentiment magnitude")
    fig.suptitle('Correlation funding gap and sentiment magnitude')
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
    
def normality_tests(cur,variable1, table):
    """
    Normality test of 2 variables
    """
    print("\nNormality Tests:\n")
    length = select(cur,variable1, table)
            
    # D’Agostino’s K^2 Test
    stat, p = stats.normaltest(length)
    alpha = 0.05
    print('Statistics=%.3f, p=%.10f' % (stat, p))
    if p < alpha:  # null hypothesis: x comes from a normal distribution
        print("The null hypothesis that the var. is normally distributed can be rejected")
    else:
        print("The null hypothesis cannot be rejected; data is normally distributed")

    # Shapiro-Wilk Test
    statc, pv = stats.shapiro(length)
    print('Statistics=%.3f, p=%.10f' % (statc, pv))
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')
        
        
    
    
#--------------- LIWC ---------------

    
def line_plot_distribution(cur):
    """
    Plot a distribution of all used LIWC categories
    """
    # Retrieve all variabbles from DB
    table = "data22"
    
    humans = np.array(select(cur,"HUMANS_COUNT", table))
    family = np.array(select(cur,"FAMILY_COUNT", table))
    x4= (humans + family)
    x5 = np.array(select(cur,"HEALTH_COUNT", table))
    work = np.array(select(cur,"WORK_COUNT", table))
    achieve = np.array(select(cur,"ACHIEVE_COUNT", table))
    x6 = (work + achieve)    
    num = np.array(select(cur,"NUM_COUNT", table))
    quant = np.array(select(cur,"QUANT_COUNT", table))
    x7 = num + quant        
    x8 = np.array(select(cur,"PRONOUNS_COUNT", table))   
    x9 = np.array(select(cur,"INSIGHTS_COUNT", table))    
    
    y,binEdges=np.histogram(x4,bins=10)
    bincenters_family = 0.5*(binEdges[1:]+binEdges[:-1])
    y,binEdges=np.histogram(x5,bins=10)
    bincenters_humans = 0.5*(binEdges[1:]+binEdges[:-1])
    y,binEdges=np.histogram(x6,bins=10)
    bincenters_health = 0.5*(binEdges[1:]+binEdges[:-1])
    y,binEdges=np.histogram(x7,bins=10)
    bincenters_work = 0.5*(binEdges[1:]+binEdges[:-1])
    y,binEdges=np.histogram(x8,bins=10)
    bincenters_num = 0.5*(binEdges[1:]+binEdges[:-1])
    y,binEdges=np.histogram(x9,bins=10)
    bincenters_insights = 0.5*(binEdges[1:]+binEdges[:-1])
    
    plt.plot(bincenters_family,y,'-', label = "Family")
    plt.plot(bincenters_humans,y,'-', label = "Health")
    plt.plot(bincenters_health,y,'-', label = "Work")
    plt.plot(bincenters_work,y,'-', label = "Numbers")
    plt.plot(bincenters_num,y,'-', label = "Pronouns")
    plt.plot(bincenters_insights,y,'-', label = "Insights")
    plt.ylabel("Number of loans")
    plt.xlabel("Count")
    plt.xlim((0, 30))
    plt.legend(loc='best')
    plt.show()
        
def scatter_LIWC(cur, var1, var2, table):
    """
    Plot a scatter plot of the LIWC counts to funding gap or speed
    to discover relationship
    """
    family = select(cur,"FAMILY_COUNT", table)
    gap = select(cur,"GAP", table)

    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Funding gap in $")
    ax.set_ylabel("Family Count")
    fig.suptitle('Scatter plot of funding gap and family count')
    x_val = np.array(gap)
    ax.set_ylim([0,10])
#    ax.set_xlim([0,400])
    y_val = np.array(family)
    ax.plot(x_val, y_val, 'x')
    ax.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 1.0)
    plt.show()
    
def LIWC_pie_chart(cur):
    """
    Plot a pie chart of the chosen LIWC categories average counts
    """
    
    table = "data22"
    
    humans = np.array(select(cur,"HUMANS_COUNT", table))
    family = np.array(select(cur,"FAMILY_COUNT", table))
    x4= (humans + family)
    x5 = np.array(select(cur,"HEALTH_COUNT", table))
    work = np.array(select(cur,"WORK_COUNT", table))
    achieve = np.array(select(cur,"ACHIEVE_COUNT", table))
    x6 = (work + achieve)    
    num = np.array(select(cur,"NUM_COUNT", table))
    quant = np.array(select(cur,"QUANT_COUNT", table))
    x7 = num + quant        
    x8 = np.array(select(cur,"PRONOUNS_COUNT", table))   
    x9 = np.array(select(cur,"INSIGHTS_COUNT", table))    
    
    labels = 'Family', 'Health', 'Work', 'Numbers', 'Pronouns', 'Insights'
    sizes = [np.average(x4), np.average(x5), np.average(x6), np.average(x7), np.average(x8), np.average(x9)]
    colors = ['gold', 'plum', 'lightcoral', 'silver', 'lightskyblue', 'yellowgreen',]
     
    # Plot
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{v:d}'.format(v=val)
        return my_autopct

    plt.pie(sizes, labels=labels, colors=colors, autopct=make_autopct(sizes), shadow=True, startangle=140)
    plt.axis('equal')
    plt.show()
    
#def importances_histo():
#    
#    N = 10
#    ind = np.arange(N)  # the x locations for the groups
#    width = 0.2      # the width of the bars
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    
#    ### SUCCESS: loan, length, pronouns, work, family, numbers, magnitude, healh, insights, sentiment
##    smallvals = []
##    largevals = []
#
#    ### NO SUCCESS: loan, work, numbers, pronouns, length, family, magnitude, health, insights, score
#    smallvals = [0.37, 0.09, 0.09, 0.09, 0.08, 0.08, 0.07, 0.05, 0.04, 0.03]
#    largevals = [0.67, 0.06, 0.04, 0.04, 0.03, 0.04, 0.06, 0.02, 0.02, 0.02]
#    rects1 = ax.bar(ind, smallvals, width, color='r', label ='Small loans')
#    rects2 = ax.bar(ind+width, largevals, width, color='g', label = 'Large loans')
#    
#    ax.set_ylabel('Importance')
#    ax.set_xticks(ind+width/2)
#    ###SUCCESS:
##    ax.set_xticklabels( ('Amount', 'Length', 'Pronouns', 'Work', 'Family', 'Numbers', 'Magnitude', 'Health', 'Insights', 'Sentiment'), rotation='vertical' )
#    ### NO SUCCESS:
#    ax.set_xticklabels( ('Amount', 'Work', 'Numbers', 'Pronouns', 'Length', 'Family', 'Magnitude', 'Health', 'Insights', 'Sentiment'), rotation='vertical' )
#    ax.legend( (rects1[0], rects2[0]), ('Small loans', 'Large loans') )
#        
#    fig.tight_layout()
#    plt.show()
    
      
    
        
    
def main():
    # Make a connection to the database
    con = sqlite3.connect('database.db')
    cur = con.cursor()

    # Plot distribution of funding speed in days
    distribution_funding_speed_histogram(cur,"DAYS_NEEDED", "data11")
    
    # Plot distribution of funding gap in $
    distribution_funding_gap_histogram(cur,"GAP", "data22")
    
    # Plot distribution of loan amount for successful and unsuccessful loans
    distribution_loan_amount_successfulloans_histogram(cur, "LOAN_AMOUNT", "SUCCESS")
    distribution_loan_amount_unsuccessfulloans_histogram(cur,"LOAN_AMOUNT", "NOSUCCESS")
    
    # Plot distribution of length
    distribution_length_histogram(cur,"WORD_COUNT", "SUCCESS")

    # Scatter plot of loan amount and funding speed
    scatter_loan_amount_vs_days(cur,"DAYS_NEEDED", "LOAN_AMOUNT", "data12")
    # Scatter plot of loan amount and funding gap
    scatter_loan_amount_vs_gap(cur,"GAP", "LOAN_AMOUNT", "data22")
    
     # Correlations
    correlations_speed(cur,"GAP", "MAGNITUDE", "data22")
    
     # 2 Normality tests of a variable (assumption of linear regression)
    normality_tests(cur,"WORD_COUNT", "data21")
        
    # LIWC distributions
    line_plot_distribution(cur)
    scatter_LIWC(cur, "FAMILY_COUNT", "GAP", "data22")
    LIWC_pie_chart(cur)

    
if __name__ == "__main__": main()

