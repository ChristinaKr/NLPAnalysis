#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 11:36:57 2018

@author: christinakronser
"""
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
#import collections
import matplotlib as mpl
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from scipy.stats.stats import kendalltau
from scipy import stats


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
    ax.set_xlabel("Log of Funding speed in days")
    ax.set_ylabel("Number of loans")
    fig.suptitle('Histogram of Funding Speed')
    ax.hist(days, bins = 100, range = (0, 10))
    plt.show()

def distribution_funding_speed_4_histograms(days):
    """
    Distribution of funding speed
    """
    
#    print("Number of entries: ", len(days))
#    print("Maximum days: ", max(days))
#    print("Minimum days: ", min(days))
    
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
#    ax = fig.add_subplot(1,1,1)
    ax_11 = fig.add_subplot(1.5,2,1)
    ax_12 = fig.add_subplot(1.5,2,2)
    # histogram the data and label the axes
    ax_11.set_xlabel("Funding speed in days")
    ax_11.set_ylabel("Number of loans")
    ax_12.set_xlabel("LOG funding speed")
    fig.suptitle('Distribution of Funding Speed')
    ax_11.hist(days, bins = 100, range = (0,80))
    ax_12.hist(np.log(days), bins = 50, range = (0, 10))
#    plt.tight_layout()
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
    ax.set_xlabel("Log of funding gap in US$")
    ax.set_ylabel("Number of loans")
    fig.suptitle('Histogram of LOG Funding Gap')
#    ax.hist(gap)
    ax.hist(gap, bins = 100, range = (0, 10))
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
    ax.hist(loan_amount, bins = 50, range = (0, 5000))
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

def distribution_length_days_histogram(length):
    """
    Distribution of description length
    """
    
    print("Number of entries from success table: ", len(length))
    print("Maximum loan amount: ", max(length))
    print("Minimum loan amount: ", min(length))
    
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("LOG Length")
    ax.set_ylabel("Number of loans")
    fig.suptitle('Distribution of LOG description length')
    ax.hist(length, bins = 100, range = (0, 10))
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

def correlations_speed(x, y):
    #x = days
    #y = length
    
    
    # Scatterplot
    mpl.style.use('ggplot')
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel("LOG speed")
    ax.set_ylabel("LOG length")
    fig.suptitle('Correlation log funding speed and log description length')
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
    
def normality_tests(length):
            
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

def histogram_quartiles():    
    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()
        
    cur.execute("SELECT SENTENCESCORES  FROM data22")
    sentence_scores = cur.fetchall()
    sentence_scores = [i[0] for i in sentence_scores]   # multiple list of strings
    cur.execute("SELECT SENTENCEMAGNITUDES  FROM data22")
    sentence_mags = cur.fetchall()
    sentence_mags = [i[0] for i in sentence_mags]   # multiple list of strings
    
    quartileBottom_score = []
    quartileBottom_mag = []
    halfMiddle_score = []
    halfMiddle_mag = []
    quartileTop_score = []
    quartileTop_mag = []

        
    for i in range(len(sentence_scores)):
        print("i: ", i)
        sentence_score = eval(sentence_scores[i])   # simple list of floats
        sentence_mag = eval(sentence_mags[i])
        for i in range(len(sentence_score)):
            if i < round((0.25*len(sentence_score))):
                quartileBottom_score.append(sentence_score[i])
                quartileBottom_mag.append(sentence_mag[i])
            if i > round((0.75*len(sentence_score))):
                quartileTop_score.append(sentence_score[i])
                quartileTop_mag.append(sentence_mag[i])
            else:
                halfMiddle_score.append(sentence_score[i])
                halfMiddle_mag.append(sentence_mag[i])
        
    n_groups = 3
#    absBottom_score = [abs(number) for number in quartileBottom_score]
#    absMiddle_score = [abs(number) for number in halfMiddle_score]
#    absTop_score = [abs(number) for number in quartileTop_score]
    means_score = (np.average(quartileBottom_score), np.average(halfMiddle_score), np.average(quartileTop_score))
#    means_score = (np.average(absBottom_score), np.average(absMiddle_score), np.average(absTop_score))
    std_score = (np.std(quartileBottom_score), np.std(halfMiddle_score), np.std(quartileTop_score))

    means_mag = (np.average(quartileBottom_mag), np.average(quartileTop_mag), np.average(quartileTop_mag))
    std_mag = (np.std(quartileBottom_mag), np.std(quartileTop_mag), np.std(quartileTop_mag))
    fig, ax = plt.subplots()
    
    print("Means Sentiment Score: ", means_score)
    print("Means Magnitude: ", means_mag)
    
    index = np.arange(n_groups)
    bar_width = 0.35
    
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    
    rects1 = ax.bar(index, means_score, bar_width,
                    alpha=opacity, color='b',
                    yerr=std_score, error_kw=error_config,
                    label='Sentiment')
        
    rects2 = ax.bar(index + bar_width, means_mag, bar_width,
                    alpha=opacity, color='r',
                    yerr=std_mag, error_kw=error_config,
                    label='Magnitude')
    
#    ax.set_xlabel('Quartiles')
    ax.set_ylabel('Scores')
    ax.set_title('Scores by sentiment and magnitude')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(('Bottom quartile', 'Middle half', 'Top quartile'))    
    ax.legend((rects1[0], rects2[0]), ('Sentiment', 'Magnitude'))
        
    
    fig.tight_layout()
    plt.show()    
    
def distribution_sentences_histo():     
    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()
    cur.execute("SELECT SENTENCESCORES  FROM data22")
    sentence_scores = cur.fetchall()
    sentence_scores = [i[0] for i in sentence_scores]   # multiple list of strings
    cur.execute("SELECT SENTENCEMAGNITUDES  FROM data22")
    sentence_mags = cur.fetchall()
    sentence_mags = [i[0] for i in sentence_mags]   # multiple list of strings
    
    quartileBottom_score = []
    quartileBottom_mag = []
#    halfMiddle_score = []
#    halfMiddle_mag = []
    quartileTop_score = []
    quartileTop_mag = []
    
        
    for i in range(len(sentence_scores)):
        print("i: ", i)
        sentence_score = eval(sentence_scores[i])   # simple list of floats
        sentence_mag = eval(sentence_mags[i])
        for i in range(len(sentence_score)):
            if i < round((0.25*len(sentence_score))):
                quartileBottom_score.append(sentence_score[i])
                quartileBottom_mag.append(sentence_mag[i])
            if i > round((0.75*len(sentence_score))):
#            if i == len(sentence_score)-1:
                quartileTop_score.append(sentence_score[i])
                quartileTop_mag.append(sentence_mag[i])
#            else:
#                halfMiddle_score.append(sentence_score[i])
#                halfMiddle_mag.append(sentence_mag[i])
    
    
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Number of loans")
    fig.suptitle('Distribution of bottom quartile sentences sentiment')
    ax.hist(quartileBottom_score)
    plt.show()
    
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Number of loans")
    fig.suptitle('Distribution of top quartile sentences sentiment')
    ax.hist(quartileTop_score)
    plt.show()    
        
    
    
def main():
#    con = sqlite3.connect('databaseTest.db')
#    cur = con.cursor()
        
#    cur.execute("SELECT GAP FROM nosuccess WHERE LOAN_AMOUNT < 2000 AND SECTOR_NAME = 'Agriculture'")
#    gap = cur.fetchall()
#    gap = [i[0] for i in gap]
#    gap_transformed, _ = stats.boxcox(gap)

    
#    cur.execute("SELECT LOAN_AMOUNT FROM nosuccess")
#    loan_amount_gap = cur.fetchall()
#    cur.execute("SELECT LOAN_AMOUNT FROM success")
#    loan_amount_days = cur.fetchall()
    
    
    
#    cur.execute("SELECT DAYS_NEEDED FROM success WHERE BORROWER_GENDERS NOT LIKE '%female%' AND LOAN_AMOUNT < 1000 AND SECTOR_NAME = 'Agriculture' AND EUROPE = 0")
#    days = cur.fetchall()
#    days = [i[0] for i in days]
    
#    for i in range(len(days)):
#        if days[i] == 0:
#            days[i] = 0.00000000000000000000000000000000000001    


#    cur.execute("SELECT WORD_COUNT FROM success WHERE BORROWER_GENDERS NOT LIKE '%female%' AND LOAN_AMOUNT < 1000 AND SECTOR_NAME = 'Agriculture' AND EUROPE = 0")
#    word_count = cur.fetchall()
#    length_speed = [i[0] for i in word_count]
##    length_transformed, _ = stats.boxcox(length)  
#    cur.execute("SELECT WORD_COUNT FROM nosuccess WHERE LOAN_AMOUNT < 2000 AND SECTOR_NAME = 'Agriculture'")
#    word_count = cur.fetchall()
#    length_gap = [i[0] for i in word_count]

    
#    correlations_speed(np.log(days), np.log(length_speed))
    
#    print("Distribution Length: ")
#    distribution_length_days_histogram(length_speed)
#    print("Distribution Length LOG: ")
#    distribution_length_days_histogram(np.log(length_speed))

    
#    print("Distribution Funding Speed LOG: ")
#    distribution_funding_speed_histogram(np.log(days))
#    distribution_funding_speed_4_histograms(days)
#    print("Distribution Funding Speed BOXCOX: ")
#    days_transformed, _ = stats.boxcox(days)
#    distribution_funding_speed_histogram(days_transformed)

#    scatter_loan_amount_vs_days(days, loan_amount_days)
#    scatter_loan_amount_vs_days(days, loan_amount_days)

    
    
#    print("Distribution Funding Gap: ")
#    distribution_funding_gap_histogram(gap)
#    correlations_speed(gap, length_gap)
#
#    print("Distribution Funding Gap LOG: ")
#    distribution_funding_gap_histogram(np.log(gap))
#    correlations_speed(np.log(gap), np.log(length_gap))

#    print("Distribution Funding Gap Boxcox: ")
#    distribution_funding_gap_histogram(gap_transformed)
#    correlations_speed(gap_transformed, length)

#    scatter_loan_amount_vs_gap(gap, loan_amount_gap)
#    plot_loan_amount_vs_gap(gap, loan_amount_gap)
    
#    distribution_loan_amount_plot()
#    distribution_loan_amount_days_histogram()
#    distribution_loan_amount_gap_histogram()
#    normality_tests(np.log(length_speed))
    
#    histogram_quartiles()
#    plt.show()
    distribution_sentences_histo()
    
if __name__ == "__main__": main()

