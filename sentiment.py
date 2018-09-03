#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27

@author: christinakronser
"""

import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import google.cloud.language
import os
from google.api_core.exceptions import InvalidArgument
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/christinakronser/Downloads/My First Project-522341c822f9.json"

def sentiment_analysis():
    con = sqlite3.connect('database.db')
    cur = con.cursor()
    
    # Extract description and translated description from DB
    cur.execute("SELECT DESCRIPTION FROM nosuccess WHERE LOAN_AMOUNT < 1000")
    descriptions = cur.fetchall()
    descriptions = [i[0] for i in descriptions]
    description = np.array(descriptions)
    
    cur.execute("SELECT DESCRIPTION_TRANSLATED FROM nosuccess WHERE LOAN_AMOUNT < 1000' ")
    description_trans = cur.fetchall()
    description_trans = [i[0] for i in description_trans]
    description_trans = np.array(description_trans)
    
    description_list = []
    sentimentscore_list=[]
    magnitude_list=[]
    sentences_score_list=[]
    sentences_magnitude_list=[]
    sum = 0
    
    # Create a language client
    language_client = google.cloud.language.LanguageServiceClient()
    
    # Use translated description if available; if not use original, English description
    for i in range(len(description)):
        if description_trans[i] == '':
            descr = description[i]
        else:
            descr = description_trans[i]
        
        document = google.cloud.language.types.Document(
            content=descr,
            type=google.cloud.language.enums.Document.Type.PLAIN_TEXT)
        # Use Language to detect the sentiment of the text
        try:
            response = language_client.analyze_sentiment(document=document)
        except InvalidArgument as e:
            print("Invalid: ", i)
            sum += 1
            continue
    
        # Store scores on sentence level
        score_all=[]
        magnitude_all=[]
        for y in range(len(response.sentences)):
            score_all.append((response.sentences[y].sentiment.score))
            magnitude_all.append((response.sentences[y].sentiment.magnitude))
        
        sentences_score_list.append(repr(score_all))
        sentences_magnitude_list.append(repr(magnitude_all))
        
        
        description_list.append(descr)
        sentiment = response.document_sentiment
        sentimentscore_list.append(sentiment.score)
        magnitude_list.append(sentiment.magnitude)
        print ('Progress: {}/{} rows processed'.format(i, len(description)))
    
    # Check how many rows were invalid
    print("Sum of skipped rows: ", sum)
    
    # Add sentiment score and magnitude for both levels to the database
    cur.execute("DROP TABLE IF EXISTS temp")
    cur.execute("CREATE TABLE temp(DESCRIPTIONS text, SENTIMENTSCORE numeric, MAGNITUDE numeric, SENTENCESCORES text, SENTENCEMAGNITUDES text)")
    
    def insert(d, ss, m, sens, senm):
        cur.execute("INSERT INTO temp (DESCRIPTIONS, SENTIMENTSCORE, MAGNITUDE, SENTENCESCORES, SENTENCEMAGNITUDES) VALUES (?, ?, ?, ?, ?)", (d, ss, m, sens, senm))
    
    for d, ss, m, sens, senm in zip(description_list, sentimentscore_list, magnitude_list, sentences_score_list, sentences_magnitude_list):
        insert(d, ss, m, sens, senm)
    
    cur.execute("DROP TABLE IF EXISTS data22")
    cur.execute("CREATE TABLE data22 AS SELECT success.*, temp.SENTIMENTSCORE, temp.MAGNITUDE, temp.SENTENCESCORES, temp.SENTENCEMAGNITUDES FROM success, temp WHERE temp.DESCRIPTIONS IN (success.DESCRIPTION, success.DESCRIPTION_TRANSLATED)")
    con.commit()


def distribution_sentimentscore_histogram(x, label):
    print("Number of entries: ", len(x))
    print("Minimum Sentiment: ", min(x))
    print("Maximum Sentiment: ", max(x))
    
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # set labels and plot histogram
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Number of Loans")
    fig.suptitle(label)
    ax.hist(x, bins = 10)
    plt.show()

def distribution_magnitude_histogram(x, label):
    print("Number of entries: ", len(x))
    print("Minimum: ", min(x))
    print("Maximum: ", max(x))
    
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # set labels and plot histogram
    ax.set_xlabel("Sentiment Magnitude")
    ax.set_ylabel("Number of Sentences")
    fig.suptitle(label)
    ax.hist(x, bins = 20)
    plt.show()

def scatter_linearity(x, x_label, y, y_label, name):
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # set labels and plot histogram
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim([0,11000])
    fig.suptitle(name)    
    ax.plot(x, y, 'x')
#    ax.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), color = 'r', linewidth = 1.0)
    plt.show()

def distribution_sentences_sentimentscore_histogram(x, label):
    
    # sentences scores are stored as string list
    sentences_sentiment = []
    for i in range(len(x)):
        sen = eval(x[i])   # simple list of floats
        sentences_sentiment.append(sen)
    sentence_sentiment = [i[0] for i in sentences_sentiment]
    
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # set labels and plot histogram
    ax.set_xlabel("Sentences Sentiment Score")
    ax.set_ylabel("Number of Loans")
    fig.suptitle(label)
    ax.hist(sentence_sentiment, bins = 10)
    plt.show()


def histogram_quartiles():    
    con = sqlite3.connect('database.db')
    cur = con.cursor()
    
    # Extract sentence scores and magnitudes from database
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
    means_score = (np.average(quartileBottom_score), np.average(halfMiddle_score), np.average(quartileTop_score))

    means_mag = (np.average(quartileBottom_mag), np.average(quartileTop_mag), np.average(quartileTop_mag))
    fig, ax = plt.subplots()
    
    print("Means Sentiment Score: ", means_score)
    print("Means Magnitude: ", means_mag)
    
    index = np.arange(n_groups)
    bar_width = 0.35
    
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    
    rects1 = ax.bar(index, means_score, bar_width,
                    alpha=opacity, color='b', error_kw=error_config,
                    label='Sentiment')
        
    rects2 = ax.bar(index + bar_width, means_mag, bar_width,
                    alpha=opacity, color='r',
                    error_kw=error_config,
                    label='Magnitude')
    
    ax.set_ylabel('Scores')
    ax.set_title('Sentiment and magnitude by quartile')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(('Bottom quartile (Beginning)', 'Middle half', 'Top quartile (End)'))    
    ax.legend((rects1[0], rects2[0]), ('Sentiment', 'Magnitude'))
    
    fig.tight_layout()
    plt.show()    
    
    
 
def random_forest():
    con = sqlite3.connect('database.db')
    cur = con.cursor()
    
    # y-variable in forest is funding gap  
    cur.execute("SELECT GAP FROM data22")
    y = cur.fetchall()
    y = np.array([i[0] for i in y])
    
    # x1-variable in forest is the project description length
    cur.execute("SELECT WORD_COUNT FROM data22") 
    x1 = cur.fetchall()
    x = np.array([i[0] for i in x1])
    description_length = x.reshape(len(x), 1)
    # x2-variable in forest is the description's sentiment score 
    cur.execute("SELECT SENTIMENTSCORE FROM data22")
    x2 = cur.fetchall()
    x2 = np.array([i[0] for i in x2])
    sentiment_score = x2.reshape(len(x2), 1)
    # x3-variable in forest is the description's magnitude score
    cur.execute("SELECT MAGNITUDE FROM data22") 
    x3 = cur.fetchall()
    x3 = np.array([i[0] for i in x3])
    magnitude = x3.reshape(len(x3), 1)
    # x4-variable in forest is the humans count
    cur.execute("SELECT HUMANS_COUNT FROM data22")
    humans = cur.fetchall()
    humans = np.array([i[0] for i in humans])
    cur.execute("SELECT FAMILY_COUNT FROM data22") 
    family = cur.fetchall()
    family = np.array([i[0] for i in family])
    x4= (humans + family)/x
    family_count = x4.reshape(len(x4), 1)
    # x5-variable in forest is the health count
    cur.execute("SELECT HEALTH_COUNT FROM data22") 
    x5 = cur.fetchall()
    x5 = np.array([i[0] for i in x5])
    x5 = x5/x
    health_count = x5.reshape(len(x5), 1)
    # x6-variable in forest is the work count
    cur.execute("SELECT WORK_COUNT FROM data22") 
    work = cur.fetchall()
    work = np.array([i[0] for i in work])
    cur.execute("SELECT ACHIEVE_COUNT FROM data22")
    achieve = cur.fetchall()
    achieve = np.array([i[0] for i in achieve])
    x6 = (work + achieve)/x
    work_count = x6.reshape(len(x6), 1)
    # x7-variable in forest is the numbers count
    cur.execute("SELECT NUM_COUNT FROM data22") 
    num = cur.fetchall()
    num = np.array([i[0] for i in num])
    cur.execute("SELECT QUANT_COUNT FROM data22")
    quant = cur.fetchall()
    quant = np.array([i[0] for i in quant])
    x7 = (num + quant)/x
    number_count = x7.reshape(len(x7), 1)
    # x8-variable in forest is the pronouns count
    cur.execute("SELECT PRONOUNS_COUNT FROM data22") 
    x8 = cur.fetchall()
    x8 = np.array([i[0] for i in x8])
    x8 = x8/x
    pronouns_count = x8.reshape(len(x8), 1)
    # x9-variable in forest is the insights count
    cur.execute("SELECT INSIGHTS_COUNT FROM data22") 
    x9 = cur.fetchall()
    x9 = np.array([i[0] for i in x9])
    x9 = x9/x
    insights_count = x9.reshape(len(x9), 1)
    # x10-variable in forest is the loan amount
    cur.execute("SELECT LOAN_AMOUNT FROM data22")
    x10 = cur.fetchall()
    x10 = np.array([i[0] for i in x10])
    loan_amount = x10.reshape(len(x10), 1)
    
    X = np.concatenate((description_length,sentiment_score, magnitude,family_count,health_count,work_count,number_count,pronouns_count,insights_count,loan_amount), axis = 1)
    
    # Using Skicit-learn to split data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.25, random_state = 42)
        
    # Instantiate model with 100 decision trees
    rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
    # Train the model on training data
    rf.fit(train_features, train_labels);
    R2 = rf.score(test_features, test_labels)
    print("R^2: ", R2)

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), '$')
        
    #Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    #Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%')
    
    # Get numerical feature importances
    importances = rf.feature_importances_
    # List of tuples with variable and importance
    feature_list = ["Description length", "Sentiment score", "Sentiment magnitude", "Family count", "Health count", "Work count", "Numbers count", "Pronouns count", "Insights count", "Loan amount"]
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances 
#    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
    return feature_importances

def importances_histo(feature_importances):
    
    N = 10
    ind = np.arange(N) 
    width = 0.2
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    smallvals = [0.37, 0.09, 0.09, 0.09, 0.08, 0.08, 0.07, 0.05, 0.04, 0.03]
    largevals = [feature_importances[0][1], feature_importances[1][1], feature_importances[2][1], feature_importances[3][1], feature_importances[4][1], feature_importances[5][1], feature_importances[6][1], feature_importances[7][1], feature_importances[8][1], feature_importances[9][1]]
    rects1 = ax.bar(ind, smallvals, width, color='r', label ='Small loans')
    rects2 = ax.bar(ind+width, largevals, width, color='g', label = 'Large loans')
    
    ax.set_ylabel('Importance')
    ax.set_xticks(ind+width/2)
    ax.set_xticklabels( (feature_importances[0][0], feature_importances[1][0], feature_importances[2][0], feature_importances[3][0], feature_importances[4][0], feature_importances[5][0], feature_importances[6][0], feature_importances[7][0], feature_importances[8][0], feature_importances[9][0]), rotation='vertical' )
    ax.legend( (rects1[0], rects2[0]), ('Small loans', 'Large loans') )
        
    fig.tight_layout()
    plt.show()


def main():
    con = sqlite3.connect('database.db')
    cur = con.cursor()
    
    # Sentiment Score
    cur.execute("SELECT SENTIMENTSCORE FROM data22")
    sentimentScore = cur.fetchall()
    sentimentScore = np.array([i[0] for i in sentimentScore])
    distribution_sentimentscore_histogram(sentimentScore, "Distribution Sentiment Score - Large Loans")
    cur.execute("SELECT GAP FROM data22")
    days = cur.fetchall()
    days = np.array([i[0] for i in days])
    scatter_linearity(sentimentScore, "Sentiment Score", days, "Funding gap", "Scatter plot of funding gap and sentiment score - Large loans")

    # Sentences Sentiment Score 
    cur.execute("SELECT SENTENCESCORES  FROM data22")
    sentence_scores = cur.fetchall()
    sentence_scores = [i[0] for i in sentence_scores]   # multiple list of strings
    distribution_sentences_sentimentscore_histogram(sentence_scores, "Distribution Sentiment Score on Sentence Level")
    
    # Histogram of sentences distribution on quartile level
    histogram_quartiles()

    # Sentiment Magnitude
    cur.execute("SELECT MAGNITUDE FROM data22")
    magnitude = cur.fetchall()
    magnitude = np.array([i[0] for i in magnitude])
    distribution_magnitude_histogram(magnitude, "Distribution Sentiment Magnitude - Large loans")
    cur.execute("SELECT GAP FROM data22")
    gap = cur.fetchall()
    gap = np.array([i[0] for i in gap])
    scatter_linearity(magnitude, "Sentiment magnitude", gap, "Funding gap", "Scatter plot of funding gap and sentiment magnitude - Large loans")
    
    
    # Random forest
    feature_importance = random_forest()
    importances_histo(feature_importance)

    
    
    
if __name__ == "__main__": main()





