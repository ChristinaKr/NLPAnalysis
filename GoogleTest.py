#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 09:52:02 2018

@author: christinakronser
"""

import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import google.cloud.language
import os
from google.api_core.exceptions import InvalidArgument
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/christinakronser/Downloads/My First Project-522341c822f9.json"

def sentiment_analysis():
    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()
    cur.execute("SELECT DESCRIPTION FROM success WHERE LOAN_AMOUNT < 1000 AND SECTOR_NAME = 'Agriculture' AND EUROPE = 0 AND BORROWER_GENDERS NOT LIKE '%female%' ")
    descriptions = cur.fetchall()
    descriptions = [i[0] for i in descriptions]
    description = np.array(descriptions)
    cur.execute("SELECT DESCRIPTION_TRANSLATED FROM success WHERE LOAN_AMOUNT < 1000 AND SECTOR_NAME = 'Agriculture' AND EUROPE = 0 AND BORROWER_GENDERS NOT LIKE '%female%' ")
    description_trans = cur.fetchall()
    description_trans = [i[0] for i in description_trans]
    description_trans = np.array(description_trans)
    
    
    description_list = []
    sentimentscore_list=[]
    magnitude_list=[]
    sentences_score_list=[]
    sentences_magnitude_list=[]
    sum= 0
    
    # Create a Language client.
    language_client = google.cloud.language.LanguageServiceClient()
    
    
    for i in range(len(description)):
        if description_trans[i] == '':
            descr = description[i]
        else:
            descr = description_trans[i]
        
        document = google.cloud.language.types.Document(
            content=descr,
            type=google.cloud.language.enums.Document.Type.PLAIN_TEXT)
        # Use Language to detect the sentiment of the text.
        try:
            response = language_client.analyze_sentiment(document=document)
        except InvalidArgument as e:
            print("Invalid: ", i)
            sum += 1
            continue
    
        ###NEW SAVE SENTENCE ATTRIBUTES###
        score_all=[]
        magnitude_all=[]
        for y in range(len(response.sentences)):
            score_all.append((response.sentences[y].sentiment.score))
            magnitude_all.append((response.sentences[y].sentiment.magnitude))
        
        sentences_score_list.append(repr(score_all))
        sentences_magnitude_list.append(repr(magnitude_all))
        # use eval() to turn it back into a list of floats
        ###-----------END-----------###
    
        description_list.append(descr)
        sentiment = response.document_sentiment
        sentimentscore_list.append(sentiment.score)
        magnitude_list.append(sentiment.magnitude)
        print ('Progress: {}/{} rows processed'.format(i, len(description)))
    
    print("Sum of skipped rows: ", sum)
    cur.execute("DROP TABLE IF EXISTS temp")
    cur.execute("CREATE TABLE temp(DESCRIPTIONS text, SENTIMENTSCORE numeric, MAGNITUDE numeric, SENTENCESCORES text, SENTENCEMAGNITUDES text)")
    
    def insert(d, ss, m, sens, senm):
        cur.execute("INSERT INTO temp (DESCRIPTIONS, SENTIMENTSCORE, MAGNITUDE, SENTENCESCORES, SENTENCEMAGNITUDES) VALUES (?, ?, ?, ?, ?)", (d, ss, m, sens, senm))
    
    for d, ss, m, sens, senm in zip(description_list, sentimentscore_list, magnitude_list, sentences_score_list, sentences_magnitude_list):
        insert(d, ss, m, sens, senm)
    
    cur.execute("DROP TABLE IF EXISTS data11")
    cur.execute("CREATE TABLE data11 AS SELECT success.*, temp.SENTIMENTSCORE, temp.MAGNITUDE, temp.SENTENCESCORES, temp.SENTENCEMAGNITUDES FROM success, temp WHERE temp.DESCRIPTIONS IN (success.DESCRIPTION, success.DESCRIPTION_TRANSLATED)")
    con.commit()


def distribution_sentimentscore_histogram(x, label):
    print("Number of entries: ", len(x))
    print("Maximum: ", max(x))
    print("Minimum: ", min(x))
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Number of Sentences")
    fig.suptitle(label)
    ax.hist(x)
    plt.show()

def distribution_magnitude_histogram(x, label):
    print("Number of entries: ", len(x))
    print("Maximum: ", max(x))
    print("Minimum: ", min(x))
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel("Sentiment Magnitude")
    ax.set_ylabel("Number of Sentences")
    fig.suptitle(label)
    ax.hist(x)
    plt.show()

def scatter_linearity(x, x_label, y, y_label, name):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.suptitle(name)    
    plt.scatter(x, y)
    plt.show()
    


def linear_regression():
    # https://stackoverflow.com/questions/11479064/multiple-linear-regression-in-python
    from sklearn.linear_model import LinearRegression
    from sklearn.feature_selection import f_regression
    
    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()
    
    # y-variable in regression is funding gap  
    cur.execute("SELECT DAYS_NEEDED FROM data12")
    y = cur.fetchall()
    y = np.array([i[0] for i in y])     # list of int
    print("y shape: ", y.shape)

    # x1-variable in regression is the project description length ("WORD_COUNT")
    cur.execute("SELECT WORD_COUNT FROM data12")
    x1 = cur.fetchall()
    x1 = np.array([i[0] for i in x1])
    x1 = x1.reshape(len(x1), 1)
    print("x1 shape: ", x1.shape)
    # x2-variable in regression is the description's sentiment score ("SENTIMENTSCORE")
    cur.execute("SELECT SENTIMENTSCORE FROM data12")
    x2 = cur.fetchall()
    x2 = np.array([i[0] for i in x2])
    x2 = x2.reshape(len(x2), 1)
    print("x2 shape: ", x2.shape)
    # x3-variable in regression is the description's magnitude score ("MAGNITUDE")
    cur.execute("SELECT MAGNITUDE FROM data12")
    x3 = cur.fetchall()
    x3 = np.array([i[0] for i in x3])
    x3 = x3.reshape(len(x3), 1)
    print("x3 shape: ", x3.shape)
    
    X = np.concatenate((x1,x2,x3), axis = 1)
    print("X shape: ", X.shape)
        
    regressor = LinearRegression()
    regressor.fit(X, y)
    
    f_values, p_values = f_regression(X,y)
    

    # display coefficients
    print("Coefficients: ", regressor.coef_)
    print("f-values: ", f_values)
    print("p-values: ", p_values)
    print("Intercept: ", regressor.intercept_)
    print("R Squared: ", regressor.score(X,y))
    
    
    



def main():
    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()
        
    ##### Sentiment Score Data11 ###
#    cur.execute("SELECT SENTIMENTSCORE FROM dataset11")
#    sentimentScore = cur.fetchall()
#    sentimentScore = np.array([i[0] for i in sentimentScore])
#    distribution_sentimentscore_histogram(sentimentScore, "Distribution Sentiment Score Set 11")
#    cur.execute("SELECT DAYS_NEEDED FROM data11")
#    days = cur.fetchall()
#    days = np.array([i[0] for i in days])
#    scatter_linearity(sentimentScore, "Sentiment Score", days, "Funding speed", "Plot of funding speed and sentiment score - set 11")
#
#
#    # Sentiment Magnitude
#    cur.execute("SELECT MAGNITUDE FROM dataset11")
#    magnitude = cur.fetchall()
#    magnitude = np.array([i[0] for i in magnitude])
#    distribution_magnitude_histogram(magnitude, "Distribution Sentiment Magnitude Set 11")
#    cur.execute("SELECT DAYS_NEEDED FROM data11")
#    days = cur.fetchall()
#    days = np.array([i[0] for i in days])
#    scatter_linearity(magnitude, "Sentiment magnitude", days, "Funding speed", "Plot of funding speed and sentiment magnitude - set 11")

#    ##### Sentiment Score Data12 ###
#    cur.execute("SELECT SENTIMENTSCORE FROM dataset12")
#    sentimentScore = cur.fetchall()
#    sentimentScore = np.array([i[0] for i in sentimentScore])
#    distribution_sentimentscore_histogram(sentimentScore, "Distribution Sentiment Score Set 12")
#    cur.execute("SELECT DAYS_NEEDED FROM data12")
#    days = cur.fetchall()
#    days = np.array([i[0] for i in days])
#    scatter_linearity(sentimentScore, "Sentiment Score", days, "Funding speed", "Plot of funding speed and sentiment score - set 12")
#     
#    # Sentiment Magnitude
#    cur.execute("SELECT MAGNITUDE FROM data12")
#    magnitude = cur.fetchall()
#    magnitude = np.array([i[0] for i in magnitude])
##    distribution_magnitude_histogram(magnitude, "Distribution Sentiment Magnitude Set 12")
#    cur.execute("SELECT DAYS_NEEDED FROM data12")
#    days = cur.fetchall()
#    days = np.array([i[0] for i in days])
#    scatter_linearity(magnitude, "Sentiment magnitude", days, "Funding speed", "Plot of funding speed and sentiment magnitude - set 12")
#
    ##### Sentiment Score Data21 ###
    cur.execute("SELECT SENTIMENTSCORE FROM dataset21")
    sentimentScore = cur.fetchall()
    sentimentScore = np.array([i[0] for i in sentimentScore])
    distribution_sentimentscore_histogram(sentimentScore, "Distribution Sentiment Score Set 21")
#    cur.execute("SELECT GAP FROM data21")
#    gap = cur.fetchall()
#    gap = np.array([i[0] for i in gap])
#    scatter_linearity(sentimentScore, "Sentiment Score", gap, "Funding gap", "Plot of funding gap and sentiment score - set 21")
#    
#    # Sentiment Magnitude
#    cur.execute("SELECT MAGNITUDE FROM data21")
#    magnitude = cur.fetchall()
#    magnitude = np.array([i[0] for i in magnitude])
##    distribution_magnitude_histogram(magnitude, "Distribution Sentiment Magnitude Set 21")
#    cur.execute("SELECT GAP FROM data21")
#    gap = cur.fetchall()
#    gap = np.array([i[0] for i in gap])
#    scatter_linearity(magnitude, "Sentiment magnitude", gap, "Funding gap", "Plot of funding gap and sentiment magnitude - set 21")
#

    ##### Sentiment Score Data22 ###
#    cur.execute("SELECT NORM_SCORE FROM dataset22")
#    sentimentScore = cur.fetchall()
#    sentimentScore = np.array([i[0] for i in sentimentScore])
#    distribution_sentimentscore_histogram(sentimentScore, "Distribution Normalised Sentiment Score Set 22")
#    cur.execute("SELECT GAP FROM dataset22")
#    gap = cur.fetchall()
#    gap = np.array([i[0] for i in gap])
#    scatter_linearity(sentimentScore, "Normalised Sentiment Score", gap, "Funding gap", "Plot of funding gap and normalised sentiment score - set 22")
    
#    cur.execute("SELECT SENTENCESCORES FROM dataset21")
#    sentimentScore = cur.fetchall()
#    sentimentScore = np.array([i[0] for i in sentimentScore])
#    sentences_score_list = []
#    for i in range(len(sentimentScore)):
#        sentence_score = eval(sentimentScore[i])
#        sentences_score_list.append(sentence_score)
#    sentimentScore = [item for sublist in sentences_score_list for item in sublist]
#    distribution_sentimentscore_histogram(sentimentScore, "Distribution Sentence Sentiment Score Set 21")
#   
#    cur.execute("SELECT SENTENCEMAGNITUDES FROM dataset21")
#    sentimentScore = cur.fetchall()
#    sentimentScore = np.array([i[0] for i in sentimentScore])
#    sentences_score_list = []
#    for i in range(len(sentimentScore)):
#        sentence_score = eval(sentimentScore[i])
#        sentences_score_list.append(sentence_score)
#    sentimentScore = [item for sublist in sentences_score_list for item in sublist]
#    distribution_magnitude_histogram(sentimentScore, "Distribution Sentence Sentiment Magnitude Set 21")



#
#    # Sentiment Magnitude
#    cur.execute("SELECT MAGNITUDE FROM data22")
#    magnitude = cur.fetchall()
#    magnitude = np.array([i[0] for i in magnitude])
##    distribution_magnitude_histogram(magnitude, "Distribution Sentiment Magnitude Set 22")
#    cur.execute("SELECT GAP FROM data22")
#    gap = cur.fetchall()
#    gap = np.array([i[0] for i in gap])
#    scatter_linearity(magnitude, "Sentiment magnitude", gap, "Funding gap", "Plot of funding gap and sentiment magnitude - set 22")
    
    
#    linear_regression()
    

    
if __name__ == "__main__": main()






