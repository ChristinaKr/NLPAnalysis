#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 09:52:02 2018

@author: christinakronser

Database to be found: https://drive.google.com/file/d/1KHmasvJFN4AWuflgicGeqvInMmNkKkio/view?usp=sharing

"""

import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import google.cloud.language
from google.api_core.exceptions import InvalidArgument
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/christinakronser/Downloads/My First Project-522341c822f9.json"

def select(cur, variable, table):
    """
    Database function to retrieve a variable
    """
    cur.execute("SELECT {v} FROM {t}".format(v = variable, t = table))
    variable = cur.fetchall()
    variable = [i[0] for i in variable]
    return variable


def sentiment_analysis(con, cur):
    """
    Retrieves and stores the sentiment score and magnitude on sentence and
    description level from the Google Cloud Natural Language API to the DB
    """
    # Retrieve data from DB
    description = np.array(select(cur,"DESCRIPTION", "data11"))
    description_trans = np.array(select(cur,"DESCRIPTION_TRANSLATED", "data11"))    
    
    description_list = []
    sentimentscore_list=[]
    magnitude_list=[]
    sentences_score_list=[]
    sentences_magnitude_list=[]
    sum= 0
    
    # Create a Language client
    language_client = google.cloud.language.LanguageServiceClient()
    
    # Check whether to use original or translated description
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
    
        #SAVE SENTENCE ATTRIBUTES
        score_all=[]
        magnitude_all=[]
        for y in range(len(response.sentences)):
            score_all.append((response.sentences[y].sentiment.score))
            magnitude_all.append((response.sentences[y].sentiment.magnitude))
        
        sentences_score_list.append(repr(score_all))
        sentences_magnitude_list.append(repr(magnitude_all))
        # use eval() to turn it back into a list of floats
    
        description_list.append(descr)
        sentiment = response.document_sentiment
        sentimentscore_list.append(sentiment.score)
        magnitude_list.append(sentiment.magnitude)
        print ('Progress: {}/{} rows processed'.format(i, len(description)))
    
    # Save all scores to the DB
    print("Sum of skipped rows: ", sum)
    cur.execute("DROP TABLE IF EXISTS temp")
    cur.execute("CREATE TABLE temp(DESCRIPTIONS text, SENTIMENTSCORE numeric, MAGNITUDE numeric, SENTENCESCORES text, SENTENCEMAGNITUDES text)")
    
    def insert(d, ss, m, sens, senm):
        cur.execute("INSERT INTO temp (DESCRIPTIONS, SENTIMENTSCORE, MAGNITUDE, SENTENCESCORES, SENTENCEMAGNITUDES) VALUES (?, ?, ?, ?, ?)", (d, ss, m, sens, senm))
    
    for d, ss, m, sens, senm in zip(description_list, sentimentscore_list, magnitude_list, sentences_score_list, sentences_magnitude_list):
        insert(d, ss, m, sens, senm)
    
    cur.execute("DROP TABLE IF EXISTS data22")
    cur.execute("CREATE TABLE data22 AS SELECT success.*, temp.SENTIMENTSCORE, temp.MAGNITUDE, temp.SENTENCESCORES, temp.SENTENCEMAGNITUDES FROM success, temp WHERE temp.DESCRIPTIONS IN (success.DESCRIPTION, success.DESCRIPTION_TRANSLATED)")
    con.commit()


def distribution_sentimentscore_histogram(cur, var, table, label):
    """
    Plots distribution of sentiment score
    """
    x = select(cur,var, table)
    print("Number of entries: ", len(x))
    print("Maximum: ", max(x))
    print("Minimum: ", min(x))
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Number of Loans")
    fig.suptitle(label)
    ax.hist(x, bins = 15)
    plt.show()

def distribution_magnitude_histogram(cur, var, table, label):
    """
    Plots distribution of sentiment magnitude
    """
    x = select(cur,var, table)
    print("Number of entries: ", len(x))
    print("Maximum: ", max(x))
    print("Minimum: ", min(x))
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel("Sentiment Magnitude")
    ax.set_ylabel("Number of Sentences")
    fig.suptitle(label)
    ax.hist(x, bins = 20)
    plt.show()

def scatter_linearity(cur, var1, var2, table, x_label, y_label, name):
    """
    Plots scatter plot of two variables to depict their relationship
    """

    x = select(cur,var1, table)
    y = select(cur,var2, table)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim([0,11000])
    fig.suptitle(name)    
    ax.plot(x, y, 'x')
    ax.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), color = 'r', linewidth = 1.0)
    plt.show()
    

#--------------- Sentiment on Sentence Level ---------------


def histogram_quartiles(cur,variable1, variable2, table):  
    """
    Sentiment distribution based on position of sentence in description
    """
    print("Sentiment distribution based on position of sentence in description")
    sentence_scores = select(cur,variable1, table) # multiple list of strings
    sentence_mags = select(cur,variable2, table) # multiple list of strings
    
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
#    std_score = (np.std(quartileBottom_score), np.std(halfMiddle_score), np.std(quartileTop_score))

    means_mag = (np.average(quartileBottom_mag), np.average(quartileTop_mag), np.average(quartileTop_mag))
#    std_mag = (np.std(quartileBottom_mag), np.std(quartileTop_mag), np.std(quartileTop_mag))
    fig, ax = plt.subplots()
    
    print("Means Sentiment Score: ", means_score)
    print("Means Magnitude: ", means_mag)
    
    index = np.arange(n_groups)
    bar_width = 0.35
    
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    
    rects1 = ax.bar(index, means_score, bar_width,
                    alpha=opacity, color='b',
                    error_kw=error_config,
                    label='Sentiment')
        
    rects2 = ax.bar(index + bar_width, means_mag, bar_width,
                    alpha=opacity, color='r',
                    error_kw=error_config,
                    label='Magnitude')
    
#    ax.set_xlabel('Quartiles')
    ax.set_ylabel('Scores')
    ax.set_title('Scores by sentiment and magnitude')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(('Bottom quartile', 'Middle half', 'Top quartile'))    
    ax.legend((rects1[0], rects2[0]), ('Sentiment', 'Magnitude'))
        
    
    fig.tight_layout()
    plt.show()    
    
def distribution_sentences_histo(cur, variable1, variable2, table):    
    """
    Sentiment distribution of sentences in bottom quartile (beginning of text) 
    and top quartile (end of text) in description
    """
    print("Sentiment distribution of sentences in bottom quartile (beginning of text) and top quartile (end of text) in description")
    # Retrieve data from DB
    # sentences scores are stored as string list   
    sentence_scores = select(cur,variable1, table) # multiple list of strings
    sentence_mags = select(cur,variable2, table) # multiple list of strings
    
    quartileBottom_score = []
    quartileBottom_mag = []
#    halfMiddle_score = []
#    halfMiddle_mag = []
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
    # Make a connection to the database
    con = sqlite3.connect('database.db')
    cur = con.cursor()

    # Plot distribution of sentiment score and magnitude
    distribution_sentimentscore_histogram(cur, "SENTIMENTSCORE", "data22", "Distribution Sentiment Score - Large loans")    
    distribution_magnitude_histogram(cur, "MAGNITUDE", "data22", "Distribution Sentiment Magnitude - Large loans")

    # Plot relationship between sentiment score and funding gap
    scatter_linearity(cur, "SENTIMENTSCORE", "GAP", "data22", "Sentiment Score", "Funding gap in $", "Scatter plot of funding gap and sentiment score - large loans")

    # Exploring sentiment score & magnitude on sentence level
    histogram_quartiles(cur,"SENTENCESCORES", "SENTENCEMAGNITUDES", "data22")
    distribution_sentences_histo(cur, "SENTENCESCORES", "SENTENCEMAGNITUDES", "data22")
    
if __name__ == "__main__": main()






