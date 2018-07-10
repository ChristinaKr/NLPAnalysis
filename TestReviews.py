#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 18:35:44 2018

@author: christinakronser
"""

import sqlite3
import matplotlib.pyplot as plt
import numpy as np

def distribution_histogram():
    """
    Distribution of variable
    """
    
    con = sqlite3.connect('TestDescriptions.db')
    cur = con.cursor()
    cur.execute("SELECT polarity_confidence FROM Descriptions")
    polarity = cur.fetchall()
    polarity = np.array([i[0] for i in polarity])
    print(type(polarity))

    
    print("Number of entries: ", len(polarity))
    print("Maximum entry: ", max(polarity))
    print("Minimum entry: ", min(polarity))
    
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
#    plt.style.use('seaborn-white')
    ax.set_xlabel("Polarity confidence")
    ax.set_ylabel("Number of reviews")
    fig.suptitle('Polarity confidence of test reviews')
    ax.hist(polarity, bins = 25)
    plt.show()
    
def sort():    
    con = sqlite3.connect('TestDescriptions.db')
    cur = con.cursor()
    cur.execute("SELECT polarity_confidence FROM Descriptions WHERE polarity_confidence > 0.55")
    pol_conf_high = cur.fetchall()
    print(np.average(pol_conf_high))
    cur.execute("SELECT polarity_confidence FROM Descriptions WHERE polarity_confidence < 0.55")
    pol_conf_low = cur.fetchall()
    print(np.average(pol_conf_low))
    
    #4- for high: plot distribution of sentiment
    plt.hist(pol_conf_high)
    
    #4- take a handful from bottom, mid, top quartiles (say 12 in total), read them, see if you would agree
    high_sort = pol_conf_high.sort()
    print(high_sort)
    
def database():
    con = sqlite3.connect('TestDescriptions.db')
    cur = con.cursor()
#    cur.execute("CREATE TABLE nosuccesssentiment AS SELECT Original.*, Descriptions.polarity, Descriptions.polarity_confidence FROM Original, Descriptions WHERE Original.Descr1 = Descriptions.Descriptions OR Original.Descr2 = Descriptions.Descriptions ")
    cur.execute("CREATE TABLE nosuccesssentiment AS SELECT Original.*, Descriptions.polarity, Descriptions.polarity_confidence FROM Original, Descriptions WHERE Descriptions.Descriptions IN (Original.Descr1, Original.Descr2) ")
    con.commit()
    
    







def main():
    database()
    
if __name__ == "__main__": main()

