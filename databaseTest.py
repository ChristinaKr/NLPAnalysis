#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 12:00:37 2018

@author: christinakronser
"""
import csv, sqlite3
import matplotlib.pyplot as plt
import numpy as np
import datetime
#import re
import enchant
    

def import_data():
    """
    Import the data from a csv into a SQLite3 database
    
    Source data import: https://stackoverflow.com/questions/2887878/importing-a-csv-file-into-a-sqlite3-database-table-using-python
    
    """
    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()
    cur.execute("CREATE TABLE loans (LOAN_ID,LOAN_NAME,ORIGINAL_LANGUAGE,DESCRIPTION,DESCRIPTION_TRANSLATED,FUNDED_AMOUNT,LOAN_AMOUNT,STATUS,IMAGE_ID,VIDEO_ID,ACTIVITY_NAME,SECTOR_NAME,LOAN_USE,COUNTRY_CODE,COUNTRY_NAME,TOWN_NAME,CURRENCY_POLICY,CURRENCY_EXCHANGE_COVERAGE_RATE,CURRENCY,PARTNER_ID,POSTED_TIME,PLANNED_EXPIRATION_TIME,DISBURSE_TIME,RAISED_TIME,LENDER_TERM,NUM_LENDERS_TOTAL,NUM_JOURNAL_ENTRIES,NUM_BULK_ENTRIES,TAGS,BORROWER_NAMES,BORROWER_GENDERS,BORROWER_PICTURED,REPAYMENT_INTERVAL,DISTRIBUTION_MODEL);")
    
    with open('loans.csv') as fin: 
        # csv.DictReader uses first line in file for column headings by default
        dr = csv.DictReader(fin) # comma is default delimiter
        to_db = [(i['LOAN_ID'], i['LOAN_NAME'], i['ORIGINAL_LANGUAGE'], i['DESCRIPTION'], i['DESCRIPTION_TRANSLATED'], i['FUNDED_AMOUNT'], i['LOAN_AMOUNT'], i['STATUS'], i['IMAGE_ID'], i['VIDEO_ID'], i['ACTIVITY_NAME'], i['SECTOR_NAME'], i['LOAN_USE'], i['COUNTRY_CODE'], i['COUNTRY_NAME'], i['TOWN_NAME'], i['CURRENCY_POLICY'], i['CURRENCY_EXCHANGE_COVERAGE_RATE'], i['CURRENCY'], i['PARTNER_ID'], i['POSTED_TIME'], i['PLANNED_EXPIRATION_TIME'], i['DISBURSE_TIME'], i['RAISED_TIME'], i['LENDER_TERM'], i['NUM_LENDERS_TOTAL'], i['NUM_JOURNAL_ENTRIES'], i['NUM_BULK_ENTRIES'], i['TAGS'], i['BORROWER_NAMES'], i['BORROWER_GENDERS'], i['BORROWER_PICTURED'], i['REPAYMENT_INTERVAL'], i['DISTRIBUTION_MODEL']) for i in dr]
    
    cur.executemany("INSERT INTO loans (LOAN_ID,LOAN_NAME,ORIGINAL_LANGUAGE,DESCRIPTION,DESCRIPTION_TRANSLATED,FUNDED_AMOUNT,LOAN_AMOUNT,STATUS,IMAGE_ID,VIDEO_ID,ACTIVITY_NAME,SECTOR_NAME,LOAN_USE,COUNTRY_CODE,COUNTRY_NAME,TOWN_NAME,CURRENCY_POLICY,CURRENCY_EXCHANGE_COVERAGE_RATE,CURRENCY,PARTNER_ID,POSTED_TIME,PLANNED_EXPIRATION_TIME,DISBURSE_TIME,RAISED_TIME,LENDER_TERM,NUM_LENDERS_TOTAL,NUM_JOURNAL_ENTRIES,NUM_BULK_ENTRIES,TAGS,BORROWER_NAMES,BORROWER_GENDERS,BORROWER_PICTURED,REPAYMENT_INTERVAL,DISTRIBUTION_MODEL) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);", to_db)
    con.commit()
    con.close()


def data_validity():
    """
    Tests to check correctness of data and validity of database
    """
    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()

#    #Test
#    cur.execute("SELECT * FROM loans WHERE LOAN_ID='657307'")
#    print(cur.fetchone())
#    
#    #Test
#    cur.execute("SELECT count(*) FROM loans")
#    print(cur.fetchall())          #1419607
#    
#    #Test
#    cur.execute("SELECT STATUS FROM loans WHERE LOAN_ID='657307'")
#    print(cur.fetchone())
#    
#    cur.execute("SELECT avg(LOAN_AMOUNT) FROM loans")
#    print(cur.fetchone())            # 832.23€
    
    cur.execute("SELECT DISTINCT STATUS FROM loans")
    print(cur.fetchall())

def subset_test_funding_speed():
    """
    Tests to check which data subset to use for further analyses
    """
    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()
    
#    cur.execute("SELECT DISTINCT BORROWER_GENDERS FROM funded WHERE BORROWER_GENDERS NOT LIKE '%female%'")
#    print("Genders:", cur.fetchall())
    
    cur.execute("SELECT COUNT(*) FROM funded WHERE EUROPE = 0 AND BORROWER_GENDERS NOT LIKE '%female%' AND LOAN_AMOUNT < 1000 AND SECTOR_NAME = 'Health'")
    print("count:", cur.fetchone())
    
    
    

def continents_and_countries():
    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()
    
    # 1. Create new integer columns EUROPE, NORTH AMERICA, SOUTH AMERICA, AFRICA
#    cur.execute("ALTER TABLE funded ADD COLUMN EUROPE BOOLEAN CHECK (EUROPE IN (0,1))")
    
    # 2. Update table and set the continent column to 1 if it's the correct continent
#    cur.execute("UPDATE funded SET EUROPE = 1 WHERE COUNTRY_NAME IN ('Ukraine', 'Kosovo', 'Turkey', 'Moldova', 'Bosnia and Herzegovina', 'Bulgaria')")
    cur.execute("UPDATE funded SET EUROPE = 0 WHERE COUNTRY_NAME NOT IN ('Ukraine', 'Kosovo', 'Turkey', 'Moldova', 'Bosnia and Herzegovina', 'Bulgaria')")
    con.commit()
    
    # 3. Test if successful
    cur.execute("SELECT COUNT(COUNTRY_NAME) FROM funded WHERE EUROPE = 0")
    print("NOT Europe: ", cur.fetchall())


def distribution_funding_speed_histogram():
    """
    Distribution of funding speed
    """
    
    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()
    
    cur.execute("SELECT DAYS_NEEDED FROM funded WHERE BORROWER_GENDERS NOT LIKE '%female%' AND LOAN_AMOUNT > 4000 AND SECTOR_NAME = 'Agriculture' AND EUROPE = 0")
    days = cur.fetchall()
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
    speed = [i[0] for i in days]
    ax.hist(speed, range = (0, 80))

def distribution_funding_gap_histogram():
    """
    Distribution of funding gap
    """
    
    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()
        
    cur.execute("SELECT GAP FROM notfunded WHERE LOAN_AMOUNT > 4000 AND SECTOR_NAME = 'Agriculture'")
    days = cur.fetchall()
    print("Number of entries: ", len(days))
    print("Maximum gap: ", max(days))
    print("Minimum gap: ", min(days))
    
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Funding gap in US$")
    ax.set_ylabel("Number of loans")
    fig.suptitle('Histogram of Funding Gap')
    gap = [i[0] for i in days]
    ax.hist(gap, range = (0, 8000))

def non_recurring_db_script_days_needed_for_funding():
    """
    Adds the days needed for funding to the database and creates the table "funded"
    """
    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()
    
    cur.execute("SELECT RAISED_TIME FROM loans WHERE STATUS = 'funded'")
    raised_times = cur.fetchall()
    RAISED_TIME = []
    rt_list = []
    
    cur.execute("SELECT POSTED_TIME FROM loans WHERE STATUS = 'funded'")
    posted_times = cur.fetchall()
    POSTED_TIME = []
    pt_list = []
    
    days = []
    
    for i in range(len(raised_times)):
        raised_time = raised_times[i]
        raisedTimes = ''.join(raised_time) # cast tuple to string
        rt_list.append(raisedTimes)
        raisedTime = raisedTimes[:10]
        RAISED_TIME.append(datetime.datetime.strptime(raisedTime, "%Y-%m-%d").date())
        
        posted_time = posted_times[i]
        postedTime = ''.join(posted_time) # cast tuple to string
        pt_list.append(postedTime)
        postedTime = postedTime[:10]
        POSTED_TIME.append(datetime.datetime.strptime(postedTime, "%Y-%m-%d").date())
        
        days.append((RAISED_TIME[i] - POSTED_TIME[i]).days)
        print ('Progress: {}/{} rows processed'.format(i, len(raised_times)))
    
    # Create table containing "DAYS_RAISED" to store the days needed for funding
    cur.execute("DROP TABLE IF EXISTS days")
    cur.execute("CREATE TABLE IF NOT EXISTS days(DAYS_NEEDED integer, RAISED_TIME text, POSTED_TIME text)")
    
    def insert(days_needed, rt, pt):
        cur.execute("INSERT INTO days (DAYS_NEEDED, RAISED_TIME, POSTED_TIME) VALUES (?, ?, ?)", (days_needed, rt, pt))
    
    for d, rt, pt in zip(days, rt_list, pt_list):
        insert(d, rt, pt)
    
    cur.execute("CREATE TABLE funded AS SELECT loans.*, days.DAYS_NEEDED FROM loans, days WHERE loans.POSTED_TIME = days.POSTED_TIME AND loans.RAISED_TIME = days.RAISED_TIME AND loans.STATUS = 'funded'")
    con.commit()

def non_recurring_db_script_funding_gap():
    """
    Adds the funding gap to the database and creates the table "notfunded"
    """
    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()
    
    cur.execute("SELECT LOAN_AMOUNT FROM loans WHERE STATUS = 'expired'")
    loan_amount_exp = cur.fetchall()
    la_list = []
    
    cur.execute("SELECT FUNDED_AMOUNT FROM loans WHERE STATUS = 'expired'")
    funded_amount_exp = cur.fetchall()
    fa_list = []
    
    gaps = []
    
    for i in range(len(loan_amount_exp)):
        loan_amount = int(loan_amount_exp[i][0])
        la_list.append(loan_amount)
        
        funded_amount = int(funded_amount_exp[i][0])
        fa_list.append(funded_amount)
        
        gaps.append(la_list[i] - fa_list[i])
        print ('Progress: {}/{} rows processed'.format(i, len(loan_amount_exp)))
    
    # Create table containing "GAP" to store the funding gaps
    cur.execute("CREATE TABLE IF NOT EXISTS gap(GAP integer, LOAN_AMOUNT integer, FUNDED_AMOUNT integer)")
    
    def insert(gaps, la, fa):
        cur.execute("INSERT INTO gap (GAP, LOAN_AMOUNT, FUNDED_AMOUNT) VALUES (?, ?, ?)", (gaps, la, fa))
    
    for d, la, fa in zip(gaps, la_list, fa_list):
        insert(d, la, fa)
    
    cur.execute("CREATE TABLE notfunded AS SELECT loans.*, gap.GAP FROM loans, gap WHERE loans.FUNDED_AMOUNT = gap.FUNDED_AMOUNT AND loans.LOAN_AMOUNT = gap.LOAN_AMOUNT AND loans.STATUS = 'expired'")
    con.commit()



def non_recurring_delete_unnecessary_data():
    """
    Delete rest of the data from database so that data subset we use is the only remaining one
    """
    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()
 
### For table: funded    
    
    # Delete entries from "funded" table with negative days needed for funding (first check count)
    cur.execute("SELECT COUNT(DISTINCT LOAN_ID)  FROM funded WHERE DAYS_NEEDED <0")
    print(cur.fetchall())
    cur.execute("DELETE FROM funded WHERE DAYS_NEEDED <0")
    con.commit()
    cur.execute("SELECT COUNT(*) FROM funded")
    print("Data after deletion: ", cur.fetchone())

    # Delete projects without descriptions
    cur.execute("SELECT COUNT(LOAN_ID) FROM funded")
    print("before deletion: ", cur.fetchone())
    cur.execute("DELETE FROM funded WHERE DESCRIPTION = ''")
    con.commit()

    # Delete duplicate rows (22)
    cur.execute("DELETE FROM funded WHERE rowid not in (select max(rowid) from funded group by LOAN_ID)")
    con.commit()

## For table: notfunded

    # Delete duplicate rows (22)
    cur.execute("SELECT COUNT(LOAN_ID) FROM notfunded WHERE rowid not in (select max(rowid) from notfunded group by LOAN_ID)")
    print("before deletion duplicates: ", cur.fetchone())
    cur.execute("DELETE FROM notfunded WHERE rowid not in (select max(rowid) from notfunded group by LOAN_ID)")
    con.commit()
    cur.execute("SELECT COUNT(*) FROM notfunded")
    print("Data without duplicates: ", cur.fetchone())


    # Delete entries from "notfunded" table with negative funding gaps
    cur.execute("SELECT COUNT(*) FROM notfunded")
    print("Data before deletion: ", cur.fetchone())
    cur.execute("SELECT COUNT(DISTINCT LOAN_ID)  FROM notfunded WHERE GAP <0")
    print(cur.fetchall())
    cur.execute("DELETE FROM notfunded WHERE GAP <0")
    con.commit()
    cur.execute("SELECT COUNT(*) FROM notfunded")
    print("Data after deletion: ", cur.fetchone())

    # Delete projects without descriptions
    cur.execute("SELECT COUNT(LOAN_ID) FROM notfunded WHERE DESCRIPTION = ''")
    print("before deletion without description: ", cur.fetchone())
    cur.execute("DELETE FROM notfunded WHERE DESCRIPTION = ''")
    con.commit()
    cur.execute("SELECT COUNT(*) FROM notfunded")
    print("Final amount of data: ", cur.fetchone())

def description_length_funded():
    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()
    
    cur.execute("SELECT DESCRIPTION FROM funded")
    description = cur.fetchall()
    word_list = []
    characters_list = []
    description_list = []
    
    cur.execute("SELECT DESCRIPTION_TRANSLATED FROM funded")
    description_trans = cur.fetchall()
    
    
    for i in range(len(description)):
        if description_trans[i][0] == '':
            word_count = len(description[i][0].split())
            characters_count = len(description[i][0])
#            print("description translated: ", description_trans[i][0])
        else:
            word_count = len(description_trans[i][0].split())
            characters_count = len(description[i][0])
#            print("description translated: ", description_trans[i][0])
        word_list.append(word_count)
        characters_list.append(characters_count)
        description_list.append(description[i][0])
                
    # Create table containing "WORD_COUNT" and "CHARACTER_COUNT"
    cur.execute("CREATE TABLE IF NOT EXISTS count(WORD_COUNT integer, CHARACTER_COUNT integer, DESCRIPTION text)")
    
    def insert(word_count, character_count, description):
        cur.execute("INSERT INTO count (WORD_COUNT, CHARACTER_COUNT, DESCRIPTION) VALUES (?, ?, ?)", (word_count, character_count, description))
    
    for word, character, description in zip(word_list, characters_list, description_list):
        insert(word, character, description)
    
    cur.execute("CREATE TABLE success AS SELECT funded.*, count.WORD_COUNT, count.CHARACTER_COUNT FROM funded, count WHERE funded.DESCRIPTION = count.DESCRIPTION")
#    cur.execute("CREATE TABLE nosuccess AS SELECT notfunded.*, count.WORD_COUNT, count.CHARACTER_COUNT FROM notfunded, count WHERE notfunded.DESCRIPTION = count.DESCRIPTION")
    con.commit()
        
def description_length_notfunded():
    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()
    
    cur.execute("SELECT DESCRIPTION FROM notfunded")
    description = cur.fetchall()
    word_list = []
    characters_list = []
    description_list = []
    
    cur.execute("SELECT DESCRIPTION_TRANSLATED FROM notfunded")
    description_trans = cur.fetchall()
    
    
    for i in range(len(description)):
        if description_trans[i][0] == '':
            word_count = len(description[i][0].split())
            characters_count = len(description[i][0])
#            print("description translated: ", description_trans[i][0])
        else:
            word_count = len(description_trans[i][0].split())
            characters_count = len(description[i][0])
#            print("description translated: ", description_trans[i][0])
        word_list.append(word_count)
        characters_list.append(characters_count)
        description_list.append(description[i][0])
                
    # Create table containing "WORD_COUNT" and "CHARACTER_COUNT"
    cur.execute("CREATE TABLE IF NOT EXISTS countnotfunded(WORD_COUNT integer, CHARACTER_COUNT integer, DESCRIPTION text)")
    
    def insert(word_count, character_count, description):
        cur.execute("INSERT INTO countnotfunded (WORD_COUNT, CHARACTER_COUNT, DESCRIPTION) VALUES (?, ?, ?)", (word_count, character_count, description))
    
    for word, character, description in zip(word_list, characters_list, description_list):
        insert(word, character, description)
    
    cur.execute("CREATE TABLE nosuccess AS SELECT notfunded.*, countnotfunded.WORD_COUNT, countnotfunded.CHARACTER_COUNT FROM notfunded, countnotfunded WHERE notfunded.DESCRIPTION = countnotfunded.DESCRIPTION")
    con.commit()

def check_English_descriptions():
    index = []

    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()
    cur.execute("SELECT DESCRIPTION FROM nosuccess WHERE LOAN_AMOUNT > 2000 AND SECTOR_NAME = 'Agriculture' ")
    descriptions = cur.fetchall()
    description = [i[0] for i in descriptions]
#    description = np.array(description)
    cur.execute("SELECT DESCRIPTION_TRANSLATED FROM nosuccess WHERE LOAN_AMOUNT > 2000 AND SECTOR_NAME = 'Agriculture' ")
    description_trans = cur.fetchall()
    description_trans = [i[0] for i in description_trans]
#    description_trans = np.array(description_trans)
    
    description_list = []

    
    for i in range(len(description)):
        if description_trans[i] == '':
            descr = description[i]
        else:
            descr = description_trans[i]
        description_list.append(descr)
    
    print("len: ", len(description_list))
    
    d = enchant.Dict("en_US")
    for i in range(len(description_list)):
        print("i", i)
        print(description_list[i])
        d = description_list[i].split(' ')[2]
        if not d:
            d = description_list[i].split(' ')[3]
        if d != True:
            index.append(description_list.index(description_list[i]))
    print(index)
    
def check_english():
        
    d = enchant.Dict("en_US")
    string = "Hello this is English"
    string2 = string.split(' ', 1)[0]
    print(d.check(string2))
    
    print(string.split(' ')[3])

def descriptions_less_words():
    con = sqlite3.connect('databaseTest.db')
    cur = con.cursor()
#    cur.execute("SELECT DESCRIPTION FROM success")
#    descriptions = cur.fetchall()
#    description = [i[0] for i in descriptions]
#    index = []
#    
#    for i in range(len(description)):
#        print(i)
#        if len(description[i].split()) < 10:
#            cur.execute("SELECT LOAN_ID FROM success WHERE DESCRIPTION = ?", [description[i]])
#            d = cur.fetchall()
#            d = [i[0] for i in d]
#            print(d)
#            index.append(d)
#    print(type(index))
#    print(index)
    
#    cur.execute("SELECT DESCRIPTION from nosuccess WHERE LOAN_ID IN (1088062, 1081925, 1087368, 1088140, 1087279, 1089034, 1084524, 1089212, 1084802)" )
#    d= cur.fetchall()
#    print(d)
    
    cur.execute("SELECT COUNT(*) FROM success WHERE EUROPE = 0 AND BORROWER_GENDERS NOT LIKE '%female%' AND LOAN_AMOUNT > 1000 AND SECTOR_NAME = 'Agriculture'")
    print("before: ", cur.fetchone())


def main():
#    check_english()
#    check_English_descriptions()
    descriptions_less_words()
    
    
if __name__ == "__main__": main()



