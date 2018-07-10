#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 16:57:06 2018

@author: christinakronser
"""

import numpy as np
import sqlite3
import xlwt
from tempfile import TemporaryFile


con = sqlite3.connect('databaseTest.db')
cur = con.cursor()
#cur.execute("SELECT DESCRIPTION FROM nosuccess WHERE LOAN_AMOUNT > 2000 AND SECTOR_NAME = 'Agriculture' ")
#descriptions = cur.fetchall()
#descriptions = [i[0] for i in descriptions]
#description = np.array(descriptions)
#cur.execute("SELECT DESCRIPTION_TRANSLATED FROM nosuccess WHERE LOAN_AMOUNT > 2000 AND SECTOR_NAME = 'Agriculture' ")
#description_trans = cur.fetchall()
#description_trans = [i[0] for i in description_trans]
#description_trans = np.array(description_trans)
#
#
#description_list = []
#
#
#for i in range(len(description)):
#    if description_trans[i] == '':
#        descr = description[i]
#    else:
#        descr = description_trans[i]
#    description_list.append(descr)
#
#
## output description array in excel which can be inputted in RapidMiner tool
#
#book = xlwt.Workbook()
#sheet1 = book.add_sheet('sheet1')
#
#supersecretdata = description_list
#
#column_number = 1
#sheet1.write(0, column_number, 'Descriptions')
#for i,e in enumerate(supersecretdata):
#    sheet1.write(i+1,1,e)
#
#name = "random.xls"
#book.save(name)
#book.save(TemporaryFile())

# perform RapidMiner sentiment analysis

# input it through DB Browser for SQLite again and put into one table
cur.execute("DROP TABLE IF EXISTS nosuccesssentiment")
cur.execute("CREATE TABLE nosuccesssentiment AS SELECT nosuccess.*, Dataset22.polarity, Dataset22.polarity_confidence FROM nosuccess, Dataset22 WHERE Dataset22.Descriptions IN (nosuccess.DESCRIPTION_TRANSLATED, nosuccess.DESCRIPTION) ")
con.commit()








