#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 18:11:18 2018

@author: christinakronser
Source: https://github.com/chbrown/liwc-python
"""

import sqlite3
import re
from collections import Counter

def tokenize(text):
    # you may want to use a smarter tokenizer
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)

import liwc
parse, category_names = liwc.load_token_parser('LIWC2007_English080730.dic')

#gettysburg = '''This is my Daughter. I love her. Husband, friend, Son. Children, Cousin.'''
#gettysburg_tokens = tokenize(gettysburg)
## now flatmap over all the categories in all of the tokens using a generator:
#gettysburg_counts = Counter(category for token in gettysburg_tokens for category in parse(token))
## and print the results:
#print(gettysburg_counts)
#print("type: ", type(gettysburg_counts))
#print(gettysburg_counts['article'])
#print(gettysburg_counts.most_common(3))
#print(gettysburg_counts.items())
#print(type(gettysburg_counts.items()))

#c.clear()      # reset all counts

con = sqlite3.connect('databaseTest.db')
cur = con.cursor()
cur.execute("SELECT DESCRIPTION FROM data22")
descriptions = cur.fetchall()
descriptions = [i[0] for i in descriptions]
cur.execute("SELECT DESCRIPTION_TRANSLATED FROM data22")
description_trans = cur.fetchall()
description_trans = [i[0] for i in description_trans]

description = []


for i in range(len(descriptions)):
    if description_trans[i] == '':
        descr = descriptions[i]
    else:
        descr = description_trans[i]
    description.append(descr)

cur.execute("SELECT LOAN_ID  FROM data22")
loan_id = cur.fetchall()
loan_id = [i[0] for i in loan_id]

#cur.execute("ALTER TABLE data22 ADD COLUMN FAMILY_COUNT integer")
#cur.execute("ALTER TABLE data22 ADD COLUMN HUMANS_COUNT integer")
#cur.execute("ALTER TABLE data22 ADD COLUMN HEALTH_COUNT integer")
#cur.execute("ALTER TABLE data22 ADD COLUMN WORK_COUNT integer")
cur.execute("ALTER TABLE data22 ADD COLUMN ACHIEVE_COUNT integer")


for i in range(len(description)):
    description_tokens = tokenize(description[i])
    c = Counter(category for token in description_tokens for category in parse(token))
#    family = c['family']
#    humans = c['humans']
#    health = c['health']
#    work = c['work']
    achieve = c['achieve']
    loanID = loan_id[i]
    print(achieve)
    
#    cur.execute("UPDATE data22 SET FAMILY_COUNT = (?), HUMANS_COUNT = (?), HEALTH_COUNT =(?), WORK_COUNT = (?) WHERE LOAN_ID = (?)", (family, humans, health, work, loanID))
    cur.execute("UPDATE data22 SET ACHIEVE_COUNT = (?) WHERE LOAN_ID = (?)", (achieve, loanID))
    c.clear()
con.commit()


#
#con = sqlite3.connect('databaseTest.db')
#cur = con.cursor()
#cur.execute("SELECT DAYS_NEEDED FROM data22")
#descriptions = cur.fetchall()
#descriptions = [i[0] for i in descriptions]
#np.percentile(descriptions,100)

