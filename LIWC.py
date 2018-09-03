#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 18:11:18 2018

@author: christinakronser
Source: https://github.com/chbrown/liwc-python

Database to be found: https://drive.google.com/file/d/1KHmasvJFN4AWuflgicGeqvInMmNkKkio/view?usp=sharing
"""

import sqlite3
import re
from collections import Counter
import numpy as np

con = sqlite3.connect('database.db')
cur = con.cursor()


def tokenize(text):
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)
        
def select(cur, variable, table):
    """
    Database function to retrieve a variable
    """
    cur.execute("SELECT {v} FROM {t}".format(v = variable, t = table))
    variable = cur.fetchall()
    variable = [i[0] for i in variable]
    return variable


import liwc
parse, category_names = liwc.load_token_parser('LIWC2007_English080730.dic')


descriptions = np.array(select(cur,"DESCRIPTION", "data11"))
description_trans = np.array(select(cur,"DESCRIPTION_TRANSLATED", "data11"))

description = []


for i in range(len(descriptions)):
    if description_trans[i] == '':
        descr = descriptions[i]
    else:
        descr = description_trans[i]
    description.append(descr)

cur.execute("SELECT LOAN_ID  FROM data11")
loan_id = cur.fetchall()
loan_id = [i[0] for i in loan_id]

cur.execute("ALTER TABLE data11 ADD COLUMN FAMILY_COUNT integer")
cur.execute("ALTER TABLE data11 ADD COLUMN HUMANS_COUNT integer")
cur.execute("ALTER TABLE data11 ADD COLUMN HEALTH_COUNT integer")
cur.execute("ALTER TABLE data11 ADD COLUMN WORK_COUNT integer")
cur.execute("ALTER TABLE data11 ADD COLUMN ACHIEVE_COUNT integer")
cur.execute("ALTER TABLE data11 ADD COLUMN NUM_COUNT integer")
cur.execute("ALTER TABLE data11 ADD COLUMN QUANT_COUNT integer")
cur.execute("ALTER TABLE data11 ADD COLUMN PRONOUNS_COUNT integer")
cur.execute("ALTER TABLE data11 ADD COLUMN INSIGHTS_COUNT integer")



for i in range(len(description)):
    description_tokens = tokenize(description[i])
    c = Counter(category for token in description_tokens for category in parse(token))
    family = c['family']
    humans = c['humans']
    health = c['health']
    work = c['work']
    achieve = c['achieve']
    num = c['number']
    quant = c['quant']
    pronouns = c['pronoun']
    insight = c['insight']
    loanID = loan_id[i]
    print ('Progress: {}/{} rows processed'.format(i, len(description)))

    
    cur.execute("UPDATE data11 SET FAMILY_COUNT = (?), HUMANS_COUNT = (?), HEALTH_COUNT =(?), WORK_COUNT = (?) WHERE LOAN_ID = (?)", (family, humans, health, work, loanID))
    cur.execute("UPDATE data11 SET NUM_COUNT = (?), QUANT_COUNT = (?), PRONOUNS_COUNT = (?), INSIGHTS_COUNT = (?) WHERE LOAN_ID = (?)", (num, quant, pronouns, insight, loanID))
    c.clear()
con.commit()

