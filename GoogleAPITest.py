#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 09:52:02 2018

@author: christinakronser
"""
import numpy as np
import google.cloud.language
from google.cloud.language import types
from google.cloud.language import enums
import os
from google.api_core.exceptions import InvalidArgument
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/christinakronser/Downloads/My First Project-522341c822f9.json"

description_list = []
sentimentscore_list=[]
magnitude_list=[]
sentences_score_list=[]
sentences_magnitude_list=[]
sum= 0


# Create a Language client.
language_client = google.cloud.language.LanguageServiceClient()

#description = np.array(["The weather is great. I am so happy to be here. Life is wonderful", "23%Ботирали житель района Джаббор Расулова города Худжанд. Он 47-летний женатый человек, имеющий троих детей в возрасте 12, 15 и 16 лет. Ботирали в семье главный добытчик. Он параллельно занимается растениеводством и животноводством. Ботирали занят в этих сферах более 15 лет. У него большой опыт как в выращивании сельскохозяйственных культур, так и в разведении крупнорогатого скота. Ему помогает в работе его рабочий и семья. Ботирали на своем участке уже провел посевные работы. Ему теперь нужны минеральные удобрения для обработки, чтобы весной он мог получить качественный и богатый урожай. К тому же ему необходимо увеличить поголовье скота для развития своего бизнеса по животноводству. Для осуществления этих целей, к сожалению у Ботилари нет достаточных средств. В связи с этим он обратился за кредитом и полагается на Вашу поддержку. Этот кредит ему очень важен.            ", "I am so sad. Life sucks. Why am I here. I am so desperate."])
description = ["Anahit lives in Kapan city with her husband and son. She works at one of the local schools as an accountant. Anahit's son works at one of the factories in Kapan. Anahit is also involved in beekeeping. She has been running this business for twelve years and now keeps ten beehives. \r\n\r\nAnahit has applied for this loan to develop her beekeeping. With the requested amount she will purchase new beehives. Anahit plans to gather and sell more honey, which will allow her to earn more income and invest so she can add even more beehives. ", "I am happy and I smile the whole day. I want to go to bed and enjoy the TV. I love life."]

for i in range(len(description)):
    descr = description[i]
    
    document = google.cloud.language.types.Document(
        content=descr,
        type=google.cloud.language.enums.Document.Type.PLAIN_TEXT)
    # Use Language to detect the sentiment of the text.
    try:
        response = language_client.analyze_sentiment(document=document)
    except InvalidArgument as e:
        continue
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
 

print(description_list)
print(sentences_score_list)
print(type(sentences_score_list))
print(sentences_magnitude_list)
print(type(sentences_magnitude_list))


