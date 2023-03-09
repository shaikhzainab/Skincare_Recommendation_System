# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 15:07:58 2023

@author: shaikh zainab
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


df = pd.read_csv('skindataall.csv', index_col=[0])

def content_recommender(product):
    df_cont = df[['Product', 'Product_id', 'Ingredients', 'Product_Url', 'Ing_Tfidf', 'Rating']]
    df_cont.drop_duplicates(inplace=True)
    df_cont = df_cont.reset_index(drop=True)
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(df_cont['Ingredients'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    titles = df_cont[['Product', 'Ing_Tfidf', 'Rating']]
    indices = pd.Series(df_cont.index, index=df_cont['Product'])
    idx = indices[product]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    product_indices = [i[0] for i in sim_scores]
    return titles.iloc[product_indices]

content_recommender('Gold Camellia Beauty Oil')
