# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 14:58:35 2023

@author: shaikh zainab
"""

import pandas as pd
from surprise import Dataset, Reader, dump
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split

df = pd.read_csv('skindataall.csv', index_col=[0])

data = df[['User_id', 'Product_Url', 'Rating_Stars']]
reader = Reader(line_format='user item rating', sep=',')
data = Dataset.load_from_df(data, reader=reader)

trainset, testset = train_test_split(data, test_size=.2)

svd = SVD()
svd.fit(trainset)

predictions = svd.test(testset)
accuracy.rmse(predictions)
accuracy.mae(predictions)

dump.dump('svd_model', algo=svd, predictions=predictions, verbose=1)