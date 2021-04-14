# -*- coding: utf-8 -*-
"""
Model Test
@author: Alexander Ngo
"""

import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

file = open("model.bin",'rb')
model = pickle.load(file)

df = pd.read_csv("news.csv")

file = open("vectorizer.bin",'rb')
vectorizer = pickle.load(file)

# Article Text Goes Here
d = """
"""

d = [d]

test = vectorizer.transform(d)
print(model.predict(test))