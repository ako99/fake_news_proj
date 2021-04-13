# -*- coding: utf-8 -*-
"""
Model Building
@author: Alexander Ngo
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

df = pd.read_csv("news.csv")
y = df.label

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, random_state=1)

# Initialize TfidVectorizer
tfidfvectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train = tfidfvectorizer.fit_transform(X_train)
tfidf_test = tfidfvectorizer.transform(X_test)

# Initialize PassiveAggressiveClassifier
p = PassiveAggressiveClassifier(max_iter=50)
p.fit(tfidf_train, y_train)

y_pred=p.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
score = round(score*100,2)
print(f'Accuracy: {score}%')

# Confusion Matrix
cf = confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
sns.heatmap(cf, annot=True)

# place model into file
with open("model.bin", 'wb') as f_out:
    pickle.dump(p, f_out) # write final_model in .bin file
    f_out.close()  # close the file
    
with open("vectorizer.bin", 'wb') as f_out:
    pickle.dump(tfidfvectorizer, f_out) # write vector in .bin file
    f_out.close()  # close the file 

with open("train_vector.bin", 'wb') as f_out:
    pickle.dump(tfidf_train, f_out) # write vector in .bin file
    f_out.close()  # close the file 