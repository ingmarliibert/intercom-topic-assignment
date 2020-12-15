#!/usr/bin/env python
# coding: utf-8

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
import pickle
from matplotlib import pyplot as plt
import json
import gc
import time

with open("english_texts.json", encoding="utf8") as file:
    texts = json.loads(file.read())

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

n_topics = [3, 5, 7]
learning_decays = [.5, .7, .9]
start = time.time()
for n_components in n_topics:
    for learning_decay in learning_decays:
        if n_components == 3 and abs(learning_decay - 0.5) < 0.1:
            continue 
        lda = LatentDirichletAllocation(n_components=n_components, learning_decay=learning_decay, learning_method="online", n_jobs=-1)
        lda.fit(X)
        with open(f"model_{n_components}_{learning_decay}.p", "wb") as file:
            pickle.dump(lda, file)
        print(f"model_{n_components}_{learning_decay}.p written.")
        print(time.time() - start)
        lda = None
        gc.collect()
