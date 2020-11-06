import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
import pickle
from matplotlib import pyplot as plt
import json
import gc
import time

start = time.time()

with open("english_texts.json", encoding="utf8") as file:
    texts = json.loads(file.read())

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

n_components = int(sys.argv[1])
learning_decay = float(sys.argv[2])

lda = LatentDirichletAllocation(n_components=n_components, learning_decay=learning_decay, learning_method="online", n_jobs=-1)
lda.fit(X)

with open(f"model_{n_components}_{learning_decay}.p", "wb") as file:
    pickle.dump(lda, file)
gc.collect()
print(time.time() - start)