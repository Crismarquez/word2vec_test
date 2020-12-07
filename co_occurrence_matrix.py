# -*- coding: utf-8 -*-
"""
Coocurency matrix
"""

import nltk
from nltk.book import text1, text2, text4, text5, text6, text7, text8
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.preprocessing
import sklearn.decomposition  # for PCA
import sklearn.cluster
import scipy.cluster.hierarchy as sch

import functions


# import corpus
text = nltk.corpus.brown.words()
text = text + text1 + text2 + text4 + text5 + text6 + text7 + text8
print("Size imported corpus: ", len(text))

# clean corpus
stopwords = nltk.corpus.stopwords.words("english")
corpus = [w.lower() for w in text if w.lower() not in stopwords]

# filter vocabulary
frequency = nltk.FreqDist(corpus)
frequency.plot(50, cumulative=False)
print([w for w, freq in frequency.items() if freq > 5000])
vocabulary = [w for w, freq in frequency.items() if 30 > freq < 5000]

print("Size of corpus: ", len(corpus))
print("Size of vocabulary: ", len(vocabulary))

S_WINDOW = 5  # width for windows
coocur_matx = functions.matrix_frequency(corpus, vocabulary, S_WINDOW)

print("Frequencies obtained: ", coocur_matx.sum())

# normalized matrix
coocur_matx = sklearn.preprocessing.normalize(coocur_matx, axis=1)


pca = sklearn.decomposition.PCA(n_components=5)
pca_r = pca.fit_transform(coocur_matx)

data = pd.DataFrame(pca_r, index=vocabulary)

choose_words = [
    "man",
    "women",
    "person",
    "human",
    "boy",
    "car",
    "road",
    "motor",
    "speed",
]

print(
    "This words not exist in the vocabulary: ",
    [w for w in choose_words if w not in list(data.index)],
)

data_filter = data.loc[choose_words]
hierarchy = sch.linkage(data_filter, method="ward")
dendrogram = sch.dendrogram(hierarchy, labels=list(data_filter.index))

plt.title("Dendogram")
plt.xlabel("Words")
plt.ylabel("Euclidean distance")
plt.show()

choose_words = ["great", "good", "fine", "satisfactory", "terrible", "bad", "poor"]

print(
    "This words not exist in the vocabulary: ",
    [w for w in choose_words if w not in list(data.index)],
)

data_filter = data.loc[choose_words]
hierarchy = sch.linkage(data_filter, method="ward")
dendrogram = sch.dendrogram(hierarchy, labels=list(data_filter.index))

plt.title("Dendogram")
plt.xlabel("Words")
plt.ylabel("Euclidean distance")
plt.show()


# hc = sklearn.cluster.AgglomerativeClustering(n_clusters = 2,
#                     affinity = 'euclidean',
#                     linkage = 'ward')

# y_hc = hc.fit_predict(data_filter)

# for word, cluster in zip(data_filter.index, y_hc):
#     print('word: ', word, ' , cluster: ', cluster)
