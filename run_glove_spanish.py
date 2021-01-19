# -*- coding: utf-8 -*-
"""
This script run the glove model in order to obtain a vector representation for
words in the vocabulary.
"""
#import time
import re
import pickle

import pandas as pd
import nltk
import textacy.datasets
import matplotlib.pyplot as plt

import utils.util
import glove.co_occurrence
import glove.cost_function
import glove.gradient


# import corpus
ds = textacy.datasets.Wikipedia(lang="es", version="current")
ds.download()
ds.info


texts = []
for text in ds.texts(limit=1000):
    texts = texts + re.compile(r'\W+', re.UNICODE).split(text)

print("\n Size imported corpus, tockens: ", len(texts))

# clean corpus
corpus = [w.lower() for w in texts]


# hyperparameters
DIMENSION = 50  # size of word vectors
S_WINDOW = 5  # width for windows

# filter vocabulary
frequency = nltk.FreqDist(corpus)
vocabulary = [w for w, freq in frequency.items() if freq > 20]

theta = utils.util.gen_theta(vocabulary, DIMENSION, seed=123)

print("\n Size of corpus: ", len(corpus))
print("Size of vocabulary: ", len(vocabulary))

print("\n Calculating the co-occurrence matrix ...")
co_occurrence_dict = glove.co_occurrence.cooccurrences(corpus, vocabulary, S_WINDOW)
print('Size of cooccurrences (conexions): ', len(co_occurrence_dict))

# if we want to save the hist cost
# learinig_rate = 0.0005  - este multiplicador no sive para 8 millones con 26 mil pala
# learinig_rate = 0.0002 # propuesta 
print('optimizing theta ...')
learinig_rate = 0.0008
hist_cost_2 = []
for i in range(10):
    print(f'started cicle {i}')
    gradient = glove.gradient.gradient_descent_dict(vocabulary, theta, co_occurrence_dict)
    theta = theta - learinig_rate * gradient
    print(gradient.max())
    print(gradient.min())
    cost_model = glove.cost_function.cost_glove_dict(vocabulary, theta, co_occurrence_dict)
    hist_cost_2.append(cost_model)
    print(f'finished cicle {i}')


plt.plot(range(len(hist_cost)), hist_cost)
plt.plot(range(len(hist_cost_2)), hist_cost_2)
plt.plot(range(len(hist_cost_3)), hist_cost_3)
plt.title("Learning")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()
plt.savefig('learning_hist.png')

# if we do not want to save the hist cost
print('optimizing theta ...')
path = 'C:/Users/Cristian Marquez/Documents/Cristian/Academico/Projects/NLP/word2vec_V2/'
print(glove.cost_function.cost_glove_dict(vocabulary, theta, co_occurrence_dict))
learinig_rate = 0.0005
for i in range(2):
    print(f'started cicle {i}')
    gradient = glove.gradient.gradient_descent_dict(vocabulary, theta, co_occurrence_dict)
    theta = theta - learinig_rate * gradient
    with open(path + 'theta_spanish' + '.pkl', 'wb') as f:
        pickle.dump(theta, f, pickle.HIGHEST_PROTOCOL)
    print(f'finished cicle {i}')


# save data
data_context = {}
for context_word in vocabulary:
    context_index = vocabulary.index(context_word)
    context_vector = utils.util.find_vector(context_index,
                                                theta,
                                                DIMENSION,
                                                central = False)
    data_context[context_word] = context_vector

df = pd.DataFrame(data_context)
df = df.T
df.to_csv('glove_V1_spanish.csv')


import numpy as np
central_index = utils.util.find_index('de', vocabulary)
central_vector = utils.util.find_vector(central_index, theta, DIMENSION)

context_index = utils.util.find_index('la', vocabulary)
context_vector = utils.util.find_vector(context_index, theta, DIMENSION, central=False)

dot_product = np.dot(context_vector, central_vector)
dot_product - np.log(co_occurrence_dict[('de', 'la')])

np.log(co_occurrence_dict[('de', 'la')])
np.log(330000)

# # save and read coocurrence dictionary
# with open(path + 'theta_spanish' + '.pkl', 'wb') as f:
#             pickle.dump(theta, f, pickle.HIGHEST_PROTOCOL)

with open(path + 'co_occurence_spanish' '.pkl', 'rb') as f:
    co_occurrence_dict = pickle.load(f)

with open(path + 'theta_spanish' '.pkl', 'rb') as f:
    theta = pickle.load(f)

with open(path + 'vocabulary_spanish' '.pkl', 'rb') as f:
    vocabulary = pickle.load(f)
  
# # implement with matrix
# print("\n Calculating the co-occurrence matrix ...")
# com = time.time()
# co_occurrence_mtx = glove.co_occurrence.matrix_frequency(corpus, vocabulary, S_WINDOW)
# fin = time.time()
# print('time to create a co-occurence matrix = ', fin-com)

# com = time.time()
# cost_model = glove.cost_function.cost_glove(vocabulary, theta, co_occurrence_mtx)
# fin = time.time()
# print('time to compute cost from matrix = ', fin-com)
