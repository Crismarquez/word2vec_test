# -*- coding: utf-8 -*-
"""
This script run the glove model in order to obtain a vector representation for
words in the vocabulary, .
"""
#import time

import pandas as pd
import nltk
import matplotlib.pyplot as plt

import utils.util
import glove.co_occurrence
import glove.cost_function
import glove.gradient


# import corpus
texts = nltk.corpus.brown.words()
texts = texts + nltk.corpus.gutenberg.words()

print("\n Size imported corpus, tockens: ", len(texts))

# clean corpus
corpus = [w.lower() for w in texts]

# hyperparameters
DIMENSION = 10  # size of word vectors
S_WINDOW = 5  # width for windows

# filter vocabulary
frequency = nltk.FreqDist(corpus)
vocabulary = [w for w, freq in frequency.items() if freq > 50]

theta = utils.util.gen_theta(vocabulary, DIMENSION, seed=123)

print("\n Size of corpus: ", len(corpus))
print("Size of vocabulary: ", len(vocabulary))

print("\n Calculating the co-occurrence matrix ...")
co_occurrence_dict = glove.co_occurrence.cooccurrences(corpus, vocabulary, S_WINDOW)

print('optimizing theta ...')
learinig_rate = 0.0005
hist_cost = []
for i in range(20):
    gradient = glove.gradient.gradient_descent_dict(vocabulary, theta, co_occurrence_dict)
    theta = theta - learinig_rate * gradient
    cost_model = glove.cost_function.cost_glove_dict(vocabulary, theta, co_occurrence_dict)
    hist_cost.append(cost_model)

plt.plot(range(len(hist_cost)), hist_cost)
plt.title("Learning")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()

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
df.to_csv('C:/Users/Cristian Marquez/Documents/Cristian/Academico/Projects/NLP/word2vec_V2/word2vec/glove_V1.csv')

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
