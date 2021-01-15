# -*- coding: utf-8 -*-
"""
This script run the glove model in order to obtain a vector representation for
words in the vocabulary.
"""
#import time

import nltk
import matplotlib.pyplot as plt

import utils.util
import glove.co_occurrence
import glove.cost_function
import glove.gradient


# import corpus
text = nltk.corpus.brown.words()
text = text + nltk.corpus.gutenberg.words()
print("\n Size imported corpus, tockens: ", len(text))

# clean corpus
corpus = [w.lower() for w in text]

# hyperparameters
DIMENSION = 10  # size of word vectors
S_WINDOW = 3  # width for windows

# filter vocabulary
frequency = nltk.FreqDist(corpus)
vocabulary = [w for w, freq in frequency.items() if freq > 50]

theta = utils.util.gen_theta(vocabulary, DIMENSION, seed=123)

print("\n Size of corpus: ", len(corpus))
print("Size of vocabulary: ", len(vocabulary))

print("\n Calculating the co-occurrence matrix ...")
co_occurrence_dict = glove.co_occurrence.cooccurrences(corpus, vocabulary, S_WINDOW)

learinig_rate = 0.0005
hist_cost = []
for i in range(5):
    gradient = glove.gradient.gradient_descent_dict(vocabulary, theta, co_occurrence_dict)
    theta = theta - learinig_rate * gradient
    cost_model = glove.cost_function.cost_glove_dict(vocabulary, theta, co_occurrence_dict)
    hist_cost.append(cost_model)

plt.plot(range(len(hist_cost)), hist_cost)
plt.title("Learning")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()


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
