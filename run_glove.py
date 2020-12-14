# -*- coding: utf-8 -*-
"""
This script run the glove model in order to obtain a vector representation for
words in the vocabulary, 
"""
import nltk
import matplotlib.pyplot as plt

import utils.util
import glove.co_occurence
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
S_WINDOW = 5  # width for windows

# filter vocabulary
frequency = nltk.FreqDist(corpus)
vocabulary = [w for w, freq in frequency.items() if freq > 50]

theta = utils.util.gen_theta(vocabulary, DIMENSION, seed = 123)

print("\n Size of corpus: ", len(corpus))
print("Size of vocabulary: ", len(vocabulary))
print('\n Calculating the co-occurrence matrix ...')

co_occurrence_mtx = glove.co_occurence.matrix_frequency(corpus,
                                                        vocabulary,
                                                        S_WINDOW)

learinig_rate = 0.0005
hist_cost = []
for i in range(50):
    gradient = glove.gradient.gradient_descent(vocabulary,
                                               theta,
                                               co_occurrence_mtx)
    theta = theta - learinig_rate * gradient
    cost_model = glove.cost_function.cost_glove(vocabulary,
                                                theta,
                                                co_occurrence_mtx)
    hist_cost.append(cost_model)

plt.plot(range(len(hist_cost)), hist_cost)
plt.title('Learning')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()
1+1
