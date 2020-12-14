"""
Show a simple example how to use the functions, also test the gradient,
compares the derivative functions with the numerical derivative.
"""

import utils.util
import functions


# input parameters
corpus = ['i', 'like', 'NLP', ',',
          'i', 'like', 'machine', 'learning', ',',
          'i', 'like', 'NLP', ',',
          'i', 'like', 'machine', 'learning', '.']
DIMENSION = 3  # dimention for each word vector
S_WINDOW = 3  # width for windows

# initial components
vocabulary = utils.util.gen_vocabulary(corpus)
vocabulary = ['i', 'like']
theta = utils.util.gen_theta(vocabulary, DIMENSION, seed=1234)

matrix_frequency = functions.matrix_frequency(corpus,
                                              vocabulary,
                                              S_WINDOW)

matrix_soft = functions.matrix_softmax(vocabulary, theta)

f_z = functions.cost_function(corpus, matrix_soft, matrix_frequency)
df_actual = functions.derivative(corpus,
                                 vocabulary,
                                 theta,
                                 matrix_soft,
                                 matrix_frequency)

df_actual
