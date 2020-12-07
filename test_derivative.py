# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:50:22 2020

@author: Cristian Marquez
"""
import numpy as np
import utils.util
import functions


def test_derivative_numeric():
    # input parameters
    corpus = ['i', 'like', 'NLP', ',',
              'i', 'like', 'machine', 'learning', ',',
              'i', 'like', 'NLP', ',',
              'i', 'like', 'machine', 'learning', '.']
    DIMENSION = 3  # dimention for each word vector
    S_WINDOW = 3  # width for windows
    
    # initial components
    vocabulary = utils.util.gen_vocabulary(corpus)
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


    # test
    TOL = 0.00001
    for k in range(10):
        h = np.zeros_like(theta)
        choice = np.random.choice(np.arange(len(theta)))
        h[choice] = 0.00001
        matrix_soft = functions.matrix_softmax(vocabulary, theta + h)
        f_zh = functions.cost_function(corpus, matrix_soft, matrix_frequency)

        df_approx = (f_zh - f_z) / h[choice]

        assert abs(df_actual[choice] - df_approx) < TOL

