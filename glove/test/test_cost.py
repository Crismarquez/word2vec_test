# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 11:55:01 2020

@author: Cristian Marquez
"""
import numpy as np
import glove.co_occurence
import glove.cost_function


def test_cost():
    '''
    Test the cost function using two words with two dimensions in the vector
    representation theta.

    Returns
    -------
    None.

    '''
    vocabulary = ["i", "like"]
    i_c = np.array([0.1, 0.2])
    like_c = np.array([-0.2, 0.1])
    i_u = np.array([-0.2, 0.2])
    like_u = np.array([-0.1, 0.3])

    theta = np.array([0.1, 0.2, -0.2, 0.1, -0.2, 0.2, -0.1, 0.3])

    cooccurrence_matx = np.array([[0, 4], [5, 0]])

    factor = 10
    f_i_like = 2 * glove.cost_function.sigmoid(5, factor) - 1
    f_like_i = 2 * glove.cost_function.sigmoid(4, factor) - 1

    cost = (
        (f_i_like * (np.dot(i_c, like_u) - np.log(5)) ** 2)
        + (f_like_i * (np.dot(like_c, i_u) - np.log(4)) ** 2)
    ) * (1 / 2)

    f_cost = glove.cost_function.cost_glove(vocabulary, theta, cooccurrence_matx)

    assert cost == f_cost


test_cost()
