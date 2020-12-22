# -*- coding: utf-8 -*-
"""
This modele tests the gradient for the cost function in the skip-gram model,
this module uses the numerical derivative as an aproximation for the derivative.
"""
import numpy as np
from ..utils import util
from .. import functions


def test_derivative_numeric():
    """
    Test derivative for skip-gram model.

    Returns
    -------
    
    None.

    """
    # input parameters
    corpus = [
        "i",
        "like",
        "NLP",
        ",",
        "i",
        "like",
        "machine",
        "learning",
        ",",
        "i",
        "like",
        "NLP",
        ",",
        "i",
        "like",
        "machine",
        "learning",
        ".",
    ]
    dimension = 3  # dimention for each word vector
    s_window = 3  # width for windows

    # initial components
    vocabulary = util.gen_vocabulary(corpus)
    vocabulary = ["i", "like"]
    theta = util.gen_theta(vocabulary, dimension)

    matrix_frequency = functions.matrix_frequency(
        corpus, vocabulary, s_window
    )

    matrix_soft = functions.matrix_softmax(vocabulary, theta)

    f_z = functions.cost_function(corpus, matrix_soft, matrix_frequency)
    df_actual = functions.derivative(
        corpus, vocabulary, theta, matrix_soft, matrix_frequency
    )

    # test
    TOL = 0.00001
    count = 0
    while count < 10:
        h = np.zeros_like(theta)
        choice = np.random.choice(np.arange(len(theta)))
        h[choice] = 0.00001
        matrix_soft = functions.matrix_softmax(vocabulary, theta + h)
        f_zh = functions.cost_function(corpus, matrix_soft, matrix_frequency)

        df_approx = (f_zh - f_z) / h[choice]

        assert abs(df_actual[choice] - df_approx) < TOL
        print(abs(df_actual[choice] - df_approx))

        count += 1


test_derivative_numeric()

5 / 'platzi'
