# -*- coding: utf-8 -*-
"""
This module tests the gradient for the cost function in the glove model,
this module uses the numerical derivative as an aproximation for the derivative.
"""
import numpy as np
import utils.util
import glove.co_occurrence
import glove.cost_function
import glove.gradient


def test_derivative_numeric_one_dimension_dict():
    """
    Tests the derivative when vector representation of words has only
    one dimension, tests a vocabulary with two words.
    """
    # input parameters
    vocabulary = ["i", "like"]
    theta = np.array([0.1, -0.2, -0.2, -0.1])

    cooccurrences = {"i<>like": 5, "like<>i": 4}

    f_z = glove.cost_function.cost_glove_dict(vocabulary, theta, cooccurrences)

    df_actual = glove.gradient.gradient_descent_dict(vocabulary, theta, cooccurrences)

    # test
    TOL = 0.0001
    count = 0
    while count < 10:
        h = np.zeros_like(theta)
        choice = np.random.choice(np.arange(len(theta)))
        h[choice] = 0.00001
        f_zh = glove.cost_function.cost_glove_dict(vocabulary, theta + h, cooccurrences)

        df_approx = (f_zh - f_z) / h[choice]

        count += 1

        assert abs(df_actual[choice] - df_approx) < TOL


def test_derivative_numeric_two_words_dict():
    """
    Tests the derivative when vector representation of words has two dimension,
    tests a vocabulary with two words.
    """
    # input parameters
    vocabulary = ["i", "like"]
    theta = np.array([0.1, 0.2, -0.2, 0.1, -0.2, 0.2, -0.1, 0.3])

    cooccurrences = {"i<>like": 5, "like<>i": 4}

    f_z = glove.cost_function.cost_glove_dict(vocabulary, theta, cooccurrences)

    df_actual = glove.gradient.gradient_descent_dict(vocabulary, theta, cooccurrences)

    # test
    TOL = 0.0001
    count = 0
    while count < 10:
        h = np.zeros_like(theta)
        choice = np.random.choice(np.arange(len(theta)))
        h[choice] = 0.00001
        f_zh = glove.cost_function.cost_glove_dict(vocabulary, theta + h, cooccurrences)

        df_approx = (f_zh - f_z) / h[choice]

        count += 1

        assert abs(df_actual[choice] - df_approx) < TOL


def test_derivative_numeric_more_words_dict():
    """
    Tests the derivative when vector representation of words has five dimension
    and a vocabulary with more than two words.
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
    DIMENSION = 5  # dimention for each word vector
    S_WINDOW = 3  # width for windows

    # initial components
    vocabulary = utils.util.gen_vocabulary(corpus)
    theta = utils.util.gen_theta(vocabulary, DIMENSION)

    cooccurrences = glove.co_occurrence.cooccurrences(corpus, vocabulary, S_WINDOW)

    f_z = glove.cost_function.cost_glove_dict(vocabulary, theta, cooccurrences)

    df_actual = glove.gradient.gradient_descent_dict(vocabulary, theta, cooccurrences)

    # test
    TOL = 0.0001
    count = 0
    while count < 10:
        h = np.zeros_like(theta)
        choice = np.random.choice(np.arange(len(theta)))
        h[choice] = 0.00001
        f_zh = glove.cost_function.cost_glove_dict(vocabulary, theta + h, cooccurrences)

        df_approx = (f_zh - f_z) / h[choice]

        count += 1

        assert abs(df_actual[choice] - df_approx) < TOL


def test_derivative_numeric_one_dimension():
    """
    Tests the derivative when vector representation of words has only
    one dimension, tests a vocabulary with two words.
    """
    # input parameters
    vocabulary = ["i", "like"]
    theta = np.array([0.1, -0.2, -0.2, -0.1])

    cooccurrence_matx = np.array([[0, 14], [10, 0]])

    f_z = glove.cost_function.cost_glove(vocabulary, theta, cooccurrence_matx)

    df_actual = glove.gradient.gradient_descent(vocabulary, theta, cooccurrence_matx)

    # test
    TOL = 0.0001
    count = 0
    while count < 10:
        h = np.zeros_like(theta)
        choice = np.random.choice(np.arange(len(theta)))
        h[choice] = 0.00001
        f_zh = glove.cost_function.cost_glove(vocabulary, theta + h, cooccurrence_matx)

        df_approx = (f_zh - f_z) / h[choice]

        count += 1

        assert abs(df_actual[choice] - df_approx) < TOL


def test_derivative_numeric_two_words():
    """
    Tests the derivative when vector representation of words has two dimension,
    tests a vocabulary with two words.
    """
    # input parameters
    vocabulary = ["i", "like"]
    theta = np.array([0.1, 0.2, -0.2, 0.1, -0.2, 0.2, -0.1, 0.3])

    cooccurrence_matx = np.array([[0, 14], [10, 0]])

    f_z = glove.cost_function.cost_glove(vocabulary, theta, cooccurrence_matx)

    df_actual = glove.gradient.gradient_descent(vocabulary, theta, cooccurrence_matx)

    # test
    TOL = 0.0001
    count = 0
    while count < 10:
        h = np.zeros_like(theta)
        choice = np.random.choice(np.arange(len(theta)))
        h[choice] = 0.00001
        f_zh = glove.cost_function.cost_glove(vocabulary, theta + h, cooccurrence_matx)

        df_approx = (f_zh - f_z) / h[choice]

        count += 1

        assert abs(df_actual[choice] - df_approx) < TOL


def test_derivative_numeric_more_words():
    """
    Tests the derivative when vector representation of words has five dimension
    and a vocabulary with more than two words.
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
    DIMENSION = 5  # dimention for each word vector
    S_WINDOW = 3  # width for windows

    # initial components
    vocabulary = utils.util.gen_vocabulary(corpus)
    theta = utils.util.gen_theta(vocabulary, DIMENSION)

    cooccurrence_matx = glove.co_occurrence.matrix_frequency(
        corpus, vocabulary, S_WINDOW
    )

    f_z = glove.cost_function.cost_glove(vocabulary, theta, cooccurrence_matx)

    df_actual = glove.gradient.gradient_descent(vocabulary, theta, cooccurrence_matx)

    # test
    TOL = 0.0001
    count = 0
    while count < 10:
        h = np.zeros_like(theta)
        choice = np.random.choice(np.arange(len(theta)))
        h[choice] = 0.00001
        f_zh = glove.cost_function.cost_glove(vocabulary, theta + h, cooccurrence_matx)

        df_approx = (f_zh - f_z) / h[choice]

        count += 1

        assert abs(df_actual[choice] - df_approx) < TOL
