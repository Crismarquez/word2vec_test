# -*- coding: utf-8 -*-
"""
This module compute the gradient for the cost function implemented by glove model
"""
from typing import List
import numpy as np
import utils.util
import glove.cost_function


def gradient_descent(
    vocabulary: List[str],
    theta: List[float],
    co_occurrence_mtx: List[int],
) -> List[float]:
    """
    Campute the gradient descent.

    Parameters
    ----------
    vocabulary : List[str]
        List of unique words in the corpus.
    theta : List[float]
        Array that contains the vector representation of words, central and context
        representation.
    co_occurrence_mtx : List[int]
        This matrix contains the co-occurrence for each combination in the corpus
        between central and context word, a matrix where the columns refers to
        central words and the rows refers to context word.

    Returns
    -------
    List[float]
        Array that contains the gradient for the vector theta.

    """

    grad_theta = np.zeros_like(theta)
    dimension = len(theta) // 2 // len(vocabulary)
    FACTOR = 10
    for central_index, _ in enumerate(vocabulary):
        central_vector = utils.util.find_vector(central_index, theta, dimension)
        central_start, central_end = utils.util.find_location(
            central_index, theta, dimension
        )
        for context_index, _ in enumerate(vocabulary):
            context_vector = utils.util.find_vector(
                context_index, theta, dimension, central=False
            )
            context_start, context_end = utils.util.find_location(
                context_index, theta, dimension, central=False
            )

            P_ij = co_occurrence_mtx[context_index, central_index]
            if P_ij != 0:
                dot_product = np.dot(context_vector, central_vector)
                context_gradient = (
                    (2 * glove.cost_function.sigmoid(P_ij, FACTOR) - 1)
                    * central_vector
                    * (dot_product - np.log(P_ij))
                )
                central_gradient = (
                    (2 * glove.cost_function.sigmoid(P_ij, FACTOR) - 1)
                    * context_vector
                    * (dot_product - np.log(P_ij))
                )

                grad_theta[context_start:context_end] = (
                    grad_theta[context_start:context_end] + context_gradient
                )
                grad_theta[central_start:central_end] = (
                    grad_theta[central_start:central_end] + central_gradient
                )

    return grad_theta
