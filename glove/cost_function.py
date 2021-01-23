# -*- coding: utf-8 -*-
"""
This module compute the cost function for glove model
"""
from typing import Union, List, Dict, Tuple
import numpy as np
import utils.util


def sigmoid(value: int, factor: Union[float, int]) -> float:
    """
    Create a scalar in order to smooth the overweighted co-occurrences

    Parameters
    ----------
    value : int
        This value is the co-occurence between a central and context word.
    factor : Union[float, int]
        This value control the slope of the smoothing function, high value
        reduce the slope

    Returns
    -------
    float
        return a scalar between 0-1.

    """

    return 1 / (1 + np.exp(-value / factor))

def cost_glove_dict(
    vocabulary: List[str],
    theta: np.ndarray,
    co_occurrences: Dict[Tuple[str, str], int],
) -> float:
    '''
    Calculate the cost function for glove model, using cooccurrences from a dictionary.

    Parameters
    ----------
    vocabulary : List[str]
        List of unique words in the corpus. initially use gen_vocabulary function
        in utils.
    theta : numpy.array
        Array that contains the vector representation of words, central and context
        representation, initially use gen_theta to get this parameter.
    co_occurrences : dict
        This dictionary contains the co-occurrence for each combination in the corpus
        between central and context word, the keys are tuples of two elements where
        the first element of key refers to central words and the second one refers
        to context word.

    Returns
    -------
    float
        Cost for the model with the currently theta.

    '''

    cost = 0
    FACTOR = 10
    dimension = len(theta) // 2 // len(vocabulary)

    for central_word, context_word in co_occurrences.keys():
        central_index = vocabulary.index(central_word)
        central_vector = utils.util.find_vector(central_index, theta, dimension)
        context_index = vocabulary.index(context_word)
        context_vector = utils.util.find_vector(
            context_index,
            theta,
            dimension,
            central = False)

        P_ij = co_occurrences[(central_word, context_word)]

        cost += (2 * sigmoid(P_ij, FACTOR) - 1) * (
            np.dot(context_vector, central_vector) - np.log(P_ij)
        ) ** 2

    return cost * (1 / 2)


def cost_glove(
    vocabulary: List[str],
    theta: np.ndarray,
    co_occurrence_mtx: np.ndarray,
) -> float:
    """
    Calculate the cost function for glove model. using cooccurrences from a matrix.

    Parameters
    ----------
    vocabulary : List[str]
        List of unique words in the corpus. initially use gen_vocabulary function
        in utils.
    theta : numpy.array
        Array that contains the vector representation of words, central and context
        representation, initially use gen_theta to get this parameter.
    co_occurrence_mtx : numpy.ndarray
        This matrix contains the co-occurrence for each combination in the corpus
        between central and context word, a matrix where the columns refers to
        central words and the rows refers to context word.

    Returns
    -------
    float
        Cost for the model with the currently theta.

    """

    cost = 0
    FACTOR = 10
    dimension = len(theta) // 2 // len(vocabulary)
    for central_index, _ in enumerate(vocabulary):
        central_vector = utils.util.find_vector(central_index, theta, dimension)
        for context_index, _  in enumerate(vocabulary):
            context_vector = utils.util.find_vector(
                context_index, theta, dimension, central=False
            )

            P_ij = co_occurrence_mtx[context_index, central_index]
            if P_ij != 0:
                cost += (2 * sigmoid(P_ij, FACTOR) - 1) * (
                    np.dot(context_vector, central_vector) - np.log(P_ij)
                ) ** 2

    return cost * (1 / 2)
