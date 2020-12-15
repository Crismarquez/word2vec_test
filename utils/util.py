"""
This is the documentation for this module
"""

from typing import List
import numpy as np


def gen_vocabulary(corpus: List[str]) -> List[str]:
    """
    get the corpus vocabulary

    Parameters
    ----------
    corpus : list
        This list contains the tokenized corpus

    Returns
    -------
    list
        unique word list.
    """
    return list(set(corpus))


def gen_theta(vocabulary: List[str], dimension: int, seed: int = None) -> List[float]:
    """
    Generate a vector that will contain the vector representacion
    for each word, both central word and context word, the first half related
    with central words and second part related with context words.

    Parameters
    ----------
    vocabulary : list
        list of unique words in the corpus.
    dimension : int
        Size of dimension that will have each vector representation of words.
    seed : int, optional
        Set random generation. The default is None.

    Returns
    -------
    numpy.ndarray
        Random vector with size: 2 * vocabulary size * dimention, contains the
        vector representation for each word.
    """
    theta_size = 2 * len(vocabulary) * dimension
    if seed:
        np.random.seed(seed)

    return np.random.uniform(-1, 1, theta_size)


def find_index(word: str, vocabulary: List[str]) -> int:
    """
    Find location of a word in the vocabulary list.

    Parameters
    ----------
    word : str
        word to search.
    vocabulary : list
        list of unique words in the corpus.

    Returns
    -------
    int
        Index value of word in the vocabulary list.
    """
    return vocabulary.index(word)


def find_location(
    word_index: int, theta: List[float], dimension: int, central: bool = True
) -> List[int]:
    """
    Find the location of a word in the theta vector in terms of start index
    and end index.

    Parameters
    ----------
    word_index : int
        Index word in the vocabulary list, use find_index function to get this
        parameter.
    theta : list
        Array that contains the vector representation of words, initially use
        gen_theta to get this parameter.
    dimension : int
        Size of dimension that have each vector representation of words.
    central : bool, optional
        To get central or context word, if the parameter is True, the return
        will be the location in theta for  a central word, in another case the
        result will be for a context word. The default is True.

    Returns
    -------
    list
        List with two elments, first element contain the index where start the
        vector representation of word_index in theta, the second element
        contains the index where end the vector of word_index.
    """
    if central is True:
        start = word_index * dimension
        end = start + dimension
    else:
        start = len(theta) // 2 + word_index * dimension
        end = start + dimension

    return [start, end]


def find_vector(
    word_index: int, theta: List[float], dimension: int, central: bool = True
) -> List[float]:
    """
    Extract the vector representation of a word in theta vector.

    Parameters
    ----------
    word_index : int
        Index word in the vocabulary list, use find_index function to get this
        parameter.
    theta : list
        Array that contains the vector representation of words, initially use
        gen_theta to get this parameter.
    dimension : int
        Size of dimension that will have each vector representation of words.
    central : bool, optional
        To get central or context representation, if the parameter is True,
        the return will be the vector representation in theta for
        a central word, in another case the result will be for a context word.
        The default is True.

    Returns
    -------
    numpy.ndarray
        the vector representation in theta for word_index.
    """
    if central is True:
        start, end = find_location(word_index, theta, dimension)
    else:
        start, end = find_location(word_index, theta, dimension, central=False)

    return theta[start:end]
