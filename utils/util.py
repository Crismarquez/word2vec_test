"""
This is the documentation for this module
"""
import random
from typing import List, Optional, Dict
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


def gen_theta(
    vocabulary: List[str], dimension: int, seed: Optional[int] = None
) -> np.ndarray:
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
    if seed is not None:
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
    word_index: int, theta: np.ndarray, dimension: int, central: bool = True
) -> List[int]:
    """
    Find the location of a word in the theta vector in terms of start index
    and end index.

    Parameters
    ----------
    word_index : int
        Index word in the vocabulary list, use find_index function to get this
        parameter.
    theta : numpy.ndarray
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
    word_index: int, theta: np.ndarray, dimension: int, central: bool = True
) -> np.ndarray:
    """
    Extract the vector representation of a word in theta vector.

    Parameters
    ----------
    word_index : int
        Index word in the vocabulary list, use find_index function to get this
        parameter.
    theta : numpy.ndarray
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
    start, end = find_location(word_index, theta, dimension, central)

    return theta[start:end]


def random_dict(
    vocabulary: List[str],
    co_occurrences: Dict[str, int],
    sample_rate: float,
    central: bool = True,
) -> Dict[str, int]:
    """
    Select a random words and filter it in co_occurrences, allows filter in terms
    of central or context words.

    Parameters
    ----------
    vocabulary : list
        list of unique words in the corpus.
    co_occurrences : Dict[str, int]
        This dictionary contains the co-occurrence for each combination in the corpus
        between central and context word, the key is a string that contain the central
        word in the right and context word on the left, this words are separed by
        "<>" character.
    sample_rate: float
        To take a sample from the vocabulary list.
    central : bool, optional
        Especificate filter in terms of central or context words.
        The default is True.

    Returns
    -------
    Dict[Tuple[str, str], int]
        co_occurrency for filter words, central or context.

    """
    n_samples = int(len(vocabulary) * sample_rate)
    muestra = random.choices(vocabulary, k=n_samples)
    if central is True:
        sample_dict = {
            choice: co_occurrences[choice]
            for choice in co_occurrences.keys()
            if choice.split("<>")[0] in muestra
        }
    else:
        sample_dict = {
            choice: co_occurrences[choice]
            for choice in co_occurrences.keys()
            if choice.split("<>")[1] in muestra
        }

    return sample_dict
