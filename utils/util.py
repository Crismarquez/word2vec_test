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
    Dict[str, int]
        co_occurrency for filter words, central or context.

    """
    n_samples = int(len(vocabulary) * sample_rate)
    sample = random.choices(vocabulary, k=n_samples)
    if central is True:
        sample_dict = {
            choice: co_occurrences[choice]
            for choice in co_occurrences.keys()
            if choice.split("<>")[0] in sample
        }
    else:
        sample_dict = {
            choice: co_occurrences[choice]
            for choice in co_occurrences.keys()
            if choice.split("<>")[1] in sample
        }

    return sample_dict


def get_glove_vectors(
    vocabulary: List[str], theta: np.ndarray, central: bool = False
) -> Dict[str, list]:
    """
    Organize the word vector representation in a dictionary.

    Parameters
    ----------
    vocabulary : list
        list of unique words in the corpus.
    theta : numpy.ndarray
        Array that contains the vector representation of words.
    central : bool, optional
        Especificate filter in terms of central or context words.
        The default is False. it means the default return the context words.

    Returns
    -------
    Dict[str, list]
        Dictionary that the key is the word and the value is the vector representation.

    """

    dimension = len(theta) // 2 // len(vocabulary)
    data = {}
    for word in vocabulary:
        word_index = vocabulary.index(word)
        word_vector = find_vector(word_index, theta, dimension, central=central)
        data[word] = list(word_vector)

    return data


def get_labels(word_label: Dict[str, str]) -> List[str]:
    """
    Obtain the labels related in the model.

    Parameters
    ----------
    word_label : Dict[str, str]
        Dictionary that contain the classification of words, the key represent
        the word and the value is the label or topic.

    Returns
    -------
    List[str]
        This list contain all posible labels in the model given.

    """

    return list(set(word_label.values()))


def gen_theta_class_words(labels: List[str], dimension: int) -> Dict[str, np.ndarray]:
    """
    Create initial parameters theta, represent the weights for model the labels.

    Parameters
    ----------
    labels : List[str]
        This list contain all posible labels in the model given.
    dimension : int
        Size of dimension that will have each vector of weights.

    Returns
    -------
    label_vector = Dict[str, np.ndarray]
        Dictionary where the key represent the label and tha value represents the
        vector of weights.
    """

    return {label: np.random.uniform(-1, 1, dimension) for label in labels}


def gen_grandient(theta: Dict[str, np.ndarray]) -> Dict[str, int]:
    """
    Generate the dictionary in order to save the futures values for the gradient,
    useful to avoid check if the key exist when the gradient is updating.

    Parameters
    ----------
    theta : Dict[str, np.ndarray]
        Dictionary where the key represent the label and tha value represents the
        vector of weights.

    Returns
    -------
    Dict[str, int]
        DESCRIPTION.

    """

    return {label: 0 for label in theta.keys()}
