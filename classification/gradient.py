"""
gradient for cost functions
"""
from typing import Dict, List
import numpy as np

import utils.util


def denom_gradient(
    word_label: Dict[str, str],
    word_vector: Dict[str, List[float]],
    theta: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Compute the summatory of exponential dot product between each label vector
    for word vector representation. usefull to not re-calculate the nominator
    in gradient for classification words model.

    Parameters
    ----------
    word_label : Dict[str, str]
        This list contain all posible labels in the model given.
    word_vector : Dict[str, list]
        Contain the word vector representation from glove.
        key for word, value for the vector.
    theta: Dict[str, np.ndarray]
        Contain the label vector weights.
        key for word, value for the vector.

    Returns
    -------
    Dict[str, float].
    the key represent the word involved in the model, the value is the sum for each
    dot product between the word and the labels

    """
    labels = utils.util.get_labels(word_label)
    denom = {}
    for word in word_label.keys():
        denom[word] = sum(
            [np.exp(np.dot(theta[label], word_vector[word])) for label in labels]
        )

    return denom


def gradient_classification_words(
    word_label: Dict[str, str],
    word_vector: Dict[str, list],
    theta: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Compute the gradient for theta, only take in cosideration the words and labels
    involve in word_label.

    Parameters
    ----------
    word_label : Dict[str, str]
        This list contain all posible labels in the model given.
    word_vector : Dict[str, list]
        Contain the word vector representation from glove.
        key for word, value for the vector.
    theta: Dict[str, np.ndarray]
        Contain the label vector weights.
        key for word, value for the vector.

    Returns
    -------
    Dict[str, np.ndarray].

    """
    labels = utils.util.get_labels(word_label)
    denom = denom_gradient(word_label, word_vector, theta)
    n_observation = len(word_label)
    gradient = utils.util.gen_grandient(theta)
    for word, i_label in word_label.items():

        grad = (1 / n_observation) * (
            np.array(word_vector[word])
            - (np.exp(np.dot(theta[i_label], word_vector[word])) / denom[word])
            * np.array(word_vector[word])
        )
        gradient[i_label] = gradient[i_label] - grad

        # gradient when W_i not in numerator
        labels_nonumer = [label for label in labels if label != i_label]
        for label in labels_nonumer:
            gradient[label] = gradient[label] + (1 / n_observation) * (
                np.exp(np.dot(theta[label], word_vector[word])) / denom[word]
            ) * np.array(word_vector[word])
    # {key: (1 / n_observation ) * value for key, value in theta.items()}
    return gradient


def update_theta(
    theta: Dict[str, np.ndarray], gradient: Dict[str, np.ndarray], learning_rate: float
) -> Dict[str, np.ndarray]:
    """


    Parameters
    ----------
    theta: Dict[str, np.ndarray]
        Contain the label vector weights.
        key for word, value for the vector.
    gradient : Dict[str, np.ndarray]
        DESCRIPTION.
    learning_rate : float
        DESCRIPTION.

    Returns
    -------
    Dict[str, np.ndarray].

    """

    return {
        key: (value - learning_rate * gradient[key]) for key, value in theta.items()
    }
