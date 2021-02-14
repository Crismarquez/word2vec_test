# -*- coding: utf-8 -*-
"""
Funtions to predict using optimized theta
"""
from typing import Dict, List
import numpy as np


def predict_classwords(
    word_label: Dict[str, str],
    word_vector: Dict[str, List[float]],
    theta: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """


    Parameters
    ----------
    word_label : Dict[str, str]
        DESCRIPTION.
    word_vector : Dict[str, List[float]]
        DESCRIPTION.
    theta : Dict[str, np.ndarray]
        DESCRIPTION.

    Returns
    -------
    Dict[str, float]
        DESCRIPTION.

    """

    prediction = {}
    for word in word_label.keys():
        prediction[word] = {
            key: np.dot(word_vector[word], theta_val)
            for key, theta_val in theta.items()
        }

    return prediction
