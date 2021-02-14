# -*- coding: utf-8 -*-
"""
Use a clasification model for words using vector representation from glove
"""
import os
import json

import utils.util
import classification.cost_function
import classification.gradient
import classification.predict


base = (
    "C:/Users/Cristian Marquez/Documents/Cristian/Academico/Projects/NLP/word2vec_V2/"
)
file_name = "glove_300d"
file_path = os.path.join(base, file_name + ".json")

with open(file_path, "r") as f:
    word_vector = json.load(f)

word_label = {
    "i": "human",
    "person": "human",
    "man": "human",
    "women": "human",
    "cat": "animal",
    "dog": "animal",
    "bird": "animal",
    "lion": "animal",
    "potato": "vegetables",
    "tomatoes": "vegetables",
    "garlic": "vegetables",
    "onions": "vegetables",
}

word_vector = {
    "i": [0.1, 0.2, 0.3],
    "person": [0.5, 0.52, 0.52],
    "man": [0.1, 0.22, 0.32],
    "women": [0.12, 0.24, 0.34],
    "cat": [0.32, 0.34, 0.44],
    "dog": [0.36, 0.36, 0.46],
    "bird": [0.15, 0.14, 0.12],
    "lion": [0.35, 0.34, 0.32],
    "potato": [0.16, 0.15, 0.14],
    "tomatoes": [0.17, 0.14, 0.13],
    "garlic": [0.15, 0.14, 0.12],
    "onions": [0.15, 0.14, 0.12],
}

labels = utils.util.get_labels(word_label)
dimension = len(list(word_label.values())[0])

theta = utils.util.gen_theta_class_words(labels, dimension)
gradient = utils.util.gen_grandient(theta)

learning_rate = 0.005

cost = classification.cost_function.cost_classification_words(
    word_label, word_vector, theta
)
gradient = classification.gradient.gradient_classification_words(
    word_label, word_vector, theta
)
