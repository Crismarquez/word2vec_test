# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 20:46:53 2021

@author: Cristian Marquez
"""
import os
import json

import numpy as np
import matplotlib.pyplot as plt

import utils.util
import glove.cost_function
import glove.gradient


# import inputs
base = "files"
files_name = ["vocabulary", "co_occurrence", "theta"]
files = []
for file_name in files_name:
    file_path = os.path.join(base, file_name + ".json")
    with open(file_path, "r") as f:
        files.append(json.load(f)[file_name])

vocabulary = files[0]
co_occurrence = utils.util.keystr_to_keytuple(files[1])
theta_SGD = np.array(files[2])
theta_GD = np.array(files[2])

learning_rates = [0.05, 0.005, 0.0005]
for learning_rate in learning_rates:
    theta_SGD = np.array(files[2])
    theta_GD = np.array(files[2])
    print(f"optimizing theta ... with a learning rate = {learning_rate}")
    hist_cost_SGD = [
        glove.cost_function.cost_glove_dict(vocabulary, theta_SGD, co_occurrence)
    ]
    hist_cost_GD = [
        glove.cost_function.cost_glove_dict(vocabulary, theta_GD, co_occurrence)
    ]
    for i in range(13):
        print(f"Iteration n°: {i}")
        gradient_SGD = glove.gradient.stocastic_gradient_descent(
            vocabulary, theta_SGD, co_occurrence, sample_rate=0.5
        )
        gradient_GD = glove.gradient.gradient_descent_dict(
            vocabulary, theta_GD, co_occurrence
        )

        theta_SGD = theta_SGD - learning_rate * gradient_SGD
        theta_GD = theta_GD - learning_rate * gradient_GD

        if i % 3 == 0:
            cost_model = glove.cost_function.cost_glove_dict(
                vocabulary, theta_SGD, co_occurrence
            )
            hist_cost_SGD.append(cost_model)

            cost_model = glove.cost_function.cost_glove_dict(
                vocabulary, theta_GD, co_occurrence
            )
            hist_cost_GD.append(cost_model)
    plt.plot(range(len(hist_cost_SGD)), hist_cost_SGD, label="SGD")
    plt.plot(range(len(hist_cost_GD)), hist_cost_GD, label="GD")
    plt.title("Learning")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()

learning_rates = [0.05, 0.005]
for learning_rate in learning_rates:
    theta_SGD = np.array(files[2])
    print(f"optimizing theta ... with a learning rate = {learning_rate}")
    hist_cost_SGD = [
        glove.cost_function.cost_glove_dict(vocabulary, theta_SGD, co_occurrence)
    ]
    for i in range(13):
        print(f"Iteration n°: {i}")
        gradient_SGD = glove.gradient.stocastic_gradient_descent(
            vocabulary, theta_SGD, co_occurrence, sample_rate=0.5
        )

        theta_SGD = theta_SGD - learning_rate * gradient_SGD

        if i % 3 == 0:
            cost_model = glove.cost_function.cost_glove_dict(
                vocabulary, theta_SGD, co_occurrence
            )
            hist_cost_SGD.append(cost_model)

    plt.plot(range(len(hist_cost_SGD)), hist_cost_SGD, label="SGD")
    plt.title("Learning")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()
