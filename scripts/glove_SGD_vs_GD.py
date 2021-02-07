# -*- coding: utf-8 -*-
"""
This script is to compare gradient descent and stochastic gradient decent in terms
of time and cost optimization, finally save a graph where axis x represents the time
in minutes and axis y represents the value in cost function.
"""
import time
import os
import json

import numpy as np
import matplotlib.pyplot as plt

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
co_occurrence = files[1]
theta_SGD = np.array(files[2])
theta_GD = np.array(files[2])

learning_rate = 0.0008

print(f"optimizing theta ... with a learning rate = {learning_rate}")
hist_cost_SGD = [
    glove.cost_function.cost_glove_dict(vocabulary, theta_SGD, co_occurrence)
]
minutes_SGD = [0]
acum = 0
for i in range(30):
    print(f"Iteration n°: {i}")
    inicio = time.time()
    gradient_SGD = glove.gradient.stochastic_gradient_descent(
        vocabulary, theta_SGD, co_occurrence, sample_rate=0.4
    )
    acum += (time.time() - inicio) / 60
    minutes_SGD.append(acum)

    theta_SGD = theta_SGD - learning_rate * gradient_SGD

    cost_model = glove.cost_function.cost_glove_dict(
        vocabulary, theta_SGD, co_occurrence
    )
    hist_cost_SGD.append(cost_model)


learning_rate = 0.0008
print(f"optimizing theta ... with a learning rate = {learning_rate}")
hist_cost_GD = [
    glove.cost_function.cost_glove_dict(vocabulary, theta_GD, co_occurrence)
]
minutes_GD = [0]
acum = 0
for i in range(12):
    print(f"Iteration n°: {i}")
    inicio = time.time()
    gradient_GD = glove.gradient.gradient_descent_dict(
        vocabulary, theta_GD, co_occurrence
    )
    acum += (time.time() - inicio) / 60
    minutes_GD.append(acum)

    theta_GD = theta_GD - learning_rate * gradient_GD

    cost_model = glove.cost_function.cost_glove_dict(
        vocabulary, theta_GD, co_occurrence
    )
    hist_cost_GD.append(cost_model)

plt.plot(minutes_SGD, hist_cost_SGD, label="SGD")
plt.plot(minutes_GD, hist_cost_GD, label="GD")
plt.title("Learning - SGD vs GD")
plt.xlabel("Minutes")
plt.ylabel("Cost")
plt.legend()
plt.savefig("SGD_GD.png")
plt.show()
