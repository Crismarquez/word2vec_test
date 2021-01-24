# -*- coding: utf-8 -*-
"""
This script run the glove model using previos inputs created by get_inputs_for_glove.py
The main objective of the script is update theta and save this update to continue
using in the future, also save a csv file with vector representation for context words.
In order to check the convergence of cost model, each three iteration the cost
will be calculated to then show in a graph.
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt

import utils.util
import glove.cost_function
import glove.gradient


# import inputs
path = "files/"
files_name = ['vocabulary', 'co_occurrence', 'theta']
files = []
for file_name in files_name:
    with open(path + file_name + '.pkl', 'rb') as f:
        files.append(pickle.load(f))

vocabulary = files[0]
co_occurrence = files[1]
theta = files[2]

print("Size of filtered vocabulary: ", "{:,.0f}".format(len(vocabulary)))
print("Size of theta: ", "{:,.0f}".format(len(theta)))

dimension = len(theta) // 2 // len(vocabulary)

learinig_rate = 0.0005
print(f"optimizing theta ... with a learning rate = {learinig_rate}")
hist_cost = [glove.cost_function.cost_glove_dict(
        vocabulary, theta, co_occurrence
    )]

for i in range(12):
    print(f'Iteration n°: {i}')
    gradient = glove.gradient.gradient_descent_dict(
        vocabulary, theta, co_occurrence
    )

    max_grad = gradient.max()
    print('max value in gradient: ', max_grad)
    print('max change in theta: ', max_grad * learinig_rate)

    theta = theta - learinig_rate * gradient

    # update file
    with open(path + 'theta' + '.pkl', 'wb') as f:
        pickle.dump(theta, f, pickle.HIGHEST_PROTOCOL)

    if i % 3 ==0:
        cost_model = glove.cost_function.cost_glove_dict(
            vocabulary, theta, co_occurrence
        )
        hist_cost.append(cost_model)

plt.plot(range(len(hist_cost)), hist_cost)
plt.title("Learning")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()

# save data
data_context = {}
for context_word in vocabulary:
    context_index = vocabulary.index(context_word)
    context_vector = utils.util.find_vector(
        context_index, theta, dimension, central=False
    )
    data_context[context_word] = context_vector

df = pd.DataFrame(data_context)
df = df.T
df.to_csv(path + "glove_English_V1.csv"
)
