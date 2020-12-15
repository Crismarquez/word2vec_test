# -*- coding: utf-8 -*-
"""
Test util functions
"""
import utils.util


# Test gen_vocabulary()
def test_size_gen_vocabulary():
    """
    Test the size for vocabulary
    """
    corpus = ["test", "for", "gen_vocabulary", "test", "corpus"]
    vocabulary = utils.util.gen_vocabulary(corpus)

    assert len(vocabulary) == 4


def test_unique_gen_vocabulary():
    """
    Test unique word in vocabulary
    """
    corpus = ["test", "for", "gen_vocabulary", "test", "corpus"]
    vocabulary = utils.util.gen_vocabulary(corpus)

    assert vocabulary.count("test") == 1


# Test gen_theta()
def test_size_gen_theta():
    """
    Test the size for vocabulary
    """
    vocabulary = ["test", "for", "gen_vocabulary", "corpus"]
    dimension = 300
    theta = utils.util.gen_theta(vocabulary, dimension)

    assert len(theta) == 2 * len(vocabulary) * dimension
