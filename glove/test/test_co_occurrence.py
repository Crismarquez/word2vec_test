# -*- coding: utf-8 -*-
"""
test co_occurence
"""
import glove.co_occurrence


def test_cooccurrences_exists():
    """
    Test a cooccurrence that exist in the vocabulary given.

    Returns
    -------
    None.

    """
    corpus = [
        "i",
        "like",
        "NLP",
        ",",
        "i",
        "like",
        "machine",
        "learning",
        ",",
        "i",
        "like",
        "NLP",
        ",",
        "i",
        "like",
        "machine",
        "learning",
        ".",
    ]

    vocabulary = ["i", "like"]
    S_WINDOW = 3
    cooccurrences = glove.co_occurrence.cooccurrences(corpus, vocabulary, S_WINDOW)
    assert cooccurrences[("i", "like")] == 5


def test_cooccurrences_not_exists():
    """
    Test a cooccurrence that not exist in the vocabulary given.

    Returns
    -------
    None.

    """
    corpus = [
        "i",
        "like",
        "NLP",
        ",",
        "i",
        "like",
        "machine",
        "learning",
        ",",
        "i",
        "like",
        "NLP",
        ",",
        "i",
        "like",
        "machine",
        "learning",
        ".",
    ]

    vocabulary = ["i", "like"]
    S_WINDOW = 3
    cooccurrences = glove.co_occurrence.cooccurrences(corpus, vocabulary, S_WINDOW)
    assert cooccurrences.get(("like", "NLP")) is None
