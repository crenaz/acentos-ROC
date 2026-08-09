# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 crenaz

import pytest

from acentos_ocr.eval.metrics import cer, levenshtein, normalise, word_miss_rate


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("# Heading", "heading"),
        ("## Job Title: Cook", "job title: cook"),
        ("* A bullet", "a bullet"),
        ("· A bullet", "a bullet"),  # U+00B7, as pasted from some transcriptions
        ("- A bullet", "a bullet"),
        ("line one\n\n   line two", "line one line two"),
        ("**Job Description:** Responsible for", "job description: responsible for"),
        ("__bold__ text", "bold text"),
        ("**Requirements:**", "requirements:"),
        # emphasis is removed before bullets, so no stray marker survives
        ("* **Must** have experience", "must have experience"),
    ],
)
def test_normalise_strips_markdown_scaffolding(raw, expected):
    assert normalise(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('# MAIN LOGO: "A GREAT Opportunity"', '"a great opportunity"'),
        ("BOTTOM LOGO: DART", "dart"),
        ("MAIN LOGO: Crest of the Cayman Islands", "crest of the cayman islands"),
    ],
)
def test_normalise_drops_the_annotation_label_but_keeps_its_value(raw, expected):
    """The label is the transcriber's note; the text after it is printed on the page."""
    assert normalise(raw) == expected


def test_normalise_leaves_labels_that_are_really_on_the_page():
    # TITLE:/SALARY:/ID: are printed in the adverts, unlike the LOGO annotations.
    assert normalise("## TITLE: SENIOR SPECIALIST") == "title: senior specialist"


@pytest.mark.parametrize(
    "a, b, expected",
    [("", "", 0), ("abc", "abc", 0), ("abc", "abd", 1), ("abc", "", 3), ("", "ab", 2)],
)
def test_levenshtein(a, b, expected):
    assert levenshtein(a, b) == expected


def test_levenshtein_is_symmetric():
    assert levenshtein("kitten", "sitting") == levenshtein("sitting", "kitten") == 3


def test_perfect_read_scores_zero_on_both_metrics():
    text = "we are seeking an experienced specialist"
    assert cer(text, text) == 0.0
    assert word_miss_rate(text, text) == 0.0


def test_reordering_costs_cer_but_not_word_miss_rate():
    """
    The distinction the corpus harness exists to make: identical words, wrong
    order. CER punishes it; the word metric does not, and the gap is the signal
    that segmentation rather than recognition is at fault.
    """
    truth = "alpha beta gamma delta epsilon zeta"
    shuffled = "delta epsilon zeta alpha beta gamma"
    assert word_miss_rate(truth, shuffled) == 0.0
    assert cer(truth, shuffled) > 0.5


def test_missing_words_cost_both_metrics():
    truth = "alpha beta gamma delta"
    assert word_miss_rate(truth, "alpha beta") == 0.5
    assert cer(truth, "alpha beta") > 0.0


def test_word_miss_rate_counts_repeats_as_a_multiset():
    assert word_miss_rate("cook cook cook", "cook") == pytest.approx(2 / 3)


def test_empty_hypothesis_misses_everything():
    assert word_miss_rate("alpha beta", "") == 1.0
    assert cer("alpha beta", "") == 1.0


def test_empty_truth_does_not_divide_by_zero():
    assert word_miss_rate("", "anything") == 0.0
    assert cer("", "") == 0.0
