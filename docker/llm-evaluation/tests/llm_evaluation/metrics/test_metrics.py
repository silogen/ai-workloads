import pytest
from llm_evaluation.metrics.metrics import (
    compute_bertscore,
    compute_bleu_score,
    compute_exact_match,
)


def test_compute_bertscore():

    predictions = ["This is a test prediction.", "Another prediction."]
    references = ["This is a test reference.", "Another reference."]
    language = "en"

    precision, recall, f1, f1_list = compute_bertscore(predictions, references, language)

    assert precision == 0.6775
    assert recall == 0.6593
    assert f1 == 0.6689
    assert f1_list == [0.7483, 0.5895]


def test_compute_exact_match():
    predictions = ["This is a test prediction.", "Another prediction."]
    references = ["This is a test prediction.", "Another reference."]
    ignore_case = True
    ignore_punctuation = True

    accuracy = compute_exact_match(predictions, references, ignore_case, ignore_punctuation)

    assert accuracy == 0.5


def test_compute_bleu_score():

    predictions = ["This is a test prediction.", "Another prediction."]
    references = [["This is a test prediction."], ["Another prediction."]]

    bleu_score = compute_bleu_score(predictions, references)

    assert bleu_score == 1.0

    predictions = ["This is a test prediction.", "Another prediction."]
    references = [["This is a test reference."], ["Another reference."]]

    bleu_score = compute_bleu_score(predictions, references)

    assert bleu_score < 1.0
