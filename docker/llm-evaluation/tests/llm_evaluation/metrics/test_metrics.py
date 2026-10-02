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

    precision_list, recall_list, f1_list = compute_bertscore(predictions, references, language)

    assert precision_list.tolist() == [0.7478331923484802, 0.6070768237113953]
    assert recall_list.tolist() == [0.747850775718689, 0.5707229375839233]
    assert f1_list.tolist() == [0.7482516765594482, 0.5895071625709534]


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
