from typing import List, Tuple

import numpy as np
from evaluate import load
from evaluation_metrics.metrics.utils import convert_negatives_to_zero


def compute_bertscore(
    predictions: List[str], references: List[str], language: str = "en"
) -> Tuple[float, float, float, List[float]]:
    """
    Computes the BERTScore for a set of predictions and references.

    Args:
        predictions (List[str]): A list of predicted text strings.
        references (List[str]): A list of reference text strings.
        language (str, optional): The language of the text. Defaults to "en".

    Returns:
        Tuple[float, float, float, List[float]]: A tuple containing:
            - precision_bert (float): The average precision score.
            - recall_bert (float): The average recall score.
            - f1_bert (float): The average F1 score.
            - f1_list (List[float]): A list of F1 scores for each prediction-reference pair.
    """
    bertscore = load("bertscore")
    results = bertscore.compute(
        predictions=predictions, references=references, lang=language, rescale_with_baseline=True, device="cpu"
    )

    precision_list = convert_negatives_to_zero(array=np.array(results["precision"]))
    recall_list = convert_negatives_to_zero(array=np.array(results["recall"]))
    f1_list = convert_negatives_to_zero(array=np.array(results["f1"]))

    precision_bert = round(np.average(precision_list), 4)
    recall_bert = round(np.average(recall_list), 4)
    f1_bert = round(np.average(f1_list), 4)

    f1_list = [round(f1, 4) for f1 in f1_list]

    return precision_bert, recall_bert, f1_bert, f1_list


def compute_exact_match(
    predictions: List[str], references: List[str], ignore_case: bool = True, ignore_punctuation: bool = True
) -> float:
    """
    Computes the exact match accuracy between predictions and references.

    Args:
        predictions (List[str]): A list of predicted strings.
        references (List[str]): A list of reference strings to compare against.
        ignore_case (bool, optional): Whether to ignore case when comparing strings. Defaults to True.
        ignore_punctuation (bool, optional): Whether to ignore punctuation when comparing strings. Defaults to True.

    Returns:
        float: The exact match accuracy as a percentage (0.0 to 100.0).
    """
    exact_match_metric = load("exact_match")
    results = exact_match_metric.compute(
        predictions=predictions, references=references, ignore_case=ignore_case, ignore_punctuation=ignore_punctuation
    )
    accuracy = results["exact_match"]

    return accuracy


def compute_bleu_score(predictions: List[str], references: List[str]) -> float:
    """
    Computes the BLEU (Bilingual Evaluation Understudy) score for a set of predictions
    against a set of reference translations.

    Args:
        predictions (List[str]): A list of predicted translations.
        references (List[str]): A list of reference translations corresponding to the predictions.

    Returns:
        float: The computed BLEU score, a value between 0 and 1, where higher values indicate
               closer matches between predictions and references.
    """

    bleu = load("bleu")
    results = bleu.compute(predictions=predictions, references=references)
    bleu_score = results["bleu"]

    return bleu_score
