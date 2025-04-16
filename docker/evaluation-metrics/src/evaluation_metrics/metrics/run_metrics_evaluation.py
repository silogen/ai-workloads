import os
from argparse import Namespace
from typing import Any, Dict, List

import jsonlines
from evaluation_metrics import logger
from evaluation_metrics.argument_parsers import get_metrics_parser
from evaluation_metrics.metrics.data.data_classes import EvaluationResults, EvaluationScores
from evaluation_metrics.metrics.metrics import compute_bertscore, compute_bleu_score, compute_exact_match
from evaluation_metrics.metrics.utils import save_results


def compute_scores(predictions: List[str], references: List[str]) -> EvaluationScores:
    """
    Computes evaluation metrics (BERTScore, BLEU score, and Exact Match) for the given predictions and references.

    Args:
        predictions (List[str]): A list of predicted strings.
        references (List[str]): A list of reference strings.

    Returns:
        EvaluationScores: An object containing the computed scores for precision, recall, F1 (BERTScore),
                          BLEU score, and Exact Match accuracy.
    """

    precision_bert, recall_bert, f1_bert, f1_list = compute_bertscore(predictions=predictions, references=references)
    bleu_score = compute_bleu_score(predictions=predictions, references=references)
    accuracy = compute_exact_match(predictions=predictions, references=references)

    return EvaluationScores(
        precision_bert=precision_bert,
        recall_bert=recall_bert,
        f1_bert=f1_bert,
        f1_list=f1_list,
        bleu_score=bleu_score,
        accuracy=accuracy,
    )


def read_jsonl_data(input_file_path: str) -> List[Dict[str, Any]]:
    """
    Reads a JSONL (JSON Lines) file and returns its contents as a list of dictionaries.

    Args:
        input_file_path (str): The file path to the JSONL file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents a line in the JSONL file.

    Raises:
        jsonlines.InvalidLineError: If a line in the file is invalid and cannot be parsed as a dictionary.
    """

    generations = list()
    with jsonlines.open(input_file_path) as reader:
        for line in reader.iter(type=dict, skip_invalid=True):
            generations.append(line)
    return generations


def run(generations: List[Dict[str, Any]]) -> EvaluationResults:
    """
    Evaluates the performance of a model by comparing its predictions to the gold standard results.

    Args:
        generations (List[Dict[str, Any]]): A list of dictionaries where each dictionary represents a data point.
            Each dictionary must contain the following keys:
            - "gold_standard_result" (List[Any]): A list containing the correct answers. Only single-answer lists are supported.
            - "inference_result" (Any): The model's prediction for the given data point.
            - "prompt" (Any): The input prompt used to generate the prediction.

    Returns:
        EvaluationResults: An object containing the computed evaluation scores and the prompts used.

    Raises:
        NotImplementedError: If any data point contains multiple correct answers in "gold_standard_result".
    """

    references = []

    for datapoint in generations:
        if len(datapoint["gold_standard_result"]) == 1:
            references.append(datapoint["gold_standard_result"][0])
        else:
            raise NotImplementedError("Multiple correct answers")
    predictions = [datapoint["inference_result"] for datapoint in generations]
    scores = compute_scores(predictions=predictions, references=references)
    prompts = [datapoint["prompt"] for datapoint in generations]

    return EvaluationResults(scores=scores, prompts=prompts)


def main(args: Namespace):
    """
    Main function to execute the metrics evaluation pipeline.

    Args:
        args (Namespace): A namespace object containing the following attributes:
            - input_file_path (str): Path to the JSONL file containing the model's generations.
            - output_dir_path (str): Directory path to save the evaluation results.

    Workflow:
        1. Reads the input JSONL file containing generations.
        2. Computes evaluation metrics for the generations.
        3. Saves the evaluation results to the specified output directory.
    """

    generations = read_jsonl_data(input_file_path=args.input_file_path)

    logger.info("Running metrics evaluation...")
    results = run(generations=generations)

    logger.info("Saving evaluation results...")
    results_dir_path = os.path.join(args.output_dir_path, "evaluation_results")

    save_results(results=results, config=args, results_dir_path=results_dir_path)
    logger.info(f"Results saved to {results_dir_path}")


if __name__ == "__main__":
    parser = get_metrics_parser()
    args = parser.parse_args()
    main(args=args)
