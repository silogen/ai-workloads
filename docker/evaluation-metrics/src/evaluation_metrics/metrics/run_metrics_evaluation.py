import glob
import json
import os
import time
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

    bert_score_start_time = time.time()

    precision_bert, recall_bert, f1_bert, f1_list = compute_bertscore(predictions=predictions, references=references)

    logger.info(f"BERT-score computation took {time.time() - bert_score_start_time:.2f} seconds")

    bleu_score_start_time = time.time()

    bleu_score = compute_bleu_score(predictions=predictions, references=references)

    logger.info(f"BLEU score computation took {time.time() - bleu_score_start_time:.2f} seconds")

    exact_match_start_time = time.time()

    accuracy = compute_exact_match(predictions=predictions, references=references)

    logger.info(f"Exact match computation took {time.time() - exact_match_start_time:.2f} seconds")

    return EvaluationScores(
        precision_bert=precision_bert,
        recall_bert=recall_bert,
        f1_bert=f1_bert,
        f1_list=f1_list,
        bleu_score=bleu_score,
        accuracy=accuracy,
    )


def read_inference_data(input_path: str) -> List[Dict[str, Any]]:
    """
    Reads inference data from a file or directory containing JSON/JSONL files.

    Args:
        input_path (str): Path to either a JSONL file or a directory containing JSON/JSONL files.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents a data point.

    Raises:
        jsonlines.InvalidLineError: If a line in a file is invalid and cannot be parsed as a dictionary.
        FileNotFoundError: If the specified file or directory doesn't exist.
    """

    logger.info(f"Reading inference data from {input_path}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    generations = list[Dict[str, Any]]()
    if os.path.isdir(input_path):
        # If input_dir_path is a directory, glob all .json and .jsonl files
        json_files = glob.glob(os.path.join(input_path, "*.json"))
        jsonl_files = glob.glob(os.path.join(input_path, "*.jsonl"))
        all_files = json_files + jsonl_files

        if not all_files:
            logger.warning(f"No JSON or JSONL files found in {input_path}")
            return generations

        # Consider adding specific handling for .json vs .jsonl files
        for file_path in all_files:
            try:
                if file_path.endswith(".json"):
                    # Handle regular JSON files
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            generations.extend(data)
                        else:
                            generations.append(data)
                elif file_path.endswith(".jsonl"):
                    # Handle JSONL files
                    with jsonlines.open(file_path) as reader:
                        for line in reader.iter(type=dict, skip_invalid=True):
                            generations.append(line)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON file: {file_path}")

        logger.info(f"Read {len(generations)} generations from {len(all_files)} files.")

        return generations

    # If input_dir_path is a file, proceed with existing logic
    try:
        if input_path.endswith(".json"):
            # Handle regular JSON files
            with open(input_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    generations.extend(data)
                else:
                    generations.append(data)
        elif input_path.endswith(".jsonl"):
            # Handle JSONL files
            with jsonlines.open(input_path) as reader:
                for line in reader.iter(type=dict, skip_invalid=True):
                    generations.append(line)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON file: {input_path}")

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

    logger.info("Computing evaluation scores...")

    start_score_computation = time.time()

    scores = compute_scores(predictions=predictions, references=references)

    logger.info(f"Score computation took {time.time() - start_score_computation:.2f} seconds")

    prompts = [datapoint["prompt"] for datapoint in generations]

    return EvaluationResults(scores=scores, prompts=prompts)


def main(args: Namespace):
    """
    Main function to execute the metrics evaluation pipeline.

    Args:
        args (Namespace): A namespace object containing the following attributes:
            - input_file_path (str): Path to a JSONL file or directory containing JSON/JSONL files
              with model generations.
            - output_dir_path (str): Directory path to save the evaluation results.

    Workflow:
        1. Reads inference data from the input file or directory.
        2. Computes evaluation metrics for the generations.
        3. Saves the evaluation results to the specified output directory.
    """

    generations = read_inference_data(input_path=args.input_file_path)

    logger.info("Running metrics evaluation...")
    results = run(generations=generations)

    logger.info("Saving evaluation results...")
    results_dir_path = os.path.join(args.output_dir_path, "evaluation_results")

    if not os.path.exists(results_dir_path):
        os.makedirs(results_dir_path)

    save_results(results=results, config=args, results_dir_path=results_dir_path)
    logger.info(f"Results saved to {results_dir_path}")


if __name__ == "__main__":
    parser = get_metrics_parser()
    args = parser.parse_args()
    main(args=args)
