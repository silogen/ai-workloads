import asyncio
import os
from argparse import Namespace
from datetime import datetime
from typing import Dict, List

from llm_evaluation import logger
from llm_evaluation.argument_parsers import get_judge_inference_parser
from llm_evaluation.call_inference_container.call_inference_container import (
    download_dataset,
    get_llm_client,
    read_prompt_template,
)
from llm_evaluation.call_inference_container.call_inference_container import run as run_call_inference_container
from llm_evaluation.call_inference_container.call_inference_container import (
    save_inference_results,
)
from llm_evaluation.data.data_classes import AggregatedJudgeResults, JudgeResult
from llm_evaluation.judge.run_judge_evaluation import run_2step_judge_on_inferences
from llm_evaluation.metrics.run_metrics_evaluation import read_inference_data
from llm_evaluation.metrics.utils import get_score_distribution_graphs, log_metrics_in_mlflow, save_results


def get_judge_score_distribution_graphs(scores: List[JudgeResult]) -> Dict[str, str]:
    """
    Generates distribution graphs for judge scores.
    Args:
        scores (List[JudgeResult]): List of JudgeResult objects containing judge scores.
    Returns:
        Dict[str, str]: A dictionary containing paths to the generated distribution graphs.
    """
    metrics = {
        "judge_scores": [r.judge_grade for r in scores],
    }

    return get_score_distribution_graphs(metrics)


async def main(args: Namespace):
    """
    Main function to orchestrate evaluate performance of an llm on a given dataset using an LLM-as-a-Judge.

    This function performs the following steps:
    1. Downloads the specified evaluation dataset.
    2. Runs inference on the dataset using the evaluated model and saves the results.
    3. Loads the generated inferences and evaluates them using a two-step judge model.
    4. Aggregates and computes evaluation metrics based on judge results.
    5. Saves the aggregated evaluation results to MinIO.

    Args:
        args (Namespace): Command-line arguments containing configuration for the evaluation process.
            - model_name (str): Name of the model to use for inference.
            - model_path (str): Path to the model checkpoint or configuration.
            - judge_model_name (str): Name of the judge model.
            - judge_model_path (str): Path to the judge model checkpoint or configuration.
            - llm_base_url (str): Base URL of the LLM inference service.
            - llm_port (int): Port of the LLM inference service.
            - llm_endpoint (str): Endpoint of the LLM inference service.
            - judge_base_url (str): Base URL of the judge model service.
            - judge_port (int): Port of the judge model service.
            - judge_endpoint (str): Endpoint of the judge model service.
            - prompt_template_path (str): Path to the prompt template file.
            - judge_prompt1_template_path (str): Path to the first step judge prompt template.
            - judge_prompt2_template_path (str): Path to the second step judge prompt template.
            - evaluation_dataset_name (str): Name of the evaluation dataset.
            - evaluation_dataset_version (str): Version of the evaluation dataset.
            - dataset_split (str): Dataset split to use (e.g., "train", "test").
            - output_dir_path (str): Path to the directory where results will be saved.
            - maximum_context_size (int): Maximum context size for the model input.
            - batch_size (int): Batch size for inference.
            - judge_maximum_context_size (int): Maximum context size for the judge model input.
            - judge_batch_size (int): Batch size for judge evaluation.
            - context_column_name (str): Name of the column containing context in the dataset.
            - id_column_name (str): Name of the column containing document IDs in the dataset.
            - gold_standard_column_name (str): Name of the column containing gold standard answers in the dataset.
            - use_data_subset (int): Number of documents to use from the dataset (0 for full dataset).

    Outputs:
            - Inference results are saved to separate files in the specified output directory.
            - Judge Evaluation results and inferences are saved to a file and copied to MinIO.
    """

    # inference_model_path = args.model_path.removeprefix("s3://").removeprefix("hf://")
    # Currently unused but here for future use
    # judge_model_path = args.judge_model_path.removeprefix("s3://").removeprefix("hf://")

    ds = download_dataset(dataset=args.evaluation_dataset_name, version=args.evaluation_dataset_version)

    logger.info("Dataset loaded...")
    if args.use_data_subset > 0:
        logger.warning(f"Using a subset of the data: {args.use_data_subset} documents.")

    prompt_template = read_prompt_template(args.prompt_template_path)
    logger.info(f"Prompt template has been read in from {args.prompt_template_path}")

    results_dir_path = os.path.join(
        args.output_dir_path,
        f"inferences_{args.model_name}--{args.evaluation_dataset_name.replace('/', '_')}--{args.evaluation_dataset_version}--{datetime.now().isoformat()}",
        "inference_results",
    )
    logger.info(f"Results will be saved to {results_dir_path}")
    if not os.path.exists(results_dir_path):
        logger.info(f"Creating path {results_dir_path}")
        os.makedirs(results_dir_path)

    saved_results = []
    parameters: dict = {}

    client = get_llm_client(base_url=args.llm_base_url, port=args.llm_port, endpoint=args.llm_endpoint)

    total_candidate_inferences = args.use_data_subset if args.use_data_subset > 0 else len(ds[args.dataset_split])
    inferenced_docs = 0
    async for inference_result in run_call_inference_container(
        dataset=ds,
        prompt_template=prompt_template,
        context_column_name=args.context_column_name,
        id_column_name=args.id_column_name,
        gold_standard_column_name=args.gold_standard_column_name,
        llm_client=client,
        model_name=args.model_name,
        model_path=args.local_model_dir_path,
        parameters=parameters,
        max_context_size=args.maximum_context_size,
        batch_size=args.batch_size,
        use_data_subset=args.use_data_subset,
        dataset_split=args.dataset_split,
    ):
        result_path = save_inference_results(
            result=inference_result,
            output_dir_path=results_dir_path,
        )
        saved_results.append(result_path)
        inferenced_docs += 1
        logger.info(
            f"Saved inference result for document {inferenced_docs}/{total_candidate_inferences} with id {inference_result['doc_id']}"
        )

    logger.info(f"Loading file with generated inferences: {results_dir_path}")
    inferences_data = read_inference_data(results_dir_path)
    logger.info("Data loaded, running metrics evaluation...")

    logger.info(inferences_data)
    logger.info("Inference ran.")

    judge_client = get_llm_client(base_url=args.judge_base_url, port=args.judge_port, endpoint=args.judge_endpoint)

    aggregated_judge_results = AggregatedJudgeResults(
        judge_results={},
        average_grade=0.0,
        total_candidate_judgments=len(inferences_data),
        prompt_template_step_1=args.judge_prompt1_template_path,
        prompt_template_step_2=args.judge_prompt2_template_path,
        evaluation_dataset_name=args.evaluation_dataset_name,
        evaluation_dataset_version=args.evaluation_dataset_version,
        llm_name=args.model_name,
        judge_name=args.judge_model_name,
    )

    total_inferences = len(inferences_data)
    judged_docs = 0
    async for judge_result in run_2step_judge_on_inferences(
        inferences_data=inferences_data,
        judge_model_name=args.judge_model_name,
        prompt_template_step1_path=args.judge_prompt1_template_path,
        prompt_template_step2_path=args.judge_prompt2_template_path,
        judge_client=judge_client,
        batch_size=args.judge_batch_size,
        output_dir_path=args.output_dir_path,
    ):
        aggregated_judge_results.judge_results[judge_result.context_document_id] = judge_result
        judged_docs += 1
        logger.info(
            f"Processed judge result {judged_docs}/{total_inferences} for document {judge_result.context_document_id}"
        )

    distribution_graphs = get_judge_score_distribution_graphs(
        scores=list(aggregated_judge_results.judge_results.values())
    )

    aggregated_judge_results.average_grade = aggregated_judge_results.get_scores_dict()["mean_grade"]

    if args.mlflow_server_uri:
        logger.info("Logging results to MLFlow...")
        log_metrics_in_mlflow(
            distribution_graphs,
            aggregated_judge_results.get_scores_dict(),
            mlflow_server_uri=args.mlflow_server_uri,
            mlflow_experiment_name=args.mlflow_experiment_name,
            mlflow_run_name=args.mlflow_run_name,
            mlflow_experiment_description="Evaluation of LLM using Judge Model",
        )

    logger.info("Aggregating judge scores...")
    logger.info(aggregated_judge_results)

    save_results(results=aggregated_judge_results, results_dir_path=results_dir_path, config=args)
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    parser = get_judge_inference_parser()
    args = parser.parse_args()
    asyncio.run(main(args))
