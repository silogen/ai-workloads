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
    save_local_results,
)
from llm_evaluation.data.data_classes import AggregatedJudgeResults, JudgeResult
from llm_evaluation.judge.run_judge_evaluation import run_2step_judge_on_inferences
from llm_evaluation.metrics.run_metrics_evaluation import read_local_inference_data
from llm_evaluation.metrics.utils import (
    copy_results_files_to_minio,
    get_score_distribution_graphs,
    log_metrics_in_mlflow,
    save_json_object_to_minio,
)
from minio import Minio


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
            - local_model_dir_path (str): Local directory path where the model is stored.
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
            - minio_output_dir_path (str): Path to the directory where results will be saved.
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

    ds = download_dataset(dataset=args.evaluation_dataset_name, version=args.evaluation_dataset_version)

    logger.info("Dataset loaded...")
    if args.use_data_subset > 0:
        logger.warning(f"Using a subset of the data: {args.use_data_subset} documents.")

    prompt_template = read_prompt_template(args.prompt_template_path)
    logger.info(f"Prompt template has been read in from {args.prompt_template_path}")

    evaluation_dir_name = f"judge--{args.model_name}--{args.evaluation_dataset_name.replace('/', '_')}--{args.evaluation_dataset_version}--{datetime.now().isoformat()}"
    minio_results_dir_path = os.path.join(args.minio_output_dir_path, evaluation_dir_name)
    local_results_dir_path = os.path.join("/home/evaluation", evaluation_dir_name)
    if not os.path.exists(local_results_dir_path):
        logger.info(f"Creating path {local_results_dir_path}")
        os.makedirs(local_results_dir_path)

    logger.info(f"Results will be saved locally to {local_results_dir_path}")

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
        save_local_results(
            result=inference_result,
            output_dir_path=local_results_dir_path,
            subdir="inferences",
        )
        inferenced_docs += 1
        logger.info(
            f"Saved inference result for document {inferenced_docs}/{total_candidate_inferences} with id {inference_result['context_document_id']}"
        )

    logger.info(f"Loading file with generated inferences: {local_results_dir_path}")
    inferences_data = read_local_inference_data(os.path.join(local_results_dir_path, "inferences"))
    logger.info("Data loaded, running judge evaluation...")

    logger.info(inferences_data)
    logger.info("Inference ran.")

    judge_client = get_llm_client(base_url=args.judge_base_url, port=args.judge_port, endpoint=args.judge_endpoint)

    judge_prompt_step1_template = read_prompt_template(args.judge_prompt1_template_path)
    judge_prompt_step2_template = read_prompt_template(args.judge_prompt2_template_path)

    aggregated_judge_results = AggregatedJudgeResults(
        judge_results={},
        average_grade=0.0,
        total_candidate_judgments=len(inferences_data),
        judge_prompt_step1_template=judge_prompt_step1_template,
        judge_prompt_step2_template=judge_prompt_step2_template,
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
        judge_prompt_step1_template=judge_prompt_step1_template,
        judge_prompt_step2_template=judge_prompt_step2_template,
        judge_client=judge_client,
        batch_size=args.judge_batch_size,
        output_dir_path=local_results_dir_path,
    ):
        aggregated_judge_results.judge_results[judge_result.context_document_id] = judge_result
        judged_docs += 1
        logger.info(
            f"Processed judge result {judged_docs}/{total_inferences} for document {judge_result.context_document_id}"
        )

        save_local_results(
            result=judge_result.to_dict(),  # type: ignore
            output_dir_path=local_results_dir_path,
            subdir="judge_results",
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

    logger.info("Writing evaluation results to MinIO...")
    minio_client = Minio(
        endpoint=os.environ["BUCKET_STORAGE_HOST"],
        access_key=os.environ["BUCKET_STORAGE_ACCESS_KEY"],
        secret_key=os.environ["BUCKET_STORAGE_SECRET_KEY"],
        secure=False,
        cert_check=False,
    )
    for json_object, destination_file in [
        (aggregated_judge_results.get_summary_dict(), os.path.join(minio_results_dir_path, "summary_results.json")),  # type: ignore
        (aggregated_judge_results.judge_prompt_step1_template, os.path.join(minio_results_dir_path, "step1_template.json")),  # type: ignore
        (aggregated_judge_results.judge_prompt_step2_template, os.path.join(minio_results_dir_path, "step2_template.json")),  # type: ignore
        (vars(args), os.path.join(minio_results_dir_path, "config.json")),
    ]:
        save_json_object_to_minio(json_object=json_object, destination_file=destination_file, client=minio_client)

    # Copying model inference results to MinIO
    copy_results_files_to_minio(
        local_results_dir_path=local_results_dir_path,
        subdir="inferences",
        minio_results_dir_path=minio_results_dir_path,
        minio_client=minio_client,
    )

    # Copying judge inference results to MinIO
    copy_results_files_to_minio(
        local_results_dir_path=local_results_dir_path,
        subdir="judge_results",
        minio_results_dir_path=minio_results_dir_path,
        minio_client=minio_client,
    )

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    parser = get_judge_inference_parser()
    args = parser.parse_args()
    asyncio.run(main(args))
