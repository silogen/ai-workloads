import json
import os
from argparse import Namespace
from typing import Any, Dict, List

import jsonlines
import mlflow
import numpy as np
from llm_evaluation import logger
from llm_evaluation.data.data_classes import AggregatedJudgeResults, EvaluationResults
from minio import Minio, S3Error


def convert_negatives_to_zero(array: np.ndarray) -> np.ndarray:
    """Converts all negative values in an array to zero.

    Args:
        array: A NumPy array.

    Returns:
        A NumPy array with all negative values replaced with zero.
    """

    array[array < 0] = 0
    return array


def write_to_minio_storage(client: Minio, source_file: str, destination_file: str, bucket_name: str):
    """
    Uploads a file to a specified bucket in Minio storage.
    Args:
        client (Minio): The Minio client instance used to interact with the Minio server.
        source_file (str): The local file path of the file to be uploaded.
        destination_file (str): The destination object name in the Minio bucket.
        bucket_name (str): The name of the bucket where the file will be uploaded.
    Raises:
        S3Error: If an error occurs during the upload process.
    Logs:
        A success message if the file is uploaded successfully.
        An error message if an exception occurs during the upload.
    """

    try:
        client.fput_object(
            bucket_name=bucket_name,
            object_name=destination_file,
            file_path=source_file,
        )
        logger.info(
            "%s successfully uploaded as object %s to bucket %s",
            source_file,
            destination_file,
            bucket_name,
        )
    except S3Error as e:
        logger.error("Error occurred in Minio upload: %s", e)


def save_results(results: EvaluationResults | AggregatedJudgeResults, config: Namespace, results_dir_path: str):
    """
    Save evaluation results locally and upload them to MinIO storage.
    Args:
        results (EvaluationResults): The evaluation results to be saved.
        config (Namespace): The configuration object containing the parameters used for evaluation.
        results_dir_path (str): The local directory path where the results will be saved.
    Raises:
        KeyError: If any required environment variable for MinIO configuration
                  (BUCKET_STORAGE_HOST, BUCKET_STORAGE_ACCESS_KEY,
                  BUCKET_STORAGE_SECRET_KEY, BUCKET_STORAGE_BUCKET) is missing.
        MinioError: If there is an error during the upload to MinIO storage.
    Logs:
        - Writes the evaluation results to a local file.
        - Uploads the results file to a MinIO bucket.
    """

    client = Minio(
        endpoint=os.environ["BUCKET_STORAGE_HOST"],
        access_key=os.environ["BUCKET_STORAGE_ACCESS_KEY"],
        secret_key=os.environ["BUCKET_STORAGE_SECRET_KEY"],
        secure=False,
        cert_check=False,
    )

    evaluation_results_file_path = os.path.join(results_dir_path, "evaluation_results.json")

    config_file_path = os.path.join(results_dir_path, "config.json")

    logger.info(f"Writing evaluation results locally to {evaluation_results_file_path}")
    with open(evaluation_results_file_path, "w") as outfile:
        outfile.write(json.dumps(results.to_dict()))  # type: ignore
    logger.info("Results written locally")

    logger.info(f"Writing config locally to {config_file_path}")
    with open(config_file_path, "w") as outfile:
        outfile.write(json.dumps(vars(config)))
    logger.info("Results written locally")

    logger.info("Copying results file to MinIO...")
    write_to_minio_storage(
        client=client,
        source_file=evaluation_results_file_path,
        destination_file=os.path.join("llm-evaluation-metrics", results_dir_path, "results.json"),
        bucket_name=os.environ["BUCKET_STORAGE_BUCKET"],
    )
    logger.info("Results saved to MinIO")

    logger.info("Copying configs to MinIO...")
    write_to_minio_storage(
        client=client,
        source_file=config_file_path,
        destination_file=os.path.join("llm-evaluation-metrics", results_dir_path, "config.json"),
        bucket_name=os.environ["BUCKET_STORAGE_BUCKET"],
    )
    logger.info("Config saved to MinIO")


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


def log_metrics_in_mlflow(distribution_graphs, scores, mlflow_server_uri, mlflow_experiment_name, mlflow_run_name):

    logger.info(f"Using MLflow tracking URI: {mlflow_server_uri}")

    experiment_description = "Evaluation of LLM using BERTScore metric."

    experiment_tags = {
        "project_name": mlflow_experiment_name,
        "mlflow.note.content": experiment_description,
    }

    client = mlflow.MlflowClient(tracking_uri=mlflow_server_uri)

    # Create the Experiment, providing a unique name
    try:
        test_experiment = client.create_experiment(name=mlflow_experiment_name, tags=experiment_tags)
        logger.info(f"Created experiment with ID: {test_experiment}")
    except mlflow.exceptions.MlflowException as e:
        # If the experiment already exists, retrieve its ID
        logger.warning(f"Experiment '{mlflow_experiment_name}' already exists. Using existing experiment.")
        test_experiment = client.get_experiment_by_name(mlflow_experiment_name).experiment_id
        logger.info(f"Using existing experiment with ID: {test_experiment}")

    mlflow.set_tracking_uri(mlflow_server_uri)
    mlflow.set_experiment(experiment_name=mlflow_experiment_name)
    with mlflow.start_run(run_name=mlflow_run_name, experiment_id=test_experiment) as run:

        for name, file in distribution_graphs.items():
            mlflow.log_metric("bert_score_mean_precision" + name, np.mean(scores.precision_avg_bert))
            mlflow.log_metric("bert_score_mean_recall" + name, np.mean(scores.recall_avg_bert))
            mlflow.log_metric("bert_score_mean_f1" + name, np.mean(scores.f1_avg_bert))
            logger.info(
                f"Saving artifact {file} (abs path: {os.path.abspath(file)}) to MLflow run {run.info.run_id}..."
            )
            mlflow.log_artifact(os.path.abspath(file), artifact_path="metrics_distributions")
