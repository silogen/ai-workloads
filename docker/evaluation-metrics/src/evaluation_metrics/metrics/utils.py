import json
import os

from evaluation_metrics import logger
from evaluation_metrics.metrics.data.data_classes import EvaluationResults
from minio import Minio, S3Error
from numpy import ndarray


def convert_negatives_to_zero(array: ndarray) -> ndarray:
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


def save_results(results: EvaluationResults, results_filepath: str):
    """
    Save evaluation results locally and upload them to MinIO storage.
    Args:
        results (EvaluationResults): The evaluation results to be saved.
        results_filepath (str): The local file path where the results will be saved.
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

    logger.info("Writing evaluation results locally to %s", results_filepath)
    with open(results_filepath, "w") as outfile:
        outfile.write(json.dumps(results.to_dict()))  # type: ignore
    logger.info("Results written locally")

    logger.info("Copying results file to MinIO...")
    write_to_minio_storage(
        client=client,
        source_file=results_filepath,
        destination_file="llm-evaluation-metrics/results.json",
        bucket_name=os.environ["BUCKET_STORAGE_BUCKET"],
    )
    logger.info("Results saved to MinIO")
