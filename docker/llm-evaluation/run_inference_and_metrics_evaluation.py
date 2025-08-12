import asyncio
import os
from argparse import Namespace
from datetime import datetime

from llm_evaluation import logger
from llm_evaluation.argument_parsers import get_inference_parser
from llm_evaluation.call_inference_container.call_inference_container import (
    download_dataset,
    get_llm_client,
    read_prompt_template,
)
from llm_evaluation.call_inference_container.call_inference_container import run as run_call_inference_container
from llm_evaluation.call_inference_container.call_inference_container import (
    save_local_results,
)
from llm_evaluation.metrics.run_metrics_evaluation import get_bert_score_distribution_graphs, read_local_inference_data
from llm_evaluation.metrics.run_metrics_evaluation import run as run_metrics_evaluation
from llm_evaluation.metrics.utils import copy_results_files_to_minio, log_metrics_in_mlflow, save_json_object_to_minio
from minio import Minio


async def main(args: Namespace):
    """
    Main function to execute the inference and metrics evaluation pipeline.

    Args:
        args (Namespace): A namespace object containing the following attributes:
            - evaluation_dataset_name (str): The name of the evaluation dataset.
            - evaluation_dataset_version (str): The version of the evaluation dataset.
            - prompt_template_path (str): Path to the prompt template file.
            - llm_base_url (str): Base URL of the LLM service.
            - llm_port (int): Port number of the LLM service.
            - llm_endpoint (str): Endpoint of the LLM service.
            - context_column_name (str): Name of the column containing context data in the dataset.
            - gold_standard_column_name (str): Name of the column containing gold standard data in the dataset.
            - model_name (str): Name of the model to be used for inference.
            - model_path (str): Path to the model.
            - local_model_dir_path (str): Local directory path where the model is stored.
            - maximum_context_size (int): Maximum size of the context for inference.
            - batch_size (int): Batch size for inference.
            - use_data_subset (int): Number of documents to use for evaluation. If > 0, limits to this many documents.
            - dataset_split (str): The dataset split to use (e.g., "train", "test").
            - minio_output_dir_path (str): Directory path to save output files.

    Workflow:
        1. Downloads the specified dataset.
        2. Reads the prompt template from the provided file path.
        3. Initializes the LLM client with the specified configuration.
        4. Runs inference using the dataset and prompt template.
        5. Saves each inference result to a separate file as it becomes available from the async operation.
        6. Collects paths of individual inference result files for later evaluation.
        7. Loads the generated inferences and evaluates metrics.
        8. Saves the evaluation results to a file.

    Outputs:
        - Inference results are saved to separate files in the specified output directory.
        - Evaluation results are saved to a file in the specified output directory and copied to MinIO.
    Prints:
        - Status messages at each step of the pipeline.
        - Final evaluation results.
    """

    ds = download_dataset(dataset=args.evaluation_dataset_name, version=args.evaluation_dataset_version)

    logger.info("Dataset loaded...")

    prompt_template = read_prompt_template(args.prompt_template_path)

    logger.info(f"Prompt template has been read in from {args.prompt_template_path}")

    parameters: dict = {}

    client = get_llm_client(base_url=args.llm_base_url, port=args.llm_port, endpoint=args.llm_endpoint)

    if args.use_data_subset > 0:
        logger.warning(f"Using a subset of the data: {args.use_data_subset} documents.")

    evaluation_dir_name = f"metrics--{args.model_name}--{args.evaluation_dataset_name.replace('/', '_')}--{args.evaluation_dataset_version}--{datetime.now().isoformat()}"
    minio_results_dir_path = os.path.join(args.minio_output_dir_path, f"metrics_{evaluation_dir_name}")
    local_results_dir_path = os.path.join("/home/evaluation/results", f"inferences_{evaluation_dir_name}")
    if not os.path.exists(local_results_dir_path):
        logger.info(f"Creating path {local_results_dir_path}")
        os.makedirs(local_results_dir_path)

    logger.info(f"Results will be saved locally to {local_results_dir_path}")

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
        save_local_results(result=inference_result, output_dir_path=local_results_dir_path, subdir="inferences")
        inferenced_docs += 1
        logger.info(
            f"Saved inference result locally for document {inferenced_docs}/{total_candidate_inferences} with id {inference_result['context_document_id']}"
        )

    logger.info("All inferences complete. Now running metrics evaluation...")
    logger.info(f"Loading all inferences: {local_results_dir_path}")
    data = read_local_inference_data(os.path.join(local_results_dir_path, "inferences"))
    logger.info(f"Loaded {len(data)} inference results.")
    logger.info("Data loaded, running metrics evaluation...")

    eval_results = run_metrics_evaluation(data)

    distribution_graphs = get_bert_score_distribution_graphs(
        scores=eval_results.scores,
    )

    if args.mlflow_server_uri:
        logger.info("Logging results to MLFlow...")
        log_metrics_in_mlflow(
            distribution_graphs,
            eval_results.get_summary_scores_dict(),
            mlflow_server_uri=args.mlflow_server_uri,
            mlflow_experiment_name=args.mlflow_experiment_name,
            mlflow_run_name=args.mlflow_run_name,
            mlflow_experiment_description="LLM Metrics Evaluation",
        )

    logger.info("Evaluation results:")
    logger.info(eval_results)

    logger.info("Writing evaluation results to MinIO...")
    minio_client = Minio(
        endpoint=os.environ["BUCKET_STORAGE_HOST"],
        access_key=os.environ["BUCKET_STORAGE_ACCESS_KEY"],
        secret_key=os.environ["BUCKET_STORAGE_SECRET_KEY"],
        secure=False,
        cert_check=False,
    )
    for json_object, destination_file in [
        (eval_results.get_summary_scores_dict(), os.path.join(minio_results_dir_path, "summary_results.json")),  # type: ignore
        (eval_results.serializable_all_scores_dict(), os.path.join(minio_results_dir_path, "all_scores_list.json")),  # type: ignore
        (prompt_template, os.path.join(minio_results_dir_path, "prompt_template.json")),  # type: ignore
        (vars(args), os.path.join(minio_results_dir_path, "config.json")),
    ]:
        save_json_object_to_minio(json_object=json_object, destination_file=destination_file, client=minio_client)

    copy_results_files_to_minio(
        local_results_dir_path=local_results_dir_path,
        subdir="inferences",
        minio_results_dir_path=minio_results_dir_path,
        minio_client=minio_client,
    )

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    parser = get_inference_parser()
    args = parser.parse_args()
    asyncio.run(main(args))
