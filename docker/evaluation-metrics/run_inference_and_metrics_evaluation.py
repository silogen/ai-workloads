import os
from argparse import Namespace
from pprint import pprint

from evaluation_metrics import logger
from evaluation_metrics.argument_parsers import get_inference_parser
from evaluation_metrics.call_inference_container.call_inference_container import (
    download_dataset,
    get_llm_client,
    read_prompt_template,
)
from evaluation_metrics.call_inference_container.call_inference_container import run as run_call_inference_container
from evaluation_metrics.call_inference_container.call_inference_container import (
    save_inference_results,
)
from evaluation_metrics.metrics.run_metrics_evaluation import read_jsonl_data
from evaluation_metrics.metrics.run_metrics_evaluation import run as run_metrics_evaluation
from evaluation_metrics.metrics.utils import save_results


def main(args: Namespace):
    """
    Main function to execute the inference and metrics evaluation pipeline.
    Args:
        args (Namespace): A namespace object containing the following attributes:
            - evaluation_dataset (str): The name of the evaluation dataset.
            - evaluation_dataset_version (str): The version of the evaluation dataset.
            - prompt_template_path (str): Path to the prompt template file.
            - llm_base_url (str): Base URL of the LLM service.
            - llm_port (int): Port number of the LLM service.
            - llm_endpoint (str): Endpoint of the LLM service.
            - context_column_name (str): Name of the column containing context data in the dataset.
            - gold_standard_column_name (str): Name of the column containing gold standard data in the dataset.
            - model_name (str): Name of the model to be used for inference.
            - model_path (str): Path to the model.
            - maximum_context_size (int): Maximum size of the context for inference.
            - use_data_subset (bool): Whether to use a subset of the dataset for evaluation.
            - dataset_split (str): The dataset split to use (e.g., "train", "test").
            - output_dir_path (str): Directory path to save output files.
    Workflow:
        1. Prints environment variables for debugging.
        2. Downloads the specified dataset.
        3. Reads the prompt template from the provided file path.
        4. Initializes the LLM client with the specified configuration.
        5. Runs inference using the dataset and prompt template.
        6. Saves the inference results to a file.
        7. Loads the generated inferences and evaluates metrics.
        8. Saves the evaluation results to a file and copies the file to MinIO.
    Outputs:
        - Inference results are saved to a file in the specified output directory.
        - Evaluation results are saved to a file in the specified output directory and to MinIO.
    Prints:
        - Status messages at each step of the pipeline.
        - Final evaluation results.
    """

    logger.info("Env Variables")
    logger.info(os.environ)

    ds = download_dataset(dataset=args.evaluation_dataset, version=args.evaluation_dataset_version)

    logger.info("Dataset loaded...")

    prompt_template = read_prompt_template(args.prompt_template_path)

    logger.info(f"Prompt template has been read in from {args.prompt_template_path}")

    parameters: dict = {}

    client = get_llm_client(base_url=args.llm_base_url, port=args.llm_port, endpoint=args.llm_endpoint)

    if args.use_data_subset > 0:
        logger.warning(f"Using a subset of the data: {args.use_data_subset} documents.")

    inference_results = run_call_inference_container(
        dataset=ds,
        prompt_template=prompt_template,
        context_column_name=args.context_column_name,
        gold_standard_column_name=args.gold_standard_column_name,
        llm_client=client,
        model_name=args.model_name,
        model_path=args.model_path,
        parameters=parameters,
        max_context_size=args.maximum_context_size,
        use_data_subset=args.use_data_subset,
        dataset_split=args.dataset_split,
    )

    inferences_filepath = save_inference_results(
        results=inference_results,
        output_dir_path=args.output_dir_path,
        model_name=args.model_name,
        evaluation_dataset=args.evaluation_dataset,
        evaluation_dataset_version=args.evaluation_dataset_version,
    )

    logger.info(f"Loading file with generated inferences: {inferences_filepath}")
    data = read_jsonl_data(inferences_filepath)
    logger.info("Data loaded, running metrics evaluation...")
    eval_results = run_metrics_evaluation(data)

    logger.info("Evaluation results:")
    logger.info(eval_results)

    results_filepath = inferences_filepath.replace("inferences", "results", 1).replace(".jsonl", ".json")

    save_results(results=eval_results, results_filepath=results_filepath)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    parser = get_inference_parser()
    args = parser.parse_args()
    main(args)
