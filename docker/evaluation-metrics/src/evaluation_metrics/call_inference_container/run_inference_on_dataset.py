import os

from evaluation_metrics import logger
from evaluation_metrics.call_inference_container.call_inference_container import (
    download_dataset,
    get_llm_client,
    read_prompt_template,
)
from evaluation_metrics.call_inference_container.call_inference_container import run as run_call_inference_container
from evaluation_metrics.call_inference_container.call_inference_container import (
    save_inference_results,
)


def run(
    evaluation_dataset_name,
    evaluation_dataset_version,
    prompt_template_path,
    llm_base_url,
    llm_port,
    llm_endpoint,
    context_column_name,
    gold_standard_column_name,
    model_name,
    model_path,
    maximum_context_size,
    use_data_subset,
    dataset_split,
    output_dir_path,
):

    ds = download_dataset(dataset=evaluation_dataset_name, version=evaluation_dataset_version)

    logger.info("Dataset loaded...")

    prompt_template = read_prompt_template(prompt_template_path)

    logger.info(f"Prompt template has been read in from {prompt_template_path}")

    parameters: dict = {}

    client = get_llm_client(base_url=llm_base_url, port=llm_port, endpoint=llm_endpoint)

    if use_data_subset > 0:
        logger.warning(f"Using a subset of the data: {use_data_subset} documents.")

    inference_results = run_call_inference_container(
        dataset=ds,
        prompt_template=prompt_template,
        context_column_name=context_column_name,
        gold_standard_column_name=gold_standard_column_name,
        llm_client=client,
        model_name=model_name,
        model_path=model_path,
        parameters=parameters,
        max_context_size=maximum_context_size,
        use_data_subset=use_data_subset,
        dataset_split=dataset_split,
    )

    inferences_filepath = save_inference_results(
        results=inference_results,
        output_dir_path=output_dir_path,
        model_name=model_name,
        evaluation_dataset_name=evaluation_dataset_name,
        evaluation_dataset_version=evaluation_dataset_version,
    )

    logger.info(f"Loading file with generated inferences: {inferences_filepath}")
    return inferences_filepath
