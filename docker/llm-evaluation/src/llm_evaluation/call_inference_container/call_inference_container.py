import asyncio
import json
import os
import time
from argparse import Namespace
from datetime import datetime
from itertools import islice
from typing import Any, AsyncGenerator, Dict, Iterable, List, Tuple

from datasets import load_dataset
from jsonlines import Writer
from llm_evaluation import logger
from llm_evaluation.argument_parsers import get_inference_parser
from openai import APIError, AsyncClient
from openai.types.chat import ChatCompletion
from tqdm import tqdm
from transformers import AutoTokenizer


def download_dataset(dataset: str, version: str) -> Dict:
    """
    Downloads a specified dataset and version.

    Args:
        dataset (str): The name of the dataset to download.
        version (str): The version of the dataset to download.

    Returns:
        Dict: A dictionary containing the downloaded dataset.
    """

    logger.info(f"Downloading dataset {dataset} version {version}")
    return load_dataset(dataset, version)


def get_llm_client(base_url: str, port: str | None, endpoint: str | None) -> AsyncClient:
    """
    Creates and returns a client for interacting with a language model (LLM) service.

    Args:
        base_url (str): The base URL of the LLM service.
        port (str | None): The port number to connect to the LLM service. If None, no port is appended.
        endpoint (str | None): The endpoint path for the LLM service. If None, no endpoint is appended.

    Returns:
        AsyncClient: An instance of the AsyncClient class configured to interact with the specified LLM service.
    """

    model_url = f"{base_url}{':'+port if port else ''}{'/'+endpoint if endpoint else ''}/"
    logger.info(f"Connecting to model at {model_url}")
    return AsyncClient(base_url=model_url, api_key="EMPTY")


def handle_llm_inference_result(doc_id: str, result: ChatCompletion) -> str | None:
    """
    Processes the result of an LLM (Large Language Model) inference and extracts the relevant content.

    Args:
        doc_id (str): The identifier for the document or request being processed.
        result (ChatCompletion): The result object from the LLM inference, containing choices and messages.

    Returns:
        str: The extracted content from the LLM inference result.
    """

    logger.info(f"Handling LLM inference result for document {doc_id}...")
    logger.info(f"Result: {result}")

    inference_result = result.choices[0].message.content

    return inference_result


def save_inference_results(result: Dict[str, Any], output_dir_path: str) -> str:
    """
    Saves a single inference result to a JSON file in the specified output directory.

    Args:
        result (Dict[str, Any]): A dictionary containing the inference result to be saved.
        output_dir_path (str): The directory path where the inference result file will be saved.

    Returns:
        str: The file path of the saved inference result.
    """
    inferences_filepath = os.path.join(
        output_dir_path,
        f"{result['doc_id']}.json",
    )

    logger.info(f"Writing inferences to {inferences_filepath}")
    with open(inferences_filepath, "w") as fp:
        fp.write(json.dumps(result))

    return inferences_filepath


def save_judge_inferences(
    results: list,
    output_dir_path: str,
    llm_name: str,
    judge_name: str,
    evaluation_dataset_name: str,
    evaluation_dataset_version: str,
) -> str:
    """
    Saves judge inference results to a JSONL file in the specified output directory.
    Args:
        results (list): A list of judge inference results to be saved.
        output_dir_path (str): The directory path where the judge inference results file will be saved.
        model_name (str): The name of the model used for inference.
        evaluation_dataset_name (str): The name of the evaluation dataset used.
        evaluation_dataset_version (str): The version of the evaluation dataset used.
    Returns:
        str: The file path of the saved judge inference results.
    Raises:
        OSError: If the output directory cannot be created or the file cannot be written.
    """

    if not os.path.exists(output_dir_path):
        logger.info(f"Creating path {output_dir_path}")
        os.makedirs(output_dir_path)

    inferences_filepath = os.path.join(
        output_dir_path,
        f"judge_{judge_name}_judging_inferences_by_{llm_name}--{evaluation_dataset_name.replace('/', '_')}--{evaluation_dataset_version}--{datetime.now().isoformat()}.jsonl",
    )

    logger.info(f"Writing inferences to {inferences_filepath}")
    with open(inferences_filepath, "w") as fp:
        writer = Writer(fp)
        writer.write_all(results)

    return inferences_filepath


def read_prompt_template(prompt_template_path: str) -> str:
    """
    Reads a prompt template from a file.

    Args:
        prompt_template_path (str): The file path to the prompt template.

    Returns:
        str: The content of the prompt template as a string.
    """
    with open(prompt_template_path, "r") as p_file:
        prompt_template = p_file.read()
    return prompt_template


def batched(iterable: Iterable, n: int, strict=False):
    """
    Splits an iterable into batches of a specified size.

    Args:
        iterable (Iterable): The input iterable to be split into batches.
        n (int): The size of each batch. Must be at least 1.
        strict (bool, optional): If True, raises a ValueError if the last batch
            is incomplete (i.e., its size is less than `n`). Defaults to False.

    Yields:
        Tuple: Batches of size `n` from the input iterable. If `strict` is False,
        the last batch may be smaller than `n` if there are not enough elements.
    Raises:
        ValueError: If `n` is less than 1.
        ValueError: If `strict` is True and the last batch is incomplete.
    Example:
        >>> list(batched(range(10), 3))
        [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9,)]
        >>> list(batched(range(10), 3, strict=True))
        ValueError: batched(): incomplete batch
    """

    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


async def async_iterable(iterable):
    for item in iterable:
        yield item


async def batched_async(iterable, batch_size: int):
    """Generator to batch an input iterator."""
    batch = []
    async for item in async_iterable(iterable):
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if len(batch) > 0:
        yield batch


async def get_inference_result(
    llm_client: AsyncClient, messages: List[Dict[str, str]], model_name: str, parameters: Dict[str, Any], doc_id: str
) -> Tuple[str, ChatCompletion]:
    """
    Sends a message to an LLM client to get an inference result and handles potential API errors.

    Args:
        llm_client (AsyncClient): The asynchronous client used to communicate with the LLM service.
        messages (List[Dict[str, str]]): OpenAI API compatible messages to be sent to the LLM.
        model_name (str): The name of the model to be used for inference.
        parameters (Dict[str, Any]): Additional parameters to configure the LLM request.
        doc_id (str): A unique identifier for the document for this inference request.

    Returns:
        Tuple[str, ChatCompletion]: A tuple containing the document ID and the response from the LLM.
    """

    try:
        response = await llm_client.chat.completions.create(
            messages=messages,
            model=model_name,
            **parameters,
        )
    except APIError as e:
        response = ChatCompletion(
            id=doc_id,
            object="chat.completion",
            created=int(datetime.now().timestamp()),
            model=model_name,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"**ERROR**: {str(e)}"},
                    "finish_reason": "length",
                }
            ],
        )
    return doc_id, response


def get_inference_result_as_dict(
    doc_id: str, inference_result: ChatCompletion, correct_answer: str, document: str, prompt_template: str
) -> Dict[str, Any]:
    """
    Generates a dictionary containing the inference result and other relevant information.

    Args:
        doc_id (str): The identifier for the document or request being processed.
        inference_result (ChatCompletion): The result object from the LLM inference.
        correct_answer (str): The gold standard answer for the document.
        document (str): The original document text.
        prompt_template (str): The prompt template used for inference.

    Returns:
        Dict[str, Any]: A dictionary containing the processed inference result, gold standard answer,
                        document ID, original document text, and prompt template.
    """
    logger.info(f"Processing inference result for ID {doc_id}...")

    processed_result = handle_llm_inference_result(doc_id=doc_id, result=inference_result)

    logger.info(f"Processed inference result: {processed_result}")

    result = {
        "inference_result": processed_result,
        "gold_standard_result": [correct_answer],
        "doc_id": doc_id,
        "document": document,
        "prompt": prompt_template,
    }

    return result


async def run(
    dataset: Dict,
    prompt_template: str,
    context_column_name: str,
    id_column_name: str,
    gold_standard_column_name: str,
    llm_client: AsyncClient,
    model_name: str,
    model_path: str,
    parameters: Dict[str, Any],
    max_context_size: int,
    batch_size: int,
    use_data_subset: int,
    dataset_split: str,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Executes inference on a dataset using a specified language model and yields results as they become available.

    Args:
        dataset (Dict): The dataset containing the input data for inference.
        prompt_template (str): The template used to format the prompt for the language model.
        context_column_name (str): The column name in the dataset containing the context documents.
        id_column_name (str): The column name in the dataset containing the document identifiers.
        gold_standard_column_name (str): The column name in the dataset containing the gold standard answers.
        llm_client (AsyncClient): The async client used to interact with the language model.
        model_name (str): The name of the language model to use for inference.
        model_path (str): The path to the model tokenizer for token counting.
        parameters (Dict[str, Any]): Additional parameters to configure the language model.
        max_context_size (int): The maximum number of tokens allowed in the context document.
        batch_size (int): The size of the batches for async LLM inference.
        use_data_subset (int): Number of documents to use for evaluation. If > 0, limits to this many documents.
        dataset_split (str): The split of the dataset to use (e.g., "train", "test", "validation").

    Yields:
        Dict[str, Any]: A dictionary containing the inference result, gold standard answer,
                        document ID, input document, and prompt for each processed document.

    Notes:
        - The function processes one batch of documents at a time, creating parallel tasks for each document in the batch.
        - Results within a batch are yielded as soon as they become available through async processing.
        - The next batch is not processed until all tasks in the current batch have completed or yielded results.
    """

    logger.info(f"Loading tokenizer from model path: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    counter = 0  # Used for running a subset of the dataset for testing purposes
    length_exclusion_counter = 0
    inference_errors_counter = 0
    processed_documents_counter = 0

    # Store document and gold standard info temporarily for each batch
    documents_map = {}
    correct_answers_map = {}

    inference_start_time = time.time()

    num_batches = (
        use_data_subset // batch_size + 1 if use_data_subset > 0 else len(dataset[dataset_split]) // batch_size + 1
    )
    logger.info(f"Starting inference on {num_batches} batches with batch size {batch_size}...")
    for batch_number, batch in tqdm(enumerate(batched(dataset[dataset_split], batch_size)), total=num_batches):
        inference_tasks = []
        batch_doc_ids = []

        logger.info(f"Processing inference batch {batch_number}/{num_batches}...")

        batch_start_time = time.time()

        for datum in batch:
            doc_id = datum[id_column_name]
            correct_answers_map[doc_id] = datum[gold_standard_column_name]
            documents_map[doc_id] = datum[context_column_name]

            logger.info(f"Running async inference. Batch {batch_number}/{num_batches}. Document: {doc_id}")
            batch_doc_ids.append(doc_id)

            # Format the prompt with the context document
            message = prompt_template.format(context=documents_map[doc_id])

            # Exclude messages that are longer than the context length
            tokens = tokenizer(message)["input_ids"]
            if len(tokens) > max_context_size:
                logger.warning(f"Message exceeds max context size of {max_context_size} tokens. Skipping...")
                logger.warning(f"Token length: {len(tokens)}, Character length: {len(message)}")
                length_exclusion_counter += 1
                continue

            inference_tasks.append(
                get_inference_result(
                    llm_client=llm_client,
                    messages=[{"role": "user", "content": message}],
                    model_name=model_name,
                    parameters=parameters,
                    doc_id=doc_id,
                )
            )

            counter += 1
            if use_data_subset > 0 and counter >= use_data_subset:

                break

        # Process results as they become available
        if inference_tasks:
            for completed_inference_task in asyncio.as_completed(inference_tasks):
                doc_id, inference_result = await completed_inference_task

                try:
                    result_dict = get_inference_result_as_dict(
                        doc_id=doc_id,
                        inference_result=inference_result,
                        correct_answer=correct_answers_map[doc_id],
                        document=documents_map[doc_id],
                        prompt_template=prompt_template,
                    )
                    processed_documents_counter += 1
                    yield result_dict
                except Exception as e:
                    logger.error(f"Error processing inference result for document {doc_id}: {str(e)}")
                    inference_errors_counter += 1

        logger.info(f"Batch {batch_number} processed in {time.time() - batch_start_time:.2f} seconds.")

        if use_data_subset > 0 and counter >= use_data_subset:
            logger.info(f"Ran inference for a subset of data: {use_data_subset} documents.")
            break

    logger.info(f"Total documents in dataset: {len(dataset[dataset_split])}")
    logger.info(f"Total documents processed for evaluation: {processed_documents_counter}")
    logger.info(f"\tDocuments excluded due to length: {length_exclusion_counter}")
    logger.info(f"\tInference errors encountered: {inference_errors_counter}")
    logger.info(f"Total inference time: {time.time() - inference_start_time:.2f} seconds.")


async def main(args: Namespace) -> str:
    """
    Main function to execute the inference pipeline.

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
            - maximum_context_size (int): Maximum size of the context for inference.
            - batch_size (int): Batch size for inference.
            - use_data_subset (int): Number of documents to use for evaluation. If > 0, limits to this many documents.
            - dataset_split (str): The dataset split to use (e.g., "train", "test").
            - output_dir_path (str): Directory path to save output files.

    Returns:
        str: The file path of the saved inference results.
    """

    logger.info(f"Loading dataset {args.evaluation_dataset_name} version {args.evaluation_dataset_version}")

    ds = download_dataset(dataset=args.evaluation_dataset_name, version=args.evaluation_dataset_version)

    logger.info("Dataset loaded...")

    prompt_template = read_prompt_template(args.prompt_template_path)

    logger.info(f"Prompt template has been read in from {args.prompt_template_path}")

    parameters: dict = {}

    client = get_llm_client(base_url=args.llm_base_url, port=args.llm_port, endpoint=args.llm_endpoint)

    results_dir_path = os.path.join(
        args.output_dir_path,
        f"inferences_{args.model_name}--{args.evaluation_dataset_name.replace('/', '_')}--{args.evaluation_dataset_version}--{datetime.now().isoformat()}",
    )

    if not os.path.exists(results_dir_path):
        os.makedirs(results_dir_path)
    logger.info(f"Results directory created at {results_dir_path}")

    async for inference_result in run(
        dataset=ds,
        prompt_template=prompt_template,
        context_column_name=args.context_column_name,
        id_column_name=args.id_column_name,
        gold_standard_column_name=args.gold_standard_column_name,
        llm_client=client,
        model_name=args.model_name,
        model_path=args.model_path,
        parameters=parameters,
        max_context_size=args.maximum_context_size,
        batch_size=args.batch_size,
        use_data_subset=args.use_data_subset,
        dataset_split=args.dataset_split,
    ):
        logger.info(f"Got inference result for document {inference_result['doc_id']}")
        logger.info(f"Inference result: {inference_result}")

        # Save the inference result to a file
        _ = save_inference_results(
            result=inference_result,
            output_dir_path=results_dir_path,
        )

        logger.info(f"Saved inference result for document {inference_result['doc_id']}")

    return results_dir_path


if __name__ == "__main__":
    parser = get_inference_parser()
    args = parser.parse_args()
    asyncio.run(main(args))
