import asyncio
import os
from argparse import Namespace
from datetime import datetime
from itertools import islice
from typing import Any, Dict, Iterable, List, Tuple

from datasets import load_dataset
from evaluation_metrics import logger
from evaluation_metrics.argument_parsers import get_inference_parser
from jsonlines import Writer
from openai import APIError, AsyncClient
from openai.types.chat import ChatCompletion
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


def handle_llm_inference_result(doc_id: str, result: ChatCompletion) -> str:
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
    logger.info(f"Got inference result for ID {doc_id}")
    logger.info(f"Inference result: {inference_result}")
    return inference_result


def save_inference_results(
    results: list, output_dir_path: str, model_name: str, evaluation_dataset: str, evaluation_dataset_version: str
) -> str:
    """
    Saves inference results to a JSONL file in the specified output directory.

    Args:
        results (list): A list of inference results to be saved.
        output_dir_path (str): The directory path where the inference results file will be saved.
        model_name (str): The name of the model used for inference.
        evaluation_dataset (str): The name of the evaluation dataset used.
        evaluation_dataset_version (str): The version of the evaluation dataset used.

    Returns:
        str: The file path of the saved inference results.
    """

    if not os.path.exists(output_dir_path):
        logger.info(f"Creating path {output_dir_path}")
        os.makedirs(output_dir_path)

    inferences_filepath = os.path.join(
        output_dir_path,
        f"inferences_{model_name}--{evaluation_dataset.replace('/', '_')}--{evaluation_dataset_version}--{datetime.now().isoformat()}.jsonl",
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


async def get_inference_result(
    llm_client: AsyncClient, message: str, model_name: str, parameters: Dict[str, Any], doc_id: str
) -> Tuple[str, ChatCompletion]:
    """
    Sends a message to an LLM client to get an inference result and handles potential API errors.

    Args:
        llm_client (AsyncClient): The asynchronous client used to communicate with the LLM service.
        message (str): The input message to be sent to the LLM.
        model_name (str): The name of the model to be used for inference.
        parameters (Dict[str, Any]): Additional parameters to configure the LLM request.
        doc_id (str): A unique identifier for the document for this inference request.

    Returns:
        Tuple[str, ChatCompletion]: A tuple containing the document ID and the response from the LLM.
    """

    try:
        response = await llm_client.chat.completions.create(
            messages=[{"role": "user", "content": message}],
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
) -> List[Dict[str, Any]]:
    """
    Executes inference on a dataset using a specified language model and returns the results.

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

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the inference results, gold standard answers,
                              document IDs, input documents, and prompts.
    """

    results = list()

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    counter = 0  # Used for running a subset of the dataset for testing purposes
    length_exclusion_counter = 0
    inference_errors_counter = 0
    correct_answers = dict()
    for batch in batched(dataset[dataset_split], batch_size):
        tasks = list()
        for datum in batch:
            doc_id = datum[id_column_name]
            correct_answers[doc_id] = datum[gold_standard_column_name]
            document = datum[context_column_name]

            logger.info(f"Running ASYNC inference on document: {doc_id}")

            # Format the prompt with the context document
            message = prompt_template.format(context=document)

            # Exclude messages that are longer than the context length
            tokens = tokenizer(message)["input_ids"]
            if len(tokens) > max_context_size:
                logger.warning(f"Message exceeds max content size of {max_context_size} tokens. Skipping...")
                logger.warning(f"Token length: {len(tokens)}, Character length: {len(message)}")
                length_exclusion_counter += 1
                continue

            tasks.append(
                get_inference_result(
                    llm_client=llm_client, message=message, model_name=model_name, parameters=parameters, doc_id=doc_id
                )
            )

            counter += 1
            if use_data_subset > 0 and counter >= use_data_subset:
                break

        for doc_id, inference_result in await asyncio.gather(*tasks):
            logger.info(f"Got inference result for ID {doc_id}")
            logger.info(f"Inference result: {inference_result}")
            processed_result = handle_llm_inference_result(doc_id=doc_id, result=inference_result)
            if processed_result.startswith("**ERROR**: "):
                logger.warning(
                    f"Error {processed_result} during handling inference results for document {doc_id}. Skipping..."
                )
                inference_errors_counter += 1
                continue

            results.append(
                {
                    "inference_result": processed_result,
                    "gold_standard_result": [correct_answers[doc_id]],
                    "doc_id": doc_id,
                    "document": document,
                    "prompt": prompt_template,
                }
            )

        if use_data_subset > 0 and counter >= use_data_subset:
            logger.info(f"Ran inference for a subset of data: {use_data_subset} documents.")
            break

        logger.info(f"Total documents: {len(dataset[dataset_split])}")
        logger.info(f"Total documents used for evaluation: {len(results)}")
        logger.info(f"\tDocuments excluded due to length: {length_exclusion_counter}")
        logger.info(f"\tInference errors encountered: {inference_errors_counter}")

    return results


def main(args: Namespace) -> str:
    """
    Main function to execute the inference pipeline.

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
            - batch_size (int): Batch size for inference.
            - use_data_subset (int): Number of documents to use for evaluation. If > 0, limits to this many documents.
            - dataset_split (str): The dataset split to use (e.g., "train", "test").
            - output_dir_path (str): Directory path to save output files.

    Returns:
        str: The file path of the saved inference results.
    """

    logger.info(f"Loading dataset {args.evaluation_dataset} version {args.evaluation_dataset_version}")

    ds = download_dataset(dataset=args.evaluation_dataset, version=args.evaluation_dataset_version)

    logger.info("Dataset loaded...")

    prompt_template = read_prompt_template(args.prompt_template_path)

    logger.info(f"Prompt template has been read in from {args.prompt_template_path}")

    parameters: dict = {}

    client = get_llm_client(base_url=args.llm_base_url, port=args.llm_port, endpoint=args.llm_endpoint)

    results = asyncio.run(
        run(
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
        )
    )

    inferences_filepath = save_inference_results(
        results=results,
        output_dir_path=args.output_dir_path,
        model_name=args.model_name,
        evaluation_dataset=args.evaluation_dataset,
        evaluation_dataset_version=args.evaluation_dataset_version,
    )

    return inferences_filepath


if __name__ == "__main__":
    parser = get_inference_parser()
    args = parser.parse_args()
    main(args)
