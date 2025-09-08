import asyncio
import glob
import json
import os
from argparse import Namespace
from unittest.mock import Mock

import httpx
from llm_evaluation.call_inference_container.call_inference_container import (
    batched,
    get_inference_result,
    get_llm_client,
    handle_llm_inference_result,
)
from llm_evaluation.call_inference_container.call_inference_container import main as call_inference_container_main
from llm_evaluation.call_inference_container.call_inference_container import (
    read_prompt_template,
    run,
)
from openai import APIError
from openai.types.chat import ChatCompletion


def test_get_inference_result(mocker):
    # Mock the AsyncClient
    mock_client = mocker.AsyncMock()
    mock_response = ChatCompletion(
        id="test_doc_id",
        object="chat.completion",
        created=42,
        model="test-model",
        choices=[{"index": 0, "message": {"role": "assistant", "content": "Test response"}, "finish_reason": "stop"}],
    )
    mock_client.chat.completions.create.return_value = mock_response

    # Test successful response
    doc_id, result = asyncio.run(
        get_inference_result(
            llm_client=mock_client,
            messages=[{"role": "system", "content": "Test message"}],
            model_name="test-model",
            parameters={"temperature": 0.7},
            doc_id="test_doc_id",
        )
    )

    assert doc_id == "test_doc_id"
    assert result == mock_response
    mock_client.chat.completions.create.assert_called_once_with(
        messages=[{"role": "system", "content": "Test message"}], model="test-model", temperature=0.7
    )

    # Test with API error
    mock_client.chat.completions.create.reset_mock()
    mock_request = mocker.Mock(spec=httpx.Request)
    mock_client.chat.completions.create.side_effect = APIError("Test API error", request=mock_request, body=None)

    doc_id, result = asyncio.run(
        get_inference_result(
            llm_client=mock_client,
            messages=[{"role": "system", "content": "Test message"}],
            model_name="test-model",
            parameters={"temperature": 0.7},
            doc_id="test_doc_id",
        )
    )

    assert doc_id == "test_doc_id"
    assert result.choices[0].message.content == "**ERROR**: Test API error"
    assert result.model == "test-model"


def test_batched():
    # Test with a complete batch
    data = list(range(10))
    batch_size = 3
    result = list(batched(data, batch_size))
    assert result == [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9,)]

    # Test with strict=True and incomplete batch
    try:
        list(batched(data, batch_size, strict=True))
    except ValueError as e:
        assert str(e) == "batched(): incomplete batch"

    # Test with n=1
    result = list(batched(data, 1))
    assert result == [(i,) for i in data]

    # Test with n greater than the length of the data
    result = list(batched(data, 15))
    assert result == [(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)]

    # Test with invalid n
    try:
        list(batched(data, 0))
    except ValueError as e:
        assert str(e) == "n must be at least one"


def test_get_llm_client():
    base_url = "https://localhost"
    port = "8080"
    endpoint = "v1"

    client = get_llm_client(base_url=base_url, port=port, endpoint=endpoint)

    assert client.base_url == "https://localhost:8080/v1/"

    base_url = "https://localhost"
    port = ""
    endpoint = "v1"

    client = get_llm_client(base_url=base_url, port=port, endpoint=endpoint)

    assert client.base_url == "https://localhost/v1/"

    base_url = "https://localhost"
    port = ""
    endpoint = ""

    client = get_llm_client(base_url=base_url, port=port, endpoint=endpoint)

    assert client.base_url == "https://localhost/"


def test_handle_llm_inference():

    corr_result = ChatCompletion(
        id="correct_test_doc",
        object="chat.completion",
        created=42,
        model="deep thought",
        choices=[
            {"index": 0, "message": {"role": "assistant", "content": "Document summary."}, "finish_reason": "stop"}
        ],
    )

    err_result = ChatCompletion(
        id="err_test_doc",
        object="chat.completion",
        created=44,
        model="shallow thought",
        choices=[
            {"index": 0, "message": {"role": "assistant", "content": "**ERROR**: APIError"}, "finish_reason": "length"}
        ],
    )

    result = handle_llm_inference_result(
        doc_id="correct_test_doc",
        result=corr_result,
    )

    assert result == "Document summary."

    result = handle_llm_inference_result(
        doc_id="err_test_doc",
        result=err_result,
    )

    assert result == "**ERROR**: APIError"


def test_run(mocker):
    mock_client = mocker.Mock()
    mock_response = mocker.Mock()
    mock_response.choices = [mocker.Mock(message=mocker.Mock(content="This is a summary."))]
    mock_client.chat.completions.create = mocker.AsyncMock(return_value=mock_response)

    # Mock the tokenizer
    mock_tokenizer = mocker.patch(
        "llm_evaluation.call_inference_container.call_inference_container.AutoTokenizer.from_pretrained"
    )
    mock_tokenizer_instance = mock_tokenizer.return_value
    mock_tokenizer_instance.return_value = {"input_ids": [1, 2, 3, 4, 5]}

    dataset = {
        "test": [{"article": "This is a test article.", "highlights": "This is a test answer.", "id": "test_doc"}]
    }
    model_name = "test-model"
    parameters = {}

    prompt_template = "Summarise the following text:\n{context}"

    async def collect_results():
        results = []
        async for result in run(
            dataset=dataset,
            prompt_template=prompt_template,
            context_column_name="article",
            gold_standard_column_name="highlights",
            id_column_name="id",
            llm_client=mock_client,
            model_name=model_name,
            model_path="meta-llama/Llama-3.1-8B-Instruct",
            parameters=parameters,
            max_context_size=100,
            batch_size=1,
            use_data_subset=False,
            dataset_split="test",
        ):
            results.append(result)
        return results

    results = asyncio.run(collect_results())

    assert len(results) == 1
    assert results[0]["llm_inference"] == "This is a summary."
    assert results[0]["gold_standard_result"] == ["This is a test answer."]
    assert "prompt_template" in results[0]


def test_read_prompt_template(tmpdir):
    tmplt = """Summarise the following text:
    {context}"""

    with open(os.path.join(tmpdir, "prompt.txt"), "w") as input_file:
        input_file.write(tmplt)

    prompt_template = read_prompt_template(os.path.join(tmpdir, "prompt.txt"))
    filled_template = prompt_template.format(context="This is a test context.")

    assert filled_template == "Summarise the following text:\n    This is a test context."


def test_main(mocker, tmpdir):
    mock_args = Namespace(
        llm_base_url="http://localhost",
        llm_port="8080",
        llm_endpoint="v1",
        evaluation_dataset_name="abisee/cnn_dailymail",
        evaluation_dataset_version="3.0.0",
        output_dir_path=tmpdir,
        model_name="test-model",
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        maximum_context_size=100,
        use_data_subset=False,
        prompt_template_path=os.path.join(tmpdir, "prompt.txt"),
        context_column_name="article",
        dataset_split="test",
        gold_standard_column_name="highlights",
        id_column_name="id",  # Add the id_column_name attribute
        batch_size=50,  # Add the batch_size attribute
    )

    dataset = {
        "test": [{"article": "This is a test article.", "highlights": "This is a test answer.", "id": "test_doc_id"}]
    }
    prompt_template = "Summarise the following text."

    with open(os.path.join(tmpdir, "prompt.txt"), "w") as input_file:
        input_file.write(prompt_template)

    # Use AsyncMock instead of Mock for async methods
    mock_client = mocker.AsyncMock()
    mocker.patch(
        "llm_evaluation.call_inference_container.call_inference_container.download_dataset", return_value=dataset
    )
    mocker.patch(
        "llm_evaluation.call_inference_container.call_inference_container.get_llm_client", return_value=mock_client
    )

    # Define mock_results before using it
    mock_results = [
        {
            "llm_inference": "This is a test inference.",
            "gold_standard_result": ["This is a test answer."],
            "prompt_template": "Test prompt",
            "context_document_id": "test_doc_id",
        }
    ]

    # Mock the run function to return a coroutine that yields mock_results
    async def mock_run_coroutine(*args, **kwargs):
        yield mock_results[0]

    # Mock AutoTokenizer to avoid actual model loading
    mock_tokenizer = mocker.MagicMock()

    # Mock the run function
    mocker.patch(
        "llm_evaluation.call_inference_container.call_inference_container.run", return_value=mock_run_coroutine()
    )

    # This is redundant code - removed duplicate mocking and call
    mocker.patch("llm_evaluation.call_inference_container.call_inference_container.AutoTokenizer.from_pretrained")

    inferences_filepath = asyncio.run(call_inference_container_main(mock_args))

    print(inferences_filepath)
    print(os.listdir(inferences_filepath))

    assert inferences_filepath.startswith(os.path.join(tmpdir, "inferences_test-model--abisee_cnn_dailymail--3.0.0--"))

    # Use glob to find the json file created in the output directory
    json_files = glob.glob(os.path.join(inferences_filepath, "inferences", "*.json"))
    assert len(json_files) == 1

    with open(json_files[0], "r") as infile:
        inference_results = infile.read()

    assert json.loads(inference_results) == mock_results[0]
