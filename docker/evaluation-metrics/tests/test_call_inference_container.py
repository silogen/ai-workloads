import json
import os
from argparse import Namespace
from unittest.mock import Mock

import pytest
from evaluation_metrics.call_inference_container.call_inference_container import (
    get_llm_client,
    get_llm_inference,
)
from evaluation_metrics.call_inference_container.call_inference_container import main as call_inference_container_main
from evaluation_metrics.call_inference_container.call_inference_container import (
    read_prompt_template,
    run,
)


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


def test_get_llm_inference():
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="This is a summary."))]
    mock_client.chat.completions.create.return_value = mock_response

    document = "This is a test document."
    model_name = "test-model"
    llm_parameters = {}

    prompt_template = "Summarise the following text."

    result = get_llm_inference(
        document=document,
        prompt_template=prompt_template,
        inference_client=mock_client,
        model_name=model_name,
        llm_parameters=llm_parameters,
    )

    assert result == "This is a summary."
    mock_client.chat.completions.create.assert_called_once()


def test_run():
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="This is a summary."))]
    mock_client.chat.completions.create.return_value = mock_response
    dataset = {"test": [{"article": "This is a test article.", "highlights": "This is a test answer."}]}
    model_name = "test-model"
    parameters = {}

    prompt_template = "Summarise the following text."

    results = run(
        dataset=dataset,
        prompt_template=prompt_template,
        context_column_name="article",
        gold_standard_column_name="highlights",
        llm_client=mock_client,
        model_name=model_name,
        parameters=parameters,
        max_context_size=100,
        use_data_subset=False,
        dataset_split="test",
    )

    assert len(results) == 1
    assert results[0]["inference_result"] == "This is a summary."
    assert results[0]["gold_standard_result"] == ["This is a test answer."]
    assert "prompt" in results[0]


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
        evaluation_dataset="abisee/cnn_dailymail",
        evaluation_dataset_version="3.0.0",
        output_dir_path="/tmp",
        model_name="test-model",
        maximum_context_size=100,
        use_data_subset=False,
        prompt_template_path=os.path.join(tmpdir, "prompt.txt"),
        context_column_name="article",
        dataset_split="test",
        gold_standard_column_name="highlights",
    )

    dataset = {"test": [{"article": "This is a test article.", "highlights": "This is a test answer."}]}
    prompt_template = "Summarise the following text."

    with open(os.path.join(tmpdir, "prompt.txt"), "w") as input_file:
        input_file.write(prompt_template)

    mock_client = Mock()
    mock_results = [
        {"answer": "This is a summary.", "correct_answer": "This is a test article.", "prompt": prompt_template}
    ]

    mocker.patch(
        "evaluation_metrics.call_inference_container.call_inference_container.download_dataset", return_value=dataset
    )
    mocker.patch(
        "evaluation_metrics.call_inference_container.call_inference_container.get_llm_client", return_value=mock_client
    )
    mocker.patch("evaluation_metrics.call_inference_container.call_inference_container.run", return_value=mock_results)
    mocker.patch("evaluation_metrics.call_inference_container.call_inference_container.LlamaTokenizer.from_pretrained")

    inferences_filepath = call_inference_container_main(mock_args)

    assert inferences_filepath.startswith("/tmp/inferences_test-model--abisee_cnn_dailymail--3.0.0--")

    with open(inferences_filepath, "r") as infile:
        inference_results = infile.read()

    assert json.loads(inference_results) == mock_results[0]
