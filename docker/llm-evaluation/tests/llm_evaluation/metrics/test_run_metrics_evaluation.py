import json
import os
import tempfile
from argparse import Namespace

import jsonlines
import pytest
from llm_evaluation.metrics.run_metrics_evaluation import main, read_local_inference_data, run


def test_read_inference_data_nonexistent_path():
    """Test reading inference data from a nonexistent path."""
    with pytest.raises(FileNotFoundError):
        read_local_inference_data("/nonexistent/path")


def test_read_inference_data_empty_directory(tmpdir):
    """Test reading inference data from an empty directory."""
    result = read_local_inference_data(tmpdir)
    assert len(result) == 0


def test_read_inference_data_directory_with_no_json_files(tmpdir):
    """Test reading inference data from a directory with no JSON or JSONL files."""
    # Create a non-JSON file
    non_json_file = os.path.join(tmpdir, "test.txt")
    with open(non_json_file, "w") as f:
        f.write("This is a test file.")

    result = read_local_inference_data(tmpdir)
    assert len(result) == 0


def test_read_inference_data_invalid_json_file():
    """Test reading inference data from an invalid JSON file."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        temp_file.write(b"This is not valid JSON.")
        temp_path = temp_file.name

    results = read_local_inference_data(temp_path)

    assert len(results) == 0

    os.unlink(temp_path)


def test_read_inference_data_invalid_jsonl_file():
    """Test reading inference data from an invalid JSONL file."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as temp_file:
        temp_file.write(b"This is not valid JSONL.\n")
        temp_file.write(b'{"message": "This is valid JSONL."}\n')
        temp_file.write(b"This is not valid JSONL.\n")
        temp_path = temp_file.name

    results = read_local_inference_data(temp_path)

    assert len(results) == 1

    os.unlink(temp_path)


#  The test below is skipped for now, until the results from metrics
#  and judge evaluation are unified (refactor with dataclasses with common functionality).
@pytest.mark.skip
def test_main_function(tmpdir, mocker):
    """Test the main function workflow."""

    # Mock the Minio client and its fput_object function
    # mock_minio_client = mocker.patch("minio.Minio")
    # mock_minio_instance = mock_minio_client.return_value
    # mock_minio_instance.fput_object = mocker.Mock()

    # Mock the return values of compute_bertscore, compute_bleu_score, and compute_exact_match
    mocker.patch("llm_evaluation.metrics.metrics.compute_bertscore", return_value=0.5)
    mocker.patch("llm_evaluation.metrics.metrics.compute_bleu_score", return_value=0.5)
    mocker.patch("llm_evaluation.metrics.metrics.compute_exact_match", return_value=0.5)

    # Mock environment variables
    mocker.patch.dict(
        os.environ,
        {
            "BUCKET_STORAGE_HOST": "localhost:9000",
            "BUCKET_STORAGE_BUCKET": "mock_bucket",
            "BUCKET_STORAGE_ACCESS_KEY": "mock_access_key",
            "BUCKET_STORAGE_SECRET_KEY": "mock_secret_key",
        },
    )

    # Create a temporary input JSONL file
    input_file = tmpdir.join("input.jsonl")
    test_data = {
        "llm_inference": "test result",
        "gold_standard_result": ["test ground truth"],
        "prompt_template": "test prompt",
        "context_document": "test context",
    }
    with jsonlines.open(input_file, mode="w") as writer:
        writer.write(test_data)

    # Create a temporary output directory
    output_dir = tmpdir.mkdir("output")

    # Create a mock Namespace object for arguments
    args = Namespace(
        input_file_path=str(input_file),
        output_dir_path=str(output_dir),
    )

    # Run the main function
    main(args)

    # Check if the results were saved
    results_dir = os.path.join(output_dir, "evaluation_results")
    assert os.path.exists(results_dir)
    assert len(os.listdir(results_dir)) > 0


def test_run_evaluation_single_correct_answer():
    data = [
        {
            "llm_inference": "This is generated text.",
            "gold_standard_result": ["This is generated text."],
            "prompt_template": "Generate text.",
            "context_document": "test context",
        }
    ]
    results = run(generations=data)
    assert len(results.full_prompts) == 1
    assert "Generate text." in results.full_prompts
    assert len(results.scores.f1_list_bert) == 1
    assert results.scores.accuracy == 1.0
    assert results.scores.bleu_score == 1.0
    assert results.scores.precision_avg_bert == 1.0
    assert results.scores.recall_avg_bert == 1.0
    assert results.scores.f1_avg_bert == 1.0


def test_run_evaluation_incorrect_answer():
    data = [
        {
            "llm_inference": "This is incorrect text.",
            "gold_standard_result": ["This is generated text."],
            "prompt_template": "Generate text.",
            "context_document": "test context",
        }
    ]
    results = run(generations=data)
    assert len(results.full_prompts) == 1
    assert "Generate text." in results.full_prompts
    assert len(results.scores.f1_list_bert) == 1
    assert results.scores.accuracy == 0.0
    assert results.scores.bleu_score < 1.0
    assert results.scores.precision_avg_bert < 1.0
    assert results.scores.recall_avg_bert < 1.0
    assert results.scores.f1_avg_bert < 1.0


def test_run_evaluation_multiple_data_points():
    data = [
        {
            "llm_inference": "This is generated text.",
            "gold_standard_result": ["This is generated text."],
            "prompt_template": "Generate text.",
            "context_document": "test context",
        },
        {
            "llm_inference": "Another generated text.",
            "gold_standard_result": ["Another generated text."],
            "prompt_template": "Generate another text.",
            "context_document": "test context 2",
        },
    ]
    results = run(generations=data)
    assert len(results.full_prompts) == 2
    assert "Generate text." in results.full_prompts
    assert "Generate another text." in results.full_prompts
    assert len(results.scores.f1_list_bert) == 2
    assert results.scores.accuracy == 1.0
    assert results.scores.bleu_score == 1.0
    assert results.scores.precision_avg_bert == 1.0
    assert results.scores.recall_avg_bert == 1.0
    assert results.scores.f1_avg_bert == 1.0


def test_run_evaluation_multiple_correct_answers():
    data = [
        {
            "llm_inference": "This is generated text.",
            "gold_standard_result": ["This is generated text.", "This is another correct answer."],
            "prompt_template": "Generate text.",
        }
    ]
    try:
        run(generations=data)
    except NotImplementedError as e:
        assert str(e) == "Multiple correct answers"
    else:
        assert False, "Expected NotImplementedError"
