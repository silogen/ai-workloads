from argparse import ArgumentParser


def get_inference_parser() -> ArgumentParser:
    """
    Returns an argument parser for the inference container script.
    """
    parser = ArgumentParser(prog="Call Inference Container")
    parser.add_argument(
        "-b", "--llm-base-url", type=str, default="http://localhost", help="Base URL of the LLM service."
    )
    parser.add_argument("-p", "--llm-port", type=str, default="8080", help="Port number of the LLM service.")
    parser.add_argument("-e", "--llm-endpoint", type=str, default="v1", help="Endpoint of the LLM service.")
    parser.add_argument(
        "-d", "--evaluation-dataset", type=str, default="abisee/cnn_dailymail", help="Name of the evaluation dataset."
    )
    parser.add_argument(
        "-v", "--evaluation-dataset-version", type=str, default="3.0.0", help="Version of the evaluation dataset."
    )
    parser.add_argument(
        "-l",
        "--dataset-split",
        type=str,
        help="Dataset split to use for evaluation (e.g., train, test, validation).",
        default="test",
    )
    parser.add_argument(
        "-o", "--output-dir-path", type=str, help="Path to the directory where output files will be saved."
    )
    parser.add_argument("-m", "--model-name", type=str, help="Name of the model to be used for inference.")
    parser.add_argument("-a", "--model-path", type=str, help="Path to the model.")
    parser.add_argument(
        "-x", "--maximum-context-size", type=int, help="Maximum size of the context to be used for inference."
    )
    parser.add_argument(
        "-c", "--context-column-name", type=str, help="Name of the column containing context data in the dataset."
    )
    parser.add_argument(
        "-i", "--id-column-name", type=str, help="Name of the column containing document id in the dataset."
    )
    parser.add_argument(
        "-g",
        "--gold-standard-column-name",
        type=str,
        help="Name of the column containing gold standard data in the dataset.",
    )
    parser.add_argument(
        "-z",
        "--batch-size",
        type=int,
        default=50,
        help="Size of batch of documents sent by the ASYNC client to the LLM service.",
    )
    parser.add_argument(
        "-s",
        "--use-data-subset",
        type=int,
        default=0,
        help="Use a subset of the data for evaluation. 0 (default) for full data, n>0 for n documents.",
    )
    parser.add_argument(
        "-r",
        "--prompt-template-path",
        type=str,
        default="/home/evaluation/example_prompts/example_summary_prompt.txt",
        help="Path to the prompt template file.",
    )
    return parser


def get_metrics_parser() -> ArgumentParser:
    """
    Returns an argument parser for the metrics evaluation script.
    """
    parser = ArgumentParser(prog="Compute Metrics")
    parser.add_argument("-i", "--input-file-path", help="Input file path for the JSON with LLM generated text.")
    parser.add_argument("-o", "--output-dir-path", help="Output directory for the results.")
    return parser
