import asyncio
import os
from argparse import Namespace
from datetime import datetime

from llm_evaluation import logger
from llm_evaluation.argument_parsers import get_judge_inference_parser
from llm_evaluation.call_inference_container.call_inference_container import (
    download_dataset,
    get_llm_client,
    read_prompt_template,
)
from llm_evaluation.call_inference_container.call_inference_container import run as run_call_inference_container
from llm_evaluation.call_inference_container.call_inference_container import (
    save_inference_results,
)
from llm_evaluation.data.data_classes import AggregatedJudgeResults
from llm_evaluation.judge.run_judge_evaluation import run_2step_judge_on_inferences
from llm_evaluation.metrics.run_metrics_evaluation import read_inference_data
from llm_evaluation.metrics.utils import save_results


async def main(args: Namespace):

    ds = download_dataset(dataset=args.evaluation_dataset_name, version=args.evaluation_dataset_version)

    logger.info("Dataset loaded...")
    if args.use_data_subset > 0:
        logger.warning(f"Using a subset of the data: {args.use_data_subset} documents.")

    prompt_template = read_prompt_template(args.prompt_template_path)
    logger.info(f"Prompt template has been read in from {args.prompt_template_path}")

    results_dir_path = os.path.join(
        args.output_dir_path,
        f"inferences_{args.model_name}--{args.evaluation_dataset_name.replace('/', '_')}--{args.evaluation_dataset_version}--{datetime.now().isoformat()}",
        "inference_results",
    )
    logger.info(f"Results will be saved to {results_dir_path}")
    if not os.path.exists(results_dir_path):
        logger.info(f"Creating path {results_dir_path}")
        os.makedirs(results_dir_path)

    saved_results = []
    parameters: dict = {}
    client = get_llm_client(base_url=args.llm_base_url, port=args.llm_port, endpoint=args.llm_endpoint)

    async for inference_result in run_call_inference_container(
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
        result_path = save_inference_results(
            result=inference_result,
            output_dir_path=results_dir_path,
        )
        saved_results.append(result_path)
        logger.info(f"Saved inference result for document {inference_result['doc_id']}")

    logger.info(f"Loading file with generated inferences: {results_dir_path}")
    inferences_data = read_inference_data(results_dir_path)
    logger.info("Data loaded, running metrics evaluation...")

    logger.info(inferences_data)
    logger.info("Inference ran.")

    judge_client = get_llm_client(base_url=args.judge_base_url, port=args.judge_port, endpoint=args.judge_endpoint)

    aggregated_judge_results = AggregatedJudgeResults(
        judge_results={},
        average_grade=0.0,
        prompt_template_step_1=args.judge_prompt1_template_path,
        prompt_template_step_2=args.judge_prompt2_template_path,
        evaluation_dataset_name=args.evaluation_dataset_name,
        evaluation_dataset_version=args.evaluation_dataset_version,
        llm_name=args.model_name,
        judge_name=args.judge_model_name,
    )

    async for judge_result in run_2step_judge_on_inferences(
        inferences_data=inferences_data,
        judge_model_name=args.judge_model_name,
        prompt_template_step1_path=args.judge_prompt1_template_path,
        prompt_template_step2_path=args.judge_prompt2_template_path,
        judge_client=judge_client,
        batch_size=args.judge_batch_size,
        output_dir_path=args.output_dir_path,
    ):
        aggregated_judge_results.judge_results[judge_result.context_document_id] = judge_result
        logger.info(f"Processed judge result for document {judge_result.context_document_id}")

    logger.info("Aggregating judge scores...")
    aggregated_judge_results.compute_average_grade()
    aggregated_judge_results.print_evaluation_results()

    save_results(results=aggregated_judge_results, results_dir_path=results_dir_path, config=args)
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    parser = get_judge_inference_parser()
    args = parser.parse_args()
    asyncio.run(main(args))
