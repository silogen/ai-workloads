import asyncio
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from llm_evaluation import logger
from llm_evaluation.call_inference_container.call_inference_container import (
    batched_async,
    get_inference_result,
    handle_llm_inference_result,
    read_prompt_template,
)
from llm_evaluation.data.data_classes import JudgeResult
from openai import AsyncClient


async def run_2step_judge(
    inference_result: Dict[str, Any],
    judge_prompt_step1_template: str,
    judge_prompt_step2_template: str,
    grade_regex: str,
    judge_client: AsyncClient,
    judge_model_name: str,
) -> Optional[JudgeResult]:
    """
    Executes a two-step judging process on a single inference result.

    Args:
        inference_result (Dict[str, Any]): The inference result containing original document, evaluated llm inference, and other information.
        judge_prompt_step1_template (str): Template for the first step of the judge prompt (explanation request).
        judge_prompt_step2_template (str): Template for the second step of the judge prompt (grade request).
        grade_regex (str): Regular expression to extract the grade from the judge's response.
        judge_client (Any): The client used to interact with the judge model.
        judge_model_name (str): The name of the judge model.

    Returns:
        Optional[JudgeResult]: A JudgeResult object containing the evaluation results, or None if an error occurred.
    """
    # JUDGE STEP 1: explanation request
    explanation_message = judge_prompt_step1_template.format(
        context=inference_result["document"], answer=inference_result["inference_result"]
    )
    explanation_messages = [{"role": "user", "content": explanation_message}]
    parameters: Dict[str, Any] = {}

    doc_id, judge_explanation_response = await get_inference_result(
        llm_client=judge_client,
        messages=explanation_messages,
        model_name=judge_model_name,
        parameters=parameters,
        doc_id=inference_result["doc_id"],
    )
    judge_explanation = handle_llm_inference_result(doc_id=doc_id, result=judge_explanation_response)
    if judge_explanation is None or judge_explanation.startswith("**ERROR**"):
        logger.error(f"Error in judge explanation inference: {judge_explanation}")
        return None

    # JUDGE STEP 2: grade request
    grade_messages = explanation_messages + [
        {"role": "assistant", "content": judge_explanation},
        {"role": "user", "content": judge_prompt_step2_template},
    ]
    # Set guided_regex to regular expression
    parameters = {"extra_body": {"guided_regex": grade_regex}}
    doc_id, judge_grade_inference_response = await get_inference_result(
        llm_client=judge_client,
        messages=grade_messages,
        model_name=judge_model_name,
        parameters=parameters,
        doc_id=inference_result["doc_id"],
    )
    judge_grade_inference = handle_llm_inference_result(doc_id=doc_id, result=judge_grade_inference_response)
    if judge_grade_inference is None or judge_grade_inference.startswith("**ERROR**"):
        logger.error(f"Error in judge grade inference: {judge_grade_inference}")
        return None
    grade_match = re.search(grade_regex, judge_grade_inference)
    if grade_match:
        judge_grade = int(grade_match.group(1))
    else:
        logger.error(f"Failed to extract grade from: {judge_grade_inference}")
        return None

    judge_result = JudgeResult(
        context_document=inference_result["document"],
        context_document_id=inference_result["doc_id"],
        gold_standard=inference_result["gold_standard_result"],
        llm_inference=inference_result["inference_result"],
        judge_explanation=judge_explanation,
        judge_grade=judge_grade,
    )
    return judge_result


async def run_2step_judge_on_inferences(
    inferences_data: List[Dict],
    judge_model_name: str,
    prompt_template_step1_path: str,
    prompt_template_step2_path: str,
    judge_client: AsyncClient,
    batch_size: int,
    output_dir_path: str,
) -> AsyncGenerator[JudgeResult, None]:
    """
    Asynchronously evaluates a set of inferences using a two-step judging process.

    Args:
        inferences_data (List[Dict]): A list of inference data dictionaries to be evaluated.
        judge_model_name (str): The name of the judge model to be used for evaluation.
        prompt_template_step1_path (str): File path to the first step's prompt template.
        prompt_template_step2_path (str): File path to the second step's prompt template.
        judge_client (AsyncClient): An instance of the judge client to interact with the model.
        batch_size (int): The number of inferences to process in each batch.
        output_dir_path (str): Directory path where the evaluation results will be saved.

    Yields:
        JudgeResult: The result of the evaluation for each inference.
    """

    judge_prompt_step1_template = read_prompt_template(prompt_template_step1_path)
    judge_prompt_step2_template = read_prompt_template(prompt_template_step2_path)
    grade_regex = "Grade: \\[\\[([1-9]|10)\\]\\]"  # We may want to parameterize this

    judged_documents_counter = 0
    judge_start_time = time.time()

    batch_number = 0
    total_batches = len(inferences_data) // batch_size + 1
    judge_errors_counter = 0
    async for inferences_batch in batched_async(inferences_data, batch_size):
        judge_tasks = []

        logger.info(f"Processing judgment batch {batch_number}/{total_batches}...")
        batch_number += 1
        batch_start_time = time.time()
        for i, inference_result in enumerate(inferences_batch):

            logger.info(f"Judging inference {i}: {inference_result['inference_result']}")
            judge_tasks.append(
                run_2step_judge(
                    inference_result=inference_result,
                    judge_prompt_step1_template=judge_prompt_step1_template,
                    judge_prompt_step2_template=judge_prompt_step2_template,
                    grade_regex=grade_regex,
                    judge_client=judge_client,
                    judge_model_name=judge_model_name,
                )
            )

        if judge_tasks:
            for completed_judge_task in asyncio.as_completed(judge_tasks):
                judge_result = await completed_judge_task
                if judge_result is None:
                    judge_errors_counter += 1
                    continue
                judge_result.save_dataclass_result(output_dir_path=output_dir_path)
                judged_documents_counter += 1
                yield judge_result

        logger.info(f"Batch {batch_number} processed in {time.time() - batch_start_time:.2f} seconds.")

    logger.info(f"Total documents to judge: {len(inferences_data)}")
    logger.info(f"Total documents judged {judged_documents_counter}")
    logger.info(f"Total judge errors: {judge_errors_counter}")
    logger.info(f"Total judge time: {time.time() - judge_start_time:.2f} seconds.")
