import json
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from dataclasses_json import dataclass_json
from llm_evaluation import logger


@dataclass_json
@dataclass
class EvaluationScores:
    precision_avg_bert: float
    recall_avg_bert: float
    f1_avg_bert: float
    precision_list_bert: List[float]
    recall_list_bert: List[float]
    f1_list_bert: List[float]
    bleu_score: float
    accuracy: float


@dataclass_json
@dataclass
class EvaluationResults:
    prompts: List[str]
    scores: EvaluationScores

    def get_scores_dict(self) -> Dict[str, float]:
        return {
            "mean_precision": self.scores.precision_avg_bert,
            "mean_recall": self.scores.recall_avg_bert,
            "mean_f1": self.scores.f1_avg_bert,
        }


@dataclass_json
@dataclass
class JudgeResult:
    context_document: str
    context_document_id: str
    gold_standard: str
    llm_inference: str
    judge_explanation: str
    judge_grade: float

    def save_dataclass_result(self, output_dir_path: str):
        """
        Saves the judge result to a JSONL file.
        """
        filepath = os.path.join(
            output_dir_path,
            f"judge_result_{self.context_document_id}.json",
        )
        with open(filepath, "w") as f:
            f.write(self.to_json())  # type: ignore
        logger.info(f"Saved judge result for document {filepath} to json file.")


@dataclass_json
@dataclass
class AggregatedJudgeResults:
    judge_results: dict[str, JudgeResult]
    average_grade: float
    total_candidate_judgments: int
    prompt_template_step_1: str
    prompt_template_step_2: str
    evaluation_dataset_name: str
    evaluation_dataset_version: str
    llm_name: str
    judge_name: str

    def get_scores_dict(self) -> Dict[str, float]:
        return {"mean_grade": np.mean([res.judge_grade for res in self.judge_results.values()])}

    def __str__(self):
        """
        Returns a string representation of the aggregated judge results for logging.
        """
        n_judgments = sum(1 for judge_result in self.judge_results.values() if judge_result.judge_grade is not None)
        return (
            f"Evaluation results:\n"
            f"\tPerformance of {self.llm_name} on {self.evaluation_dataset_name} v{self.evaluation_dataset_version} as judged by {self.judge_name}\n"
            f"\tAverage grade: {self.average_grade}\n"
            f"\tNumber of inferences for judging: {self.total_candidate_judgments}\n"
            f"\tNumber of judgments: {n_judgments}\n"
            f"\tPrompt template step 1: {self.prompt_template_step_1}\n"
            f"\tPrompt template step 2: {self.prompt_template_step_2}\n"
            f"\tEvaluation dataset name: {self.evaluation_dataset_name}\n"
            f"\tEvaluation dataset version: {self.evaluation_dataset_version}\n"
            f"\tLLM name: {self.llm_name}\n"
            f"\tJudge name: {self.judge_name}"
        )
