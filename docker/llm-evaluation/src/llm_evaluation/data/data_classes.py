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
    full_prompts: List[str]
    generations: List[str]
    scores: EvaluationScores

    def get_summary_scores_dict(self) -> Dict[str, float]:
        return {
            "mean_precision": self.scores.precision_avg_bert,
            "mean_recall": self.scores.recall_avg_bert,
            "mean_f1": self.scores.f1_avg_bert,
            "bleu_score": self.scores.bleu_score,
            "accuracy": self.scores.accuracy,
        }

    def serializable_all_scores_dict(self) -> Dict[str, List[float]]:
        return {
            "precision_list_bert": self.scores.precision_list_bert.tolist(),  # type: ignore
            "recall_list_bert": self.scores.recall_list_bert.tolist(),  # type: ignore
            "f1_list_bert": self.scores.f1_list_bert.tolist(),  # type: ignore
        }


@dataclass_json
@dataclass
class JudgeResult:
    context_document: str
    context_document_id: str
    gold_standard_result: str
    llm_inference: str
    judge_explanation: str
    judge_grade: float


@dataclass_json
@dataclass
class AggregatedJudgeResults:
    judge_results: dict[str, JudgeResult]
    average_grade: float
    total_candidate_judgments: int
    judge_prompt_step1_template: str
    judge_prompt_step2_template: str
    evaluation_dataset_name: str
    evaluation_dataset_version: str
    llm_name: str
    judge_name: str

    def get_scores_dict(self) -> Dict[str, float]:
        return {"mean_grade": np.mean([res.judge_grade for res in self.judge_results.values()])}

    def get_summary_dict(self) -> Dict[str, float]:
        summary = self.to_dict()  # type: ignore
        summary.pop("judge_results", None)
        return summary

    def get_grades_dict(self) -> Dict[str, float]:
        return {doc_id: judge_result.judge_grade for doc_id, judge_result in self.judge_results.items()}

    def get_generations_dict(self) -> Dict[str, str]:
        return {doc_id: judge_result.llm_inference for doc_id, judge_result in self.judge_results.items()}

    def get_judgments_dict(self) -> Dict[str, str]:
        return {doc_id: judge_result.judge_explanation for doc_id, judge_result in self.judge_results.items()}

    def get_full_prompts_dict(self) -> Dict[str, str]:
        return {
            doc_id: self.judge_prompt_step1_template.format(
                context=judge_result.context_document, answer=judge_result.llm_inference
            )
            for doc_id, judge_result in self.judge_results.items()
        }

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
            f"\tPrompt template step 1: {self.judge_prompt_step1_template}\n"
            f"\tPrompt template step 2: {self.judge_prompt_step2_template}\n"
            f"\tEvaluation dataset name: {self.evaluation_dataset_name}\n"
            f"\tEvaluation dataset version: {self.evaluation_dataset_version}\n"
            f"\tLLM name: {self.llm_name}\n"
            f"\tJudge name: {self.judge_name}"
        )
