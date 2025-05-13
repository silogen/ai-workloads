from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class EvaluationScores:
    precision_bert: float
    recall_bert: float
    f1_bert: float
    f1_list: List[float]
    bleu_score: float
    accuracy: float


@dataclass_json
@dataclass
class EvaluationResults:
    prompts: List[str]
    scores: EvaluationScores


@dataclass_json
@dataclass
class JudgeScores:
    average_grade: float


@dataclass_json
@dataclass
class JudgeResults:
    judge_scores: JudgeScores
    prompts: List[str]
    inferences: List[str]
    judge_explanations: List[str]
    judge_grades: List[float]
