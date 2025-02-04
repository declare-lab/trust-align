# # model_eval/evaluator.py

import copy
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.stats as stats
import torch.distributed as dist
from fuzzywuzzy import fuzz
from tqdm import tqdm

from .config import EvaluationConfig
from .logging_config import logger
from .metrics import (
    compute_trust_score,
    get_all_cm_scores,
    get_all_em5_scores,
    get_all_em_scores,
    get_basic_stats,
    get_citation_scores,
    get_macro_scores,
)
from .utils import normalize_answer, remove_citations


class Evaluator:
    def __init__(self, config: EvaluationConfig) -> None:
        """
        Initializes the evaluator with configurations.

        Args:
            
        """
        self.config = config
        self.result_path = self.config.result_path
        
        self.rejection_flag = config.rejection_flag
        self.rejection_threshold = config.rejection_threshold

        self.eval_type = config.eval_type
        self.eval_data = self.load_data()
        self.normalized_data, self.normalized_answered_data, self.normalized_answerable_data = self._process_data(self.eval_data)

        self.result: Dict[str, Any] = {}


    def _normalize_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize the data by removing citations.

        Args:
            data (list): Evaluation data to normalize.

        Returns:
            list: Normalized data.
        """
        normalized_data = copy.deepcopy(data)
        for item in normalized_data:
            item["output"] = remove_citations(item["output"])
        return normalized_data

    def _compute_answered_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract answered data from the dataset.

        Args:
            data (list): Evaluation dataset.

        Returns:
            list: Filtered answered data.
        """
        answered_data = []
        for item in data:
            rejection = fuzz.partial_ratio(
                normalize_answer(self.rejection_flag), normalize_answer(item["output"])
            ) > self.rejection_threshold
            if not rejection:
                answered_data.append(copy.deepcopy(item))
        return answered_data

    def _compute_answerable_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract answerable data from the dataset.

        Args:
            data (list): Evaluation dataset.

        Returns:
            list: Filtered answerable data.
        """
        answerable_data = []
        for item in data:
            answerable = any(
                np.bitwise_or.reduce([doc["answers_found"] for doc in item["docs"]])
            )
            if answerable:
                answerable_data.append(copy.deepcopy(item))
        return answerable_data
    
    def _process_data(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        normalized_data = self._normalize_data(data)
        answered_data = self._compute_answered_data(data)
        answerable_data = self._compute_answerable_data(data)

        normalized_answered_data = self._normalize_data(answered_data)
        normalized_answerable_data = self._normalize_data(answerable_data)
        return normalized_data, normalized_answered_data, normalized_answerable_data

    def load_data(self) -> List[Dict[str, Any]]:
        assert self.config.eval_file is not None
        self.eval_data = json.load(open(self.config.eval_file))["data"]
        return self.eval_data

    def compute_metrics(self, correctness: bool = True, citations: bool = True) -> Dict[str, Any]:
        """
        Compute evaluation metrics for the given dataset.

        Args:
            metric_funcs (dict): Dictionary of metric computation functions.

        Returns:
            dict: Dictionary containing computed metrics.
        """
        self.config.compute_correctness = correctness
        self.config.compute_citations = citations

        # Compute metrics
        logger.info("Computing basic stats...")
        self.result["num_samples"] = len(self.eval_data)
        metric_funcs = [get_basic_stats, get_macro_scores]
        for func in metric_funcs:
            metric_results = func(
                self.eval_data,
                self.normalized_data,
                self.normalized_answered_data,
                self.normalized_answerable_data,
                self.config
            )
            self.result.update(metric_results)

        correctness_funcs = {"em": get_all_em_scores, "em5": get_all_em5_scores, "cm": get_all_cm_scores}
        if self.config.compute_correctness:
            logger.info("Computing correctness scores...")
            func = correctness_funcs[self.eval_type]
            metric_results = func(
                self.eval_data,
                self.normalized_data,
                self.normalized_answered_data,
                self.normalized_answerable_data,
                self.config
            )
            self.result.update(metric_results)
        
        if self.config.compute_citations:
            logger.info("Computing citation scores...")
            metric_results = get_citation_scores(
                self.eval_data,
                self.normalized_data,
                self.normalized_answered_data,
                self.normalized_answerable_data,
                self.config
            )
            self.result.update(metric_results)
        
        if self.config.compute_correctness and self.config.compute_citations:
            logger.info("Computing trust scores...")
            self.result = compute_trust_score(self.result, self.config)
            
        logger.info(f"{self.result}")

        dist.destroy_process_group()

        return self.result

    def save_results(self, output_path: Optional[str] = None) -> None:
        """
        Save evaluation results to a JSON file.

        Args:
            results (dict): Evaluation results.
            result_path (str): Path to save the JSON file.
        """
        if output_path is not None:
            self.result_path = output_path

        assert self.result_path is not None
        with open(self.result_path, "w") as f:
            json.dump(self.result, f, indent=4)
