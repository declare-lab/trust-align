# # model_eval/evaluator.py

# from .metrics import custom_metric

# class Evaluator:
#     def __init__(self):
#         pass

#     def evaluate_responses(self, generated_responses, gold_answers):
#         # Code to evaluate using custom metrics
#         results = custom_metric(generated_responses, gold_answers)
#         return results

import copy
import json

import numpy as np
import scipy.stats as stats
from fuzzywuzzy import fuzz
from tqdm import tqdm

from .config import EvaluationConfig
from .utils import normalize_answer, remove_citations


class Evaluator:
    def __init__(self, yaml_file: str, config: EvaluationConfig):
        """
        Initializes the evaluator with configurations.

        Args:
            rejection_flag (str): The rejection flag text.
            rejection_fuzz_threshold (int): Fuzzy matching threshold for rejection detection.
        """
        self.config = EvaluationConfig.from_yaml(yaml_file)
        self.rejection_flag = config.rejection_flag
        self.rejection_fuzz_threshold = config.rejection_fuzz_threshold


    def _normalize_data(self, data):
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

    def _compute_answered_data(self, data):
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
            ) > self.rejection_fuzz_threshold
            if not rejection:
                answered_data.append(copy.deepcopy(item))
        return answered_data

    def _compute_answerable_data(self, data):
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
    
    def _process_data(self, data):
        normalized_data = self._normalize_data(data)
        answered_data = self._compute_answered_data(data)
        answerable_data = self._compute_answerable_data(data)

        normalized_answered_data = self._normalize_data(answered_data)
        normalized_answerable_data = self._normalize_data(answerable_data)
        return normalized_data, normalized_answered_data, normalized_answerable_data

    def load_data(self):
        self.eval_data = json.load(open(self.config.eval_file))
        return self.eval_data

    def compute_metrics(self, metric_funcs):
        """
        Compute evaluation metrics for the given dataset.

        Args:
            metric_funcs (dict): Dictionary of metric computation functions.

        Returns:
            dict: Dictionary containing computed metrics.
        """
        data = self.load_data()
        normalized_data, normalized_answered_data, normalized_answerable_data = self._process_data(data)

        result = {}

        # Compute metrics
        for metric_name, func in metric_funcs.items():
            metric_results = func(
                normalized_data,
                normalized_answered_data,
                normalized_answerable_data,
            )
            result.update(metric_results)

        return result

    def save_results(self, results, output_path):
        """
        Save evaluation results to a JSON file.

        Args:
            results (dict): Evaluation results.
            output_path (str): Path to save the JSON file.
        """
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
