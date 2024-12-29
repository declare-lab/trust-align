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
import numpy as np
import json
from tqdm import tqdm
from fuzzywuzzy import fuzz
import scipy.stats as stats


class Evaluator:
    def __init__(self, rejection_flag, rejection_fuzz_threshold):
        """
        Initializes the evaluator with configurations.

        Args:
            rejection_flag (str): The rejection flag text.
            rejection_fuzz_threshold (int): Fuzzy matching threshold for rejection detection.
        """
        self.rejection_flag = rejection_flag
        self.rejection_fuzz_threshold = rejection_fuzz_threshold

    def _normalize_data(self, data, remove_citations_func):
        """
        Normalize the data by removing citations.

        Args:
            data (list): Evaluation data to normalize.
            remove_citations_func (function): Function to remove citations from text.

        Returns:
            list: Normalized data.
        """
        normalized_data = copy.deepcopy(data)
        for item in normalized_data:
            item["output"] = remove_citations_func(item["output"])
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

    def compute_metrics(self, eval_data, remove_citations_func, metric_funcs):
        """
        Compute evaluation metrics for the given dataset.

        Args:
            eval_data (dict): Evaluation data, including args and data fields.
            remove_citations_func (function): Function to remove citations.
            metric_funcs (dict): Dictionary of metric computation functions.

        Returns:
            dict: Dictionary containing computed metrics.
        """
        data = eval_data["data"]
        normalized_data = self._normalize_data(data, remove_citations_func)
        answered_data = self._compute_answered_data(data)
        answerable_data = self._compute_answerable_data(data)

        normalized_answered_data = self._normalize_data(
            answered_data, remove_citations_func
        )
        normalized_answerable_data = self._normalize_data(
            answerable_data, remove_citations_func
        )

        result = {}

        # Compute metrics
        for metric_name, func in metric_funcs.items():
            result[metric_name] = func(
                normalized_data,
                normalized_answered_data,
                normalized_answerable_data,
            )

        return result

    def calculate_macro_metrics(self, data):
        """
        Calculate macro-level metrics for rejection and answerable detection.

        Args:
            data (list): Dataset to evaluate.

        Returns:
            dict: Macro metrics.
        """
        reject_rec_num = 0
        reject_rec = 0
        reject_prec_num = 0
        reject_prec = 0

        ans_rec_num = 0
        ans_rec = 0
        ans_prec_num = 0
        ans_prec = 0

        for item in data:
            answerable = any(
                np.bitwise_or.reduce([doc["answers_found"] for doc in item["docs"]])
            )
            rejection = fuzz.partial_ratio(
                normalize_answer(self.rejection_flag), normalize_answer(item["output"])
            ) > self.rejection_fuzz_threshold

            # Rejection recall
            if not answerable:
                reject_rec_num += 1
                if rejection:
                    reject_rec += 1

            # Rejection precision
            if rejection:
                reject_prec_num += 1
                if not answerable:
                    reject_prec += 1

            # Answerable recall
            if answerable:
                ans_rec_num += 1
                if not rejection:
                    ans_rec += 1

            # Answerable precision
            if not rejection:
                ans_prec_num += 1
                if answerable:
                    ans_prec += 1

        reject_recall = (
            100 * reject_rec / reject_rec_num if reject_rec_num > 0 else 0
        )
        reject_precision = (
            100 * reject_prec / reject_prec_num if reject_prec_num > 0 else 0
        )
        reject_f1_score = (
            2 * (reject_precision * reject_recall) / (reject_precision + reject_recall)
            if (reject_precision + reject_recall) > 0
            else 0
        )

        ans_recall = 100 * ans_rec / ans_rec_num if ans_rec_num > 0 else 0
        ans_precision = 100 * ans_prec / ans_prec_num if ans_prec_num > 0 else 0
        ans_f1_score = (
            2 * (ans_precision * ans_recall) / (ans_precision + ans_recall)
            if (ans_precision + ans_recall) > 0
            else 0
        )

        return {
            "reject_rec": reject_recall,
            "reject_prec": reject_precision,
            "reject_f1": reject_f1_score,
            "answerable_rec": ans_recall,
            "answerable_prec": ans_precision,
            "answerable_f1": ans_f1_score,
            "macro_avg": np.mean([reject_recall, ans_recall]),
            "macro_f1": np.mean([reject_f1_score, ans_f1_score]),
        }

    def save_results(self, results, output_path):
        """
        Save evaluation results to a JSON file.

        Args:
            results (dict): Evaluation results.
            output_path (str): Path to save the JSON file.
        """
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
