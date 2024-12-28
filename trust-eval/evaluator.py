# model_eval/evaluator.py

from .metrics import custom_metric

class Evaluator:
    def __init__(self):
        pass

    def evaluate_responses(self, generated_responses, gold_answers):
        # Code to evaluate using custom metrics
        results = custom_metric(generated_responses, gold_answers)
        return results
