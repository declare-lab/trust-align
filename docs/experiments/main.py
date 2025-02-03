
from trust_eval.config import EvaluationConfig, ResponseGeneratorConfig
from trust_eval.evaluator import Evaluator
from trust_eval.logging_config import logger
from trust_eval.response_generator import ResponseGenerator

# Generate responses
generator_config = ResponseGeneratorConfig.from_yaml(yaml_path="configs/asqa_closedbook_rejection_baseline.yaml")
for key, value in vars(generator_config).items():
    logger.info(f"  {key}: {value}")
generator = ResponseGenerator(generator_config)
generator.generate_responses()
generator.save_responses()

# Evaluate responses
evaluation_config = EvaluationConfig.from_yaml(yaml_path="configs/asqa_closedbook_rejection_baseline.yaml")
for key, value in vars(evaluation_config).items():
    logger.info(f"  {key}: {value}")
evaluator = Evaluator(evaluation_config)
evaluator.compute_metrics()
evaluator.save_results()