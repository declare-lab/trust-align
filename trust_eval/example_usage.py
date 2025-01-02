
from config import EvaluationConfig, ResponseGeneratorConfig
from evaluator import Evaluator
from logging_config import logger
from response_generator import ResponseGenerator

# Generate responses
generator_config = ResponseGeneratorConfig.from_yaml(yaml_path="generator_config.yaml")
logger.info(generator_config)
generator = ResponseGenerator(generator_config)
generator.generate_responses()
generator.save_responses()

# Evaluate responses
evaluation_config = EvaluationConfig.from_yaml(yaml_path="eval_config.yaml")
logger.info(evaluation_config)
evaluator = Evaluator(evaluation_config)
evaluator.compute_metrics()
evaluator.save_results()