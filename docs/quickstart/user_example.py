from trust_eval.config import EvaluationConfig, ResponseGeneratorConfig
from trust_eval.data import construct_data
from trust_eval.evaluator import Evaluator
from trust_eval.logging_config import logger
from trust_eval.response_generator import ResponseGenerator
from trust_eval.retrieval import retrieve

# Get configs
generator_config = ResponseGeneratorConfig.from_yaml(yaml_path="generator_config.yaml")
logger.info(generator_config)

# Construct data
questions = ["Where do Asian elephants live?", 
         "Can Orthodox Jewish people eat shellfish?", 
         "How many species of sea urchin are there?", 
         "How heavy is an adult golden jackal?",
         "Who owns FX network?",
         "How many letters are used in the French alphabet?"]
answers = ["Asian elephants live in the forests and grasslands of South and Southeast Asia, including India, Sri Lanka, Thailand, and Indonesia.",
          "No, Orthodox Jewish people do not eat shellfish because it is not kosher according to Jewish dietary laws.",
          "There are about 950 species of sea urchins worldwide.",
          "An adult golden jackal typically weighs between 6 and 14 kilograms (13 to 31 pounds).",
          "FX network is owned by The Walt Disney Company.",
          "The French alphabet uses 26 letters, the same as the English alphabet."]

raw_docs = retrieve(questions, top_k=5)
data = construct_data(questions, answers, raw_docs, generator_config)

# Generator
generator = ResponseGenerator(generator_config)
data = generator.generate_responses(data)
generator.save_responses(output_path = "output/custom_data.json")

# Evaluate
evaluation_config = EvaluationConfig.from_yaml(yaml_path="eval_config.yaml")
logger.info(evaluation_config)
evaluator = Evaluator(evaluation_config)
evaluator.compute_metrics()
evaluator.save_results(output_path = "results/custom_data.json")

