# Trust Eval

Trust Eval is a holistic metric for evaluating trustworthiness of inline cited LLM outputs within the RAG framework. 

## Project Structure

```text
trust-eval/
├── trust-eval/
│   ├── __init__.py
│   ├── config.py
│   ├── llm.py
│   ├── response_generator.py
│   ├── evaluator.py
│   ├── metrics.py
│   ├── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_response_generator.py
│   ├── test_evaluator.py
├── README.md
├── poetry.lock
├── pyproject.toml
```

## Installation

```bash
conda create -n trust-eval python=3.10.13
conda activate trust-eval
poetry install
```

```bash
import nltk
nltk.download('punkt_tab')
```

## Example usage

```python
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
```

