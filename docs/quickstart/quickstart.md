# Using Trust Eval to evaluate your RAG setup

In this tutorial, we will show you how to evaluate your RAG pipeline using our benchmark documents and Trust Eval metrics.

## Set up

Download `eval_data` from [Huggingface](https://huggingface.co/datasets/declare-lab/Trust-Score/tree/main/Trust-Score) and place it at the same level as the prompt folder. If you would like to use the default path configurations, please do not rename the folders. If you rename your folders, you will need to specify your own path.

```text
quickstart/
├── eval_data/
├── prompts/
```

## Quick look at the data

In this tutorial, we are working with ASQA where the questions are of the type long form factoid QA. Each sample has 3 fields: `question`, `answers`, `docs`. Below is one example of the dataset:

```json
[ ...
    {
        "question": "Who has the highest goals in world football?",
        "answers": [
            ["Daei","Ali Daei"],
            ["Bican","Josef Bican"],
            ["Sinclair","Christine Sinclair"]
        ],
        "docs": [
            {
                "title": "Argentina\u2013Brazil football rivalry",
                "text": "\"Football Player of the Century\", ...",
                "answers_found": [0,0,0],
                "rec_score": 0.0
            },
            {
                "title": "Godfrey Chitalu",
                "text": "have beaten Gerd M\u00fcller's record of 85 goals in a year, ...",
                "answers_found": [0,0,0],
                "rec_score": 0.0
            },
            {
                "title": "Godfrey Chitalu",
                "text": "highest official tally claimed by a national football association. ...",
                "answers_found": [0,0,0],
                "rec_score": 0.0
            },
            {
                "title": "Wartan Ghazarian",
                "text": "goals (4 in World Cup qualifiers, 3 in Asian Cup qualifiers, 12 in friendlies). ...",
                "answers_found": [0,0,0],
                "rec_score": 0.0
            },
            {
                "title": "Josef Bican",
                "text": "for Christmas, but died less than 2 weeks before that, at the age of 88. Josef Bican Josef \"Pepi\" Bican (25 September ...",
                "answers_found": [0,0,0],
                "rec_score": 0.0
            },
        ]
    },
...
]
    
```

Please see the [datasets](../concepts/datasets.md) page for a more in depth explanation of the data.

## Configuring yaml files

For generator related configurations, there are three field that are mandatory: `data_type`, `model` and `max_length`. `data_type` determines which benchamrk dataset to evaluate on. `model` determines which model to evaluate and `max_length` is the maximum context length of the model. We will be using `Qwen2.5-3B-Instruct` in this tutorial but you can replace it with the path to your model checkpoints to evaluate your model.

```yaml
data_type: "asqa"
model: Qwen/Qwen2.5-3B-Instruct
max_length: 8192
```

For evaluation related configurations, only `data_type` is mandatory.

```yaml
data_type: "asqa"
```

Your directory should look like this:

```text
quickstart/
├── eval_data/
├── prompts/
├── generator_config.yaml
├── eval_config.yaml
```

## Running evals

Now define your main script:

```python
from trust_eval.config import EvaluationConfig, ResponseGeneratorConfig
from trust_eval.evaluator import Evaluator
from trust_eval.logging_config import logger
from trust_eval.response_generator import ResponseGenerator

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

Your directory should look like this:

```text
quickstart/
├── eval_data/
├── prompts/
├── example_usage.py
├── generator_config.yaml
├── eval_config.yaml
```

```bash
CUDA_VISIBLE_DEVICES=0,1 python example_usage.py 
```

> Note: Define the GPUs you wish to run on in `CUDA_VISIBLE_DEVICES`. For reference, we are able to run up to 7b models on two A40s.

Sample output:

```json
{
    "answered_ratio": 50.0,
    "answered_num": 5,
    "answerable_num": 7,
    "overlapped_num": 5,
    "regular_length": 46.6,
    "answered_length": 28.0,
    "reject_rec": 100.0,
    "reject_prec": 60.0,
    "reject_f1": 75.0,
    "answerable_rec": 71.42857142857143,
    "answerable_prec": 100.0,
    "answerable_f1": 83.33333333333333,
    "macro_avg": 85.71428571428572,
    "macro_f1": 79.16666666666666,
    "regular_str_em": 41.666666666666664,
    "regular_str_hit": 20.0,
    "answered_str_em": 66.66666666666666,
    "answered_str_hit": 40.0,
    "calib_answered_str_em": 100.0,
    "calib_answered_str_hit": 100.0,
    "calib_answerable_str_em": 71.42857142857143,
    "calib_answerable_str_hit": 71.42857142857143,
    "calib_str_em_f1": 83.33333333333333,
    "parametric_str_em": 0.0,
    "parametric_str_hit": 0.0,
    "regular_citation_rec": 28.333333333333332,
    "regular_citation_prec": 35.0,
    "regular_citation_f1": 31.315789473684212,
    "answered_citation_rec": 50.0,
    "answered_citation_prec": 60.0,
    "answered_citation_f1": 54.54545454545455,
    "trust_score": 72.34848484848486
}
```

For a more in-depth explanation of the output, please see [metrics](../concepts/metrics.md).
