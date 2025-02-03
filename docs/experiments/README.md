# Benchmarking your RAG setup with Trust Eval

In this tutorial, we will show you how to evaluate your RAG pipeline using our benchmark datasets (ASQA, ELI5 and QAMPARI) and Trust Eval metrics.

## Set up

Download `eval_data` from [Huggingface](https://huggingface.co/datasets/declare-lab/Trust-Score/tree/main/Trust-Score) and place it at the same level as the prompt folder. If you would like to use the default path configurations, please do not rename the folders. If you rename your folders, you will need to specify your own path.

```text
quickstart/
├── eval_data/
├── prompts/
```

## Quick look at the data

In this tutorial, we are working with **ASQA** where the questions are of the type long form factoid QA. Each sample has 3 fields: `question`, `answers`, `docs`. Below is one example of the dataset:

```javascript
[ ...
    {   // The question asked.
        "question": "Who has the highest goals in world football?",

        // A list containing all correct (short) answers to the question, represented as arrays where each element contains variations of the answer. 
        "answers": [
            ["Daei", "Ali Daei"],                // Variations for Ali Daei
            ["Bican", "Josef Bican"],            // Variations for Josef Bican
            ["Sinclair", "Christine Sinclair"]   // Variations for Christine Sinclair
        ],

        // A list of 100 dictionaries where each dictionary contains one document.
        "docs": [
            {   
                // The title of the document being referenced.
                "title": "Argentina\u2013Brazil football rivalry",

                // A snippet of text from the document.
                "text": "\"Football Player of the Century\", ...",

                // A binary list where each element indicates if the respective answer was found in the document (1 for found, 0 for not found).
                "answers_found": [0,0,0],

                // A recall score calculated as the percentage of correct answers that the document entails.
                "rec_score": 0.0
            },
        ]
    },
...
]
    
```

Please refer to [datasets](../concepts/datasets.md) page for examples of how ELI5 or QAMPARI samples.

## Configuring yaml files

For generator related configurations, there are three field that are mandatory: `data_type`, `model` and `max_length`. `data_type` determines which benchmark dataset to evaluate on. `model` determines which model to evaluate and `max_length` is the maximum context length of the model. We will be using `Qwen2.5-3B-Instruct` in this tutorial but you can replace it with the path to your model checkpoints to evaluate your model.

```yaml
data_type: "asqa"
model: Qwen/Qwen2.5-3B-Instruct
max_length: 8192
```

For evaluation related configurations, only `data_type` is mandatory.

```yaml
data_type: "asqa"
```

Your directory should now look like this:

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

```javascript
{ // refusal response: "I apologize, but I couldn't find an answer..."
    
    // Basic statistics
    "num_samples": 948,
    "answered_ratio": 50.0, // Ratio of (# answered qns / total # qns)
    "answered_num": 5, // # of qns where response is not refusal response
    "answerable_num": 7, // # of qns that ground truth answerable, given the documents
    "overlapped_num": 5, // # of qns that are both answered and answerable
    "regular_length": 46.6, // Average length of all responses
    "answered_length": 28.0, // Average length of non-refusal responses

    // Refusal groundedness metrics

    // # qns where (model refused to respond & is ground truth unanswerable) / # qns is ground truth unanswerable
    "reject_rec": 100.0,

    // # qns where (model refused to respond & is ground truth unanswerable) / # qns where model refused to respond
    "reject_prec": 60.0,

    // F1 of reject_rec and reject_prec
    "reject_f1": 75.0,

    // # qns where (model respond & is ground truth answerable) / # qns is ground truth answerable
    "answerable_rec": 71.42857142857143,

    // # qns where (model respond & is ground truth answerable) / # qns where model responded
    "answerable_prec": 100.0,

    // F1 of answerable_rec and answerable_prec
    "answerable_f1": 83.33333333333333,

    // Avg of reject_rec and answerable_rec
    "macro_avg": 85.71428571428572,

    // Avg of reject_f1 and answerable_f1
    "macro_f1": 79.16666666666666,

    // Response correctness metrics

    // Regardless of response type (refusal or answered), check if ground truth claim is in the response. 
    "regular_str_em": 41.666666666666664,

    // Only for qns with answered responses, check if ground truth claim is in the response. 
    "answered_str_em": 66.66666666666666,

    // Calculate EM for all qns that are answered and answerable, avg by # of answered questions (EM_alpha)
    "calib_answered_str_em": 100.0,

    // Calculate EM for all qns that are answered and answerable, avg by # of answerable questions (EM_beta)
    "calib_answerable_str_em": 71.42857142857143,

    // F1 of calib_answered_claims_nli and calib_answerable_claims_nli
    "calib_str_em_f1": 83.33333333333333,

    // EM score of qns that are answered and ground truth unanswerable, indicating use of parametric knowledge
    "parametric_str_em": 0.0,

    // Citation quality metrics

    // (Avg across all qns) Does the set of citations support statement s_i? 
    "regular_citation_rec": 28.333333333333332,

    // (Avg across all qns) Any redundant citations? (1) Does citation c_i,j fully support statement s_i? (2) Is the set of citations without c_i,j insufficient to support statement s_i? 
    "regular_citation_prec": 35.0,

    // F1 of regular_citation_rec and regular_citation_prec
    "regular_citation_f1": 31.315789473684212,

    // (Avg across answered qns only)
    "answered_citation_rec": 50.0,

    // (Avg across answered qns only)
    "answered_citation_prec": 60.0,

    // F1 answered_citation_rec and answered_citation_prec
    "answered_citation_f1": 54.54545454545455,

    // Avg (macro_f1, calib_claims_nli_f1, answered_citation_f1)
    "trust_score": 72.34848484848486
}
```

Please refer to [metrics](../concepts/metrics.md) for explanations of outputs when evaluating with ELI5 or QAMPARI.

## The end

Congratulations! You have reached the end of the tutorial and you are now ready to benchmark your own RAG application! 🥳

If you are interested in running the evaluations with custom data instead of benchmark data, please refer to the guide in [quickstart](../quickstart).
