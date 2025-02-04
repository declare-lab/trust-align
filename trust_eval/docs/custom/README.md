# Using Trust Eval to evaluate your RAG setup

In this tutorial, we will set up Trust-Eval for a custom RAG pipeline. This allows you to measure how well your application is performing over a **fixed** set of data.

## Create a dataset

As `Trust-Eval` works over a few datapoints, you will first need to define the datapoints used for evaluation. There are a few aspects to consider here:

1. **Define the schema**: Each datapoint should include the inputs to your application and the expected outputs. These expected outputs represent what the application should ideally produce. Don't worry on obtaining perfect outputs right away; evaluation is iterative, and initial outputs can serve as a starting point.
2. **Decide the number of datapoints**: There is no hard and fast rule in how much datapoints you should gather, rather, you should focus more on having proper coverage of edge cases you may want to guard against. Even 10-50 examples can provide a lot of value. Don't worry about getting a large number of datapoints to start as you can (and should) add over time.
3. **Gathering the datapoints**: Most teams that are starting a new project, generally start by collecting the first 10-20 datapoints by hand. After starting with these datapoints, you can add more datapoints based on user feedback and real-world use cases. There is also the option of augmenting the dataset with synthetically generated data. However, we recommend not worrying about those at the start and just hand labeling ~10-20 examples.

For this tutorial, we will create a simple dataset of six datapoints for a RAG-based question-answering application.

```python
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
```

*Parts of this section was adapted from the [LangSmith Evaluation tutorial](https://docs.smith.langchain.com/evaluation/tutorials/evaluation).*

## Set up

Before running the program, let us define some of the parameters for the generator.

```python
# Get configs
generator_config = ResponseGeneratorConfig.from_yaml(yaml_path="generator_config.yaml")
logger.info(generator_config)
```

Most likely, you will want to test your own model. Hence, at the very minimum, you need to specify `model` and `max_length` (maximum context length that you model can take) in the yaml file.

```yaml
model: Qwen/Qwen2.5-3B-Instruct
max_length: 8192
```

> While we left `vllm = True`, note that enabling vllm on small amount of data may not bring much time savings as vllm takes about as much time to set up the CUDA graphs as it takes to do normal response generation without vllm on small amounts of datapoints.

Before moving further, ensure that your working directory looks like this.

```text
quickstart/
├── prompts/
├── example_usage.py
├── generator_config.yaml
```

## Document retrieval

Before we perform document retrieval, we need to do some setup. First, as we are retrieving over the DPR wikipedia snapshot, you will need to download using the following command:

```bash
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
gzip -dv psgs_w100.tsv.gz
export DPR_WIKI_TSV=$PWD/psgs_w100.tsv
```

Additionally, we will use the pre-built dense index to save on GPU memory. The entire index will take about 31GB of space. You can obtain using this command:

```bash
wget https://huggingface.co/datasets/princeton-nlp/gtr-t5-xxl-wikipedia-psgs_w100-index/resolve/main/gtr_wikipedia_index.pkl
export GTR_EMB=$PWD/gtr_wikipedia_index.pkl
```

You can opt to build your own index; `retrieval.py` will automatically build an index if it does not find `gtr_wikipedia_index.pkl`. However, note that building the dense index is expensive for GPU memory ([ALCE](https://github.com/princeton-nlp/ALCE) use 80GB GPUs for this) and time-consuming.  

If you plan to build the index yourself, note that you will need the following additional packages: `pyserini==0.21.0` and `sentence-transformers==2.2.2`. You cam refer to the [pyserini repo](https://github.com/castorini/pyserini/tree/master) and [DPR repo](https://github.com/facebookresearch/DPR) for additional resources.

Next, let us put together the full dataset with question, answers and documents.

```python
raw_docs = retrieve(questions, top_k=5)
data = construct_data(questions, answers, raw_docs, generator_config)
```

The first line of code allows us to retrieve documents that could help answer this question. The second line does two things: (1) annotate the answerability of the question based on the documents (i.e. can the question be answered using the documents only) and (2) format the data into a structure that the evaluation rig expects.

## Response Generation

```python
# Generator
generator = ResponseGenerator(generator_config)
data = generator.generate_responses(data)
generator.save_responses(output_path = "output/custom_data.json")
```

Now, we will pass the data generated and formatted from the previous step into the `generator` object to generate the LLM's response that we can judge. Here, it will be helpful to save the output for further analysis hence we can call `generator.save_responses()` with a custom output path (e.g. `output_path = "output/custom_data.json"`).

## Quick look at the data

Below is one sample from the dataset that was created, together with the model generated response of the dataset:

```javascript
[ ...
    {       // The question asked.
            "question": "Where do Asian elephants live?",

            // Hand annotated answer
            "answers": [
                "Asian elephants live in the forests and grasslands of South and Southeast Asia, including India, Sri Lanka, Thailand, and Indonesia."
            ],

            // A list of 5 dictionaries where each dictionary contains one document.
            "docs": [
                {
                    "id": "2338517",

                    // The title of the document being referenced.
                    "title": "Asian elephant",

                    // A snippet of text from the document.
                    "text": "Asian elephant The Asian elephant ...",

                    // A recall score calculated as the percentage of correct answers that the document entails.
                    "score": 0.77099609375,

                    // A binary list where each element indicates if the respective answer was found in the document (1 for found, 0 for not found).
                    "answers_found": [
                        1
                    ]
                },
            ],

            // Output given by the model
            "output": "Asian elephants (\"Elephas maximus\") live in various habitats across the Indian subcontinent and Southeast Asia, including grasslands, tropical evergreen forests, semi-evergreen forests, moist deciduous forests, dry deciduous forests, and dry thorn forests [1][3]. They also inhabit cultivated and secondary forests, as well as scrublands. The range of these habitats extends from sea level to elevations of over 2,000 meters [3]. In Sri Lanka, Asian elephants inhabit the dry zone in the north, east, and southeast of the country, and are present in several national parks such as Udawalawe, Yala, Lunugamvehera, and Wilpattu [5]."
        },
...
]
    
```

## Run Evaluations

With the LLM output, expected output, documents and questions, we are now ready for evaluation. First define the evaluation parameters in `eval_config.yaml`. At the very least, you need to define `eval_file` which contains the data generated  and saved from `generator` that will be evaluated by the evaluation rig.

```python
# Evaluate
evaluation_config = EvaluationConfig.from_yaml(yaml_path="eval_config.yaml")
logger.info(evaluation_config)
evaluator = Evaluator(evaluation_config)
evaluator.compute_metrics()
evaluator.save_results(output_path = "results/custom_data.json")
```

Your eval config yaml file should at least look like this:

```yaml
eval_file: "output/custom_data.json"
```

When evaluating custom data, we will by default use the `claim match (cm)` mode to evaluate the LLM's outputs as opposed to `exact match (em)`. Briefly, claim match works by testing if the claim given by the LLM entails the expected or ground truth claim you have defined previously. If it entails, then the LLM's output is correct. Exact match on the other hand is  more suitable if you have key facts that you know the LLM's answer has to contain. IF that is the case, you can define the facts and our program will try to find exact matches of the ground truth answers that you defined in the LLM output. We set the default to `claim match (cm)` as most applications will have questions-answer pairs with no clearly defined fact set.

## Results Analysis

Trust-Score is a metric that comprehensively evaluates LLM trustworthiness on three main axes:

1) **Response Correctness**: Correctness of the generated claims
2) **Attribution Quality**: Quality of citations generated. Concerns the recall (Are generated statements well-supported by the set citations?) and precision (Are the citations relevant to the statements?) of citations.
3) **Refusal Groundedness**: Ability of the model to discern if the question can be answered given the documents

Sample output:

```javascript
{ // refusal response: "I apologize, but I couldn't find an answer..."
    
    // Basic statistics
    "num_samples": 6,
    "answered_ratio": 50.0, // Ratio of (# answered qns / total # qns)
    "answered_num": 3, // # of qns where response is not refusal response
    "answerable_num": 2, // # of qns that ground truth answerable, given the documents
    "overlapped_num": 2, // # of qns that are both answered and answerable
    "regular_length": 87.0, // Average length of all responses
    "answered_length": 90.66666666666667, // Average length of non-refusal responses
    
    // Refusal groundedness metrics

    // # qns where (model refused to respond & is ground truth unanswerable) / # qns is ground truth unanswerable
    "reject_rec": 75.0,

    // # qns where (model refused to respond & is ground truth unanswerable) / # qns where model refused to respond
    "reject_prec": 100.0,

    // F1 of reject_rec and reject_prec
    "reject_f1": 85.71428571428571,

    // # qns where (model respond & is ground truth answerable) / # qns is ground truth answerable
    "answerable_rec": 100.0,

    // # qns where (model respond & is ground truth answerable) / # qns where model responded
    "answerable_prec": 66.66666666666667,

    // F1 of answerable_rec and answerable_prec
    "answerable_f1": 80.0,

    // Avg of reject_rec and answerable_rec
    "macro_avg": 87.5,

    // Avg of reject_f1 and answerable_f1
    "macro_f1": 82.85714285714286,

    // Response correctness metrics

    // Regardless of response type (refusal or answered), check if ground truth claim is in the response. 
    "regular_claims_nli": 33.33333333333333,
    
    // Only for qns with answered responses, check if ground truth claim is in the response. 
    "answered_claims_nli": 33.33333333333333,

    // Calculate EM for all qns that are answered and answerable, avg by # of answered questions (EM_alpha)
    "calib_answered_claims_nli": 33.33333333333333,

    // Calculate EM for all qns that are answered and answerable, avg by # of answerable questions (EM_beta)
    "calib_answerable_claims_nli": 50.0,

    // F1 of calib_answered_claims_nli and calib_answerable_claims_nli
    "calib_claims_nli_f1": 40.0,

    // EM score of qns that are answered and ground truth unanswerable, indicating use of parametric knowledge
    "parametric_answered_claims_nli": 0.0,

    // Citation quality metrics

    // (Avg across all qns) Does the set of citations support statement s_i? 
    "regular_citation_rec": 32.5,

    // (Avg across all qns) Any redundant citations? (1) Does citation c_i,j fully support statement s_i? (2) Is the set of citations without c_i,j insufficient to support statement s_i? 
    "regular_citation_prec": 63.33333333333333,

    // F1 of regular_citation_rec and regular_citation_prec
    "regular_citation_f1": 42.95652173913043,

    // (Avg across answered qns only)
    "answered_citation_rec": 50.0,

    // (Avg across answered qns only)
    "answered_citation_prec": 76.66666666666666,

    // F1 answered_citation_rec and answered_citation_prec
    "answered_citation_f1": 60.526315789473685,

    // Avg (macro_f1, calib_claims_nli_f1, answered_citation_f1)
    "trust_score": 61.127819548872175
}
```

<img src="../../assets/trust_score.png" alt="Trust-Score" width="100%">

## Complete script

Congratulations! You have reached the end of the tutorial and you are now ready to evaluate your own RAG application! 🥳

Please find the complete code used in this tutorial in `custom_example.py`. You can use the following command to run the script:

```bash
CUDA_VISIBLE_DEVICES=0,1 python custom_example.py 
```

If you are interested in running the evaluations with benchmark data instead of custom data (similar to how it was done in the paper), please refer to the guide in [guides](../guides).
