# Using Trust Eval to evaluate your RAG setup

In this tutorial, we will set up Trust-Eval for a custom RAG pipeline. This allows you to measure how well your application is performing over a **fixed** set of data.

## Create a dataset

As `Trust-Eval` works over a few datapoints, you will first need to define the datapoints used for evaluation. There are a few aspects to consider here:

1. **Define the schema**: Each datapoint should include the inputs to your application and the expected outputs. These expected outputs represent what the application should ideally produce. Don't worry on obtaining perfect outputs right away; evaluation is iterative, and initial outputs can serve as a starting point.
2. **Decide the number of datapoints**: There is no hard and fast rule in how much datapoints you should gather, rather, you should focus more on having proper coverage of edge cases you may want to guard against. Even 10-50 examples can provide a lot of value. Don't worry about getting a large number of datapoints to start as you can (and should) add over time.
3. **Gathering the datapoints**: Most teams that are starting a new project, generally start by collecting the first 10-20 datapoints by hand. After starting with these datapoints, you can add ore datapoints based on user feedback and real-world use cases. There is also the option of augmenting the dataset with synthetically generated data. However, we recommend not worrying about those at the start and just hand labeling ~10-20 examples.

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

Before running the program, let us define some of the parameters.

```python
# Get configs
generator_config = ResponseGeneratorConfig.from_yaml(yaml_path="generator_config.yaml")
logger.info(generator_config)
```

Most likely, you will want to test your own model. Hence, at the very minimum, you need to specify `model` and `max_length` (maximum context length that you model can take) in the yaml file.

> While we left `vllm = True`, note that enabling vllm on small amount of data may not bring much time savings as vllm takes about as much time to set up the CUDA graphs as it takes to do normal response generation without vllm on small amounts of datapoints.

## Document retrieval

Next, let us put together the full dataset with question, answers and documents.

```python
raw_docs = retrieve(questions, top_k=5)
data = construct_data(questions, answers, raw_docs, generator_config)
```

The first line of code allows us to retrieve documents that could help us answer this question. The second line does two things: (1) annotate the answerability of the question based on the documents (i.e. can the question be answered using the documents only) and (2) format the data into a structure that the evaluation rig expects.

Here, we are retrieving over the DPR wikipedia snapshot, which you can obtain using the following command:

```bash
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
gzip -xzvf psgs_w100.tsv.gz
export DPR_WIKI_TSV=$PWD/psgs_w100.tsv
```

Additionally, we use the pre-built dense index to save on GPU memory. The entire index will take about 31GB of space. You can obtain using this command:

```bash
wget https://huggingface.co/datasets/princeton-nlp/gtr-t5-xxl-wikipedia-psgs_w100-index/resolve/main/gtr_wikipedia_index.pkl
export GTR_EMB=$PWD/gtr_wikipedia_index.pkl
```

You can opt to build your own index; `retrieval.py` will automatically build an index if it does not find `gtr_wikipedia_index.pkl`. However, note that building the dense index is expensive for GPU memory ([ALCE](https://github.com/princeton-nlp/ALCE) use 80GB GPUs for this) and time-consuming.  

## Response Generation

```python
# Generator
generator = ResponseGenerator(generator_config)
data = generator.generate_responses(data)
generator.save_responses(output_path = "output/custom_data.json")
```

Now, we will pass the data generated and formatted from the previous step into the `generator` object to generate the LLM's response that we can judge. Here, it will be helpful to save the output for further analysis hence we can call `generator.save_responses()` with a custom output path (e.g. `output_path = "output/custom_data.json"`).

## Run Evaluations

```python
# Evaluate
evaluation_config = EvaluationConfig.from_yaml(yaml_path="eval_config.yaml")
logger.info(evaluation_config)
evaluator = Evaluator(evaluation_config)
evaluator.compute_metrics()
evaluator.save_results(output_path = "results/custom_data.json")
```

With the LLM output, expected output, documents and questions, we are now ready for evaluation. First define the evalution parameters in `eval_config.yaml`. At the very least, you need to define `eval_file` which contains the data generated  and saved from `generator` that will be evaluated by the evaluation rig.

When evaluating custom data, we will by default use the `claim match (cm)` mode to evaluate the LLM's outputs as opposed to `exact match (em)`. Briefly, claim match works by testing if the claim given by the LLM entails the expected or ground truth claim you have defined previously. If it entails, then the LLM's outout is correct. Exact match on the other hand is  more suitable if you have key facts that you know the LLM's answer has to contain. IF that is the case, you can define the facts and our program will try to find exact matches of the ground truth answers that you defined in the LLM output. We set the default to `claim match (cm)` as most applications will have questions-answer pairs with no clearly defined fact set.

## Results Analysis

Sample output:

```javascript
{
    "num_samples": 6,
    "answered_ratio": 50.0,
    "answered_num": 3,
    "answerable_num": 2,
    "overlapped_num": 2,
    "regular_length": 87.0,
    "answered_length": 90.66666666666667,
    "reject_rec": 75.0,
    "reject_prec": 100.0,
    "reject_f1": 85.71428571428571,
    "answerable_rec": 100.0,
    "answerable_prec": 66.66666666666667,
    "answerable_f1": 80.0,
    "macro_avg": 87.5,
    "macro_f1": 82.85714285714286,
    "regular_claims_nli": 33.33333333333333,
    "answered_claims_nli": 33.33333333333333,
    "calib_answered_claims_nli": 33.33333333333333,
    "calib_answerable_claims_nli": 50.0,
    "parametric_answered_claims_nli": 0.0,
    "calib_claims_nli_f1": 40.0,
    "regular_citation_rec": 32.5,
    "regular_citation_prec": 63.33333333333333,
    "regular_citation_f1": 42.95652173913043,
    "answered_citation_rec": 50.0,
    "answered_citation_prec": 76.66666666666666,
    "answered_citation_f1": 60.526315789473685,
    "trust_score": 61.127819548872175
}
```

## Complete script

```bash
CUDA_VISIBLE_DEVICES=0,1 python user_example.py 
```
