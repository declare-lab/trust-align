import os
import re
import torch
import time
import json
import random
import colorlog
import argparse
import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter, defaultdict, OrderedDict
from transformers import AutoTokenizer
from text_clustering import ClusterClassifier
from claims_gen import generate_claims
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets

INSTRUCTION_SINGLE_TOPIC = "The examples below are questions from the same cluster. Identify a single short topic they share in common, \
for example: Philosophy, Lifestyle, Linear Algebra, Biochemistry, Economics, etc. \
Additionally, evaluate if the topics in the examples are broadly suitable as knowledge-demanding questions that require additional research or grounding. \
Exclude any sensitive, inappropriate, or irrelevant content, such as sex, explicit violence, ads & scams, and other NSFW subjects. \
Consider a wide range of content, including scientific, educational, historical, cultural, and practical applications. \
Provide a rating from 1 to 7 based on the topic's dependence on additional knowledge or search materials: \
a score of 1 indicates the question can be answered with common sense alone, without needing any additional information lookup; \
a score of 5 means the topic requires a combination of common sense and additional lookup, roughly an equal split between the two; \
a score of 7 indicates that answering the question directly would be difficult, and without additional information, the answer would likely be incorrect. \
The output format should be like this: Topic: the_topic, Demanding value rating: score."

TEMPLATE_SINGLE_TOPIC = "<s>[INST]{instruction}\n\nExamples:\n{examples}\nRemember that the output format should be like this: Topic: the_topic, Demanding value rating: score.[/INST]"


fmt_string = '%(log_color)s %(asctime)s - %(levelname)s - %(message)s'
log_colors = {
        'DEBUG': 'white',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'purple'
        }
colorlog.basicConfig(log_colors=log_colors, format=fmt_string, level=colorlog.INFO)
logger = colorlog.getLogger(__name__)
logger.setLevel(colorlog.INFO)


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
    "--n_samples",
    type=int,
    default=100_000,
    )
    parser.add_argument(
        "--start",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--end",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--n_selected",
        type=int,
        default=3_000,
    )
    parser.add_argument(
        "--embed_devices",
        type=str,
        nargs='+',
        default=["cuda:5"],
        help="List of devices to use",
    )
    parser.add_argument(
        "--embed_batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--save_load_dir",
        type=str,
        default="asqa_data",
    )
    parser.add_argument(
        "--build_hf_ds",
        action="store_true",
        help="Builds HF datasets",
    )
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        choices=["asqa", "qampari", "eli5"],
        default="asqa", 
        help="specify dataset name")
    parser.add_argument(
        "--input_dataset",
        type=str,
        default="HuggingFaceFW/FW-12-12-2023-CC-2023-06",
        help="dataset with the samples to use for clustering",
    )
    parser.add_argument(
        "--data_subset",
        type=str,
        default=None,
        help="dataset subset",
    )
    parser.add_argument(
        "--input_content", 
        type=str, 
        default="question"
    )
    parser.add_argument(
        "--topic_mode",
        type=str,
        choices=["single_topic", "multiple_topics"],
        default="multiple_topics",
        help="Specify 'single_topic' to generate only one topic and score its demanding value, or 'multiple_topics' to generate the 3 most relevant topics in the cluster.",
    )
    parser.add_argument(
        "--dbscan_eps",
        type=float,
        default=0.08,
        help="The maximum distance between two samples for them to be considered as in the neighborhood of each other.",
    )
    parser.add_argument(
        "--dbscan_min_samples",
        type=int,
        default=50,
        help="The number of samples in a neighborhood for a point to be considered as a core point.",
    )
    parser.add_argument(
        "--mode",
        choices=["run", "infer", "load", "skip"],
        default="run",
        help="Run the pipeline from scratch/infer on new texts/load existing model to build selective hf datasets and plot/skip mode for just processing data format",
    )
    parser.add_argument(
        '--summary_model_kwargs', 
        type=str, 
        nargs='*', 
        help='Key=value pairs for summary model keyword arguments'
    )
    
    args = parser.parse_args()

    summary_model_kwargs = {}
    if args.summary_model_kwargs:
        for item in args.summary_model_kwargs:
            key, value = item.split('=', 1)
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            if key == "dtype":
                keywords = ["float16", "float32"]
                for keyword in keywords:
                    if keyword in value:
                        value = getattr(torch, keyword)
                        break
            summary_model_kwargs[key] = value
    
    args.summary_model_kwargs = summary_model_kwargs
    return args


def find_valid_cluster(summary: str):
    score = summary.split(" Score: ")[1].strip()
    return score.isdecimal()


def format_raw_dataset(cluster):
    summary = cluster["summary"]
    category = summary.split(". Score")[0].strip()
    score = summary.split(" Score: ")[1].strip()
    return {
        "cluster_id": cluster['cluster_id'],
        "examples": cluster['examples'],
        "category": category, 
        "demanding_score": score
    }


def select_by_demanding_score(data, n_selected):
    score_groups = defaultdict(list)
    for item in data:
        score_groups[item['demanding_score']].append(item['examples'])
    
    # Calculate the amount of data for each score segment
    total_count = sum(len(group) for group in score_groups.values())
    proportions = {score: len(group) / total_count for score, group in score_groups.items()}
    
    # Select data according to scale
    selected_data = []
    for score, group in score_groups.items():
        n_to_select = int(proportions[score] * n_selected)
        selected_data.extend(random.sample(group, min(n_to_select, len(group))))
    
    # If the selected data is insufficient n_selected, add the data randomly
    if len(selected_data) < n_selected:
        remaining_data = [item for group in score_groups.values() for item in group if item not in selected_data]
        additional_needed = n_selected - len(selected_data)
        selected_data.extend(random.sample(remaining_data, min(additional_needed, len(remaining_data))))
    
    # If the number of selected data exceeds n_selected, it is randomly reduced
    if len(selected_data) > n_selected:
        selected_data = random.sample(selected_data, n_selected)
    
    return selected_data


def plot_distributions(ds_path, args):
    """Plot distribution of educational score of topics & distribution of samples accross topics"""
    if not os.path.exists(f"{args.save_load_dir}/figs"):
        os.makedirs(f"{args.save_load_dir}/figs")

    ds = load_from_disk(ds_path, keep_in_memory=True)
    ds = ds.map(format_raw_dataset, num_proc=1, keep_in_memory=True)
    logger.critical(f"valid cluster categories num: {len(ds['category'])}")
    ds = ds.filter(lambda x: x["demanding_score"] not in ["None", ""])
    # distribution of scores
    df = ds.to_pandas()
    df["demanding_score"] = pd.to_numeric(df["demanding_score"], errors="coerce")
    df.dropna(subset=["demanding_score"], inplace=True)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(df["demanding_score"], kde=False, bins=10)
    plt.title("Distribution of Demanding Scores")
    plt.xlabel("Demanding Score")
    plt.ylabel("Frequency")
    plt.savefig(f"{args.save_load_dir}/figs/{args.dataset_name}_{args.mode}_demanding_score.png", bbox_inches="tight")

    # distribution of samples
    df = ds.to_pandas().explode("examples")
    sorted_filtered_ds = df.groupby(by="category").size().sort_values(ascending=False)
    category_df = sorted_filtered_ds.reset_index()
    category_df.columns = ["category", "number_files"]
    logger.info(f"Saving csv in {args.save_load_dir}!")
    category_df.to_csv(f"{args.save_load_dir}/figs/{args.dataset_name}_{args.mode}_categories_count.csv", index=False)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(25, 20))

    barplot = sns.barplot(
        x="number_files", y="category", data=category_df, palette="Blues_d", ci=None
    )

    plt.xlabel("Number of Examples")
    plt.ylabel("Categories")
    plt.title("Histogram of Categories and their number of FW files")
    plt.tight_layout(pad=1.0)
    plt.show()
    plt.savefig(f"{args.save_load_dir}/figs/{args.dataset_name}_{args.mode}_topics_dist.png", bbox_inches="tight", dpi=200)


def build_hf_data_clusters(cc, texts=None, labels=None):
    """
    Build an HF dataset containing information on each cluster.

    Args:
        cc: ClusterClassifier object.
        texts: list of texts used for inference mode.
        labels: list of cluster labels corresponding to the texts for inference mode.

    If `texts` and `labels` are not provided, the function will use the data available in `cc`
    to construct the dataset. Otherwise it will run in inference mode on texts.
    """
    cluster_data = []
    for cluster_id in cc.label2docs.keys():
        if cluster_id == -1:
            continue

        # inference mode
        if texts is not None and labels is not None:
            labels_array = np.array(labels)
            files_in_cluster = np.where(labels_array == cluster_id)[0]
            examples = [texts[doc_id] for doc_id in files_in_cluster]
        else:
            doc_ids = cc.label2docs[cluster_id]
            if texts:
                examples = [texts[doc_id] for doc_id in doc_ids]
            else:
                examples = [cc.texts[doc_id] for doc_id in doc_ids]

        cluster_info = {
            "cluster_id": cluster_id,
            "summary": cc.cluster_summaries[cluster_id],
            "examples": examples,
        }

        if not labels:
            cluster_info["position"] = cc.cluster_centers[cluster_id]

        cluster_data.append(cluster_info)

    return Dataset.from_pandas(pd.DataFrame(cluster_data))


def build_hf_datasets(cc, new_data, cluster_labels, embeddings, args):
    """Build HF files & clusters datasts"""
    logger.info("Building HF datasets for clusters info...")
    clusters_ds = build_hf_data_clusters(cc, new_data, cluster_labels)
    logger.warning(f"Eliminate '-1' labels: leaving {sum([len(examples) for examples in clusters_ds['examples']])} => {sum([len(examples) for examples in clusters_ds['examples']])*100/len(new_data):.2f}% of the original dataset")
    if embeddings is not None:
        logger.info(f"Saving infer embeddings to {args.save_load_dir}")
        with open(f"{args.save_load_dir}/{args.dataset_name}_{args.mode}_embeddings.npy", "wb") as f:
            np.save(f, embeddings)

    logger.info(f"Filtering failed summary case")
    valid_ds = clusters_ds.filter(find_valid_cluster, input_columns="summary", num_proc=1, keep_in_memory=True)
    valid_ds = valid_ds.map(format_raw_dataset, num_proc=1, remove_columns="summary", keep_in_memory=True)
    valid_df = valid_ds.to_pandas().explode("examples")
    valid_df.sort_values(by=['cluster_id'], inplace=True)
    valid_df.reset_index(drop=True, inplace=True)  # remove index_level
    logger.info("Valid df info...")
    logger.critical(valid_df.head())
    logger.info(valid_df.info())
    valid_ds = Dataset.from_pandas(valid_df)
    logger.warning(f"Size after dropping invalid summarised clusters: {len(valid_ds)} => {len(valid_ds)*100/sum([len(examples) for examples in clusters_ds['examples']]):.2f}% of the cluster dataset")

    n_selected = min(args.n_selected, len(valid_ds))
    logger.info(f"Select {n_selected} samples from {args.dataset_name} dataset...")
    selected_ds = select_by_demanding_score(valid_ds, n_selected)

    logger.info(f"Saving at {args.save_load_dir}...")
    clusters_ds.save_to_disk(f"{args.save_load_dir}/{args.dataset_name}_{args.mode}_clusters")
    with open(f"{args.save_load_dir}/{args.dataset_name}_{args.mode}.json", "w") as f:
        json.dump(selected_ds, f, indent=4)
    
    
def asqa_load(args):
    with open(args.input_dataset) as f:
        data = json.load(f)

    new_data = []
    q_max = ""
    q_max_length = 0
    for k, v in data['train'].items():
        qa_pair = [{
        'context': qa['context'],
        'question': qa['question'],
        'short_answers': qa['short_answers'],
        'wikipage': qa['wikipage']
        } for qa in v['qa_pairs']]

        wikipages = v['wikipages']

        annotations = [{
            'knowledge': anno['knowledge'],
            'long_answer': anno['long_answer']
        } for anno in v['annotations']]

        sample_id = k

        question = v['ambiguous_question']

        if len(question) > q_max_length:
            q_max = question
            q_max_length = len(question)

        answer = v['annotations'][-1]['long_answer']

        new_data.append(
            OrderedDict({
                "qa_pairs": qa_pair,
                "wikipages": wikipages,
                "annotations": annotations,
                "sample_id": sample_id,
                "question": question,
                "answer": answer
            })
        )
    
    # show first example:
    logger.critical(new_data[0])

    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
    max_length = len(tokenizer(q_max)['input_ids'])
    
    logger.info(f"total num of {args.dataset_name} = {len(new_data)}")
    logger.info(f"max token num of {args.dataset_name} = {max_length}")
    
    return new_data, max_length


def qampari_load(args):
    def format_data(data: list):
        data[0] = re.sub(r'"|\[\[|\]\]|{{.*?}}', '', data[0])
        data[0] = re.sub(r'\s+', ' ', data[0])
        data[0] = data[0].strip()

        if '|' in data[0]:
            parts = data[0].split('|')
            parts.extend(data[1:])
            return parts
        else:
            return data
        
    def remove_duplicates(data):
        seen = set()
        unique_data = []
        
        for item in data:
            if item[0] and item[0] not in seen:
                seen.add(item[0])
                unique_data.append(item)
        
        return unique_data
    
    train_data = [json.loads(line) for line in open(args.input_dataset)]
    # train set
    new_data = []
    q_max = ""
    q_max_length = 0
    for item in train_data:
        id = item['qid']
        if item['question_text']:
            question = item['question_text']
            if len(question) > q_max_length:
                q_max = question
                q_max_length = len(question)
            
            answers = remove_duplicates([format_data(ans['aliases']) for ans in item['answer_list']])
            answer = ", ".join([ans[0] for ans in answers]) + "."
            new_data.append(
                OrderedDict({
                    "id": id,
                    "question": question,
                    "answers": answers,
                    "answer": answer
                })
            ) 
    # test set
    if os.path.exists(args.input_dataset.replace("train", "test")):
        test_data = [json.loads(line) for line in open(args.input_dataset.replace("train", "test"))]
        for item in test_data:
            id = item['qid']
            if item['question_text']:
                question = item['question_text']
                answers = remove_duplicates([format_data(ans['aliases']) for ans in item['answer_list']])
                answer = ", ".join([ans[0] for ans in answers]) + "."
                new_data.append(
                    {
                        "id": id,
                        "question": question,
                        "answers": answers,
                        "answer": answer
                    }
                )
    
    # show first example:
    logger.critical(new_data[0])

    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
    max_length = len(tokenizer(q_max)['input_ids'])
    
    logger.info(f"total num of {args.dataset_name} = {len(new_data)}")
    logger.info(f"max token num of {args.dataset_name} = {max_length}")
    
    return new_data, max_length
    
    
def eli5_load(args):
    def filter_unwanted(example, categories):
        # filter empty data
        if not example['title'].strip() or not example['answers']['text'][0].strip() or not example['answers']['score'][0]:
            return False
        # filter useless categories
        if example['category'] not in categories:
            return False
        return True

    def format_data(example):
        # sort answer
        sorted_indices = sorted(range(len(example['answers']['score'])), key=lambda i: example['answers']['score'][i], reverse=True)
        example['answers']['a_id'] = [example['answers']['a_id'][i] for i in sorted_indices]
        example['answers']['text'] = [example['answers']['text'][i] for i in sorted_indices]
        example['answers']['score'] = [example['answers']['score'][i] for i in sorted_indices]
        example['answers']['text_urls'] = [example['answers']['text_urls'][i] for i in sorted_indices]
        
        # format data
        e = {}
        e['question'] = example['title']
        e['question_ctx'] = example['selftext']
        e['answer'] = example['answers']['text'][0]
        e['category'] = example['category']
        e['max_score'] = example['answers']['score'][0]
        return e
    
    ds = load_from_disk(args.input_dataset)
    ds = concatenate_datasets([ds['train'], ds['validation1'], ds['validation2'], ds['test']])
    # deduplicate
    ds = Dataset.from_pandas(ds.to_pandas().drop_duplicates(subset=['title'], keep='last'))

    # filter unwantd data
    categories = ['Biology', 'Chemistry', 'Culture', 'Earth Science', 'Economics', 'Engineering', 'Mathematics', 'Physics', 'Psychology', 'Technology']
    valid_ds = ds.filter(filter_unwanted, fn_kwargs={"categories": categories}, num_proc=5, keep_in_memory=True)
    logger.warning(f"Size after dropping unwanted datapoints: {len(valid_ds)} => {len(valid_ds)*100/len(ds):.2f}% of the cluster dataset")

    # format data and sort according to max_score
    valid_ds = valid_ds.map(format_data, num_proc=5, keep_in_memory=True, remove_columns=ds.column_names)
    valid_ds = valid_ds.sort('max_score', reverse=True)
    
    # select target samples
    n_selected = min(args.n_selected, len(valid_ds))
    logger.info(f"Select {n_selected} samples from {args.dataset_name} dataset...")
    category_groups = defaultdict(list)
    for category in categories:
        category_groups[category] = valid_ds.filter(lambda example: example['category'] == category, batch_size=10000, keep_in_memory=True)
    # Calculate the amount of data for each score segment
    proportions = {category: len(group) / len(valid_ds) for category, group in category_groups.items()}
    logger.info(f"Selection based on data ratio:\n{proportions}")

    # Select indexes according to scale
    selected_indexes = []
    for category, group in category_groups.items():
        n_to_select = int(proportions[category] * n_selected)
        selected_indexes.extend(random.sample(range(len(group)), min(n_to_select, len(group))))

    # If the selected data is insufficient n_selected, add the data randomly
    if len(selected_indexes) < n_selected:
        remaining_indexes = [index for group in category_groups.values() for index in range(len(group)) if index not in selected_indexes]
        additional_needed = n_selected - len(selected_indexes)
        selected_indexes.extend(random.sample(remaining_indexes, additional_needed))
    
    # If the number of selected data exceeds n_selected, it is randomly reduced
    if len(selected_indexes) > n_selected:
        selected_indexes = random.sample(selected_indexes, n_selected)

    new_data = valid_ds.select(selected_indexes).to_list()

    # show first example:
    logger.critical(new_data[0])
    
    q_max = max([item[args.input_content] for item in new_data], key=len)
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
    max_length = len(tokenizer(q_max)['input_ids'])
    
    logger.info(f"total num of {args.dataset_name} = {len(new_data)}")
    logger.info(f"max token num of {args.dataset_name} = {max_length}")
    
    return new_data, max_length


def expertqa_load(args):
    from collections import OrderedDict
    def read_jsonl(filepath, limit=None, verbose=False):
        """Read jsonl file to a List of Dicts."""
        data = []
        with open(filepath, "r") as jsonl_file:
            for idx, line in enumerate(jsonl_file):
                if limit is not None and idx >= limit:
                    break
                if verbose and idx % 100 == 0:
                    # Print the index every 100 lines.
                    print("Processing line %s." % idx)
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print("Failed to parse line: `%s`" % line)
                    raise e
        print("Loaded %s lines from %s." % (len(data), filepath))
        return data
    
    examples_json = read_jsonl(args.input_dataset)
    new_data = []
    for item in examples_json:
        annotator_id = item['annotator_id']
        category = ": ".join([item['metadata']['field'], item['metadata']['specific_field']])
        question = item['question']
        answer = list(item['answers'].values())[0]['revised_answer_string']

        if answer:
            new_data.append(
                    OrderedDict({
                        "annotator_id": annotator_id,
                        "category": category,
                        "question": question,
                        "answer": answer,
                    })
                )
    
    # show first example:
    logger.critical(new_data[0])

    q_max = max([item[args.input_content] for item in new_data], key=len)
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
    max_length = len(tokenizer(q_max)['input_ids'])
    
    logger.info(f"total num of {args.dataset_name} = {len(new_data)}")
    logger.info(f"max token num of {args.dataset_name} = {max_length}")
    
    return new_data, max_length



if __name__ == "__main__":
    args = get_args()

    template = (
        None
        if args.topic_mode == "multiple_topics"
        else TEMPLATE_SINGLE_TOPIC
    )
    instruction = (
        None
        if args.topic_mode == "multiple_topics"
        else INSTRUCTION_SINGLE_TOPIC
    )
    logger.info(f"Using {args.topic_mode} for topic labeling")
    
    if args.dataset_name == "asqa":
        new_data, embed_max_seq_length = asqa_load(args)
    elif args.dataset_name == "qampari":
        new_data, embed_max_seq_length = qampari_load(args)
    elif args.dataset_name == "eli5" or args.dataset_name == "expertqa":
        if args.dataset_name == "eli5":
            new_data, embed_max_seq_length = eli5_load(args)
        if args.dataset_name == "expertqa":
            new_data, embed_max_seq_length = expertqa_load(args)
        # Get claims for each question
        questions = []
        passages = []
        for item in new_data:
            questions.append(item['question'])
            passages.append(item['answer'])
        claims = generate_claims(questions, passages)
        claims.sort(key=lambda x: x[0])
        filtered_data = []
        for d, c in zip(new_data, claims):
            if len(c[1]) == 3:
                d['claims'] = c[1]
                filtered_data.append(d)
        logger.warning(f"Size after dropping claims less than 3: {len(filtered_data)} => {len(filtered_data)*100/len(new_data):.2f}% of the selected {args.dataset_name} dataset")
        # save processed data
        with open(f"{args.save_load_dir}/{args.dataset_name}_run.json", "w") as f:
            json.dump(filtered_data, f, indent=4)

    else:
        raise ValueError("No corresponding dataset format")
    
    cc = ClusterClassifier(
        embed_model_name="BAAI/bge-m3",
        summary_model="Mixtral-8x7B-Instruct-v0.1-awq",
        summary_model_kwargs=args.summary_model_kwargs,
        summary_create=True if args.mode=="run" else False,
        embed_batch_size=args.embed_batch_size,
        embed_max_seq_length=embed_max_seq_length,
        embed_devices=args.embed_devices,
        topic_mode=args.topic_mode,
        summary_template=template,
        summary_instruction=instruction,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
    )

    if args.mode == "run":
        # Run a new pipeline on texts
        indexes = (
            range(args.start, args.end) if args.start > 0 else range(min(args.n_samples, len(new_data)))
        )
        text_start = f" starting from {args.start}" if args.start > 0 else ""
        logger.critical(f"Processing {len(indexes)} samples{text_start}")
        queries = [new_data[i][args.input_content] for i in indexes]
        
        _, _, summaries = cc.fit(queries)
        # show clustering results
        cc.show(interactive=True, folder=args.save_load_dir)
        cc.show(interactive=False, folder=args.save_load_dir)

        e_list = [e for e in summaries.values()]
        logger.critical(f"First {min(10, len(e_list))} example Summaries:\n{e_list[:min(10, len(e_list))]}")

        cc.save(args.save_load_dir)
        logger.info(f"Saved clusters in folder {args.save_load_dir}.")

        if args.build_hf_ds:
            build_hf_datasets(cc, new_data, None, None, args)

        ds_path = f"{args.save_load_dir}/{args.dataset_name}_{args.mode}_clusters"
        if args.topic_mode == "single_topic":
            plot_distributions(ds_path, args)
            logger.info("📊 Saved plots for demanding score and files distribution.")

    elif args.mode == "infer":
        # Run inference mode on texts using an existing pipeline
        cc.load(args.save_load_dir)
        indexes = (
            range(args.start, args.end) if args.start > 0 else range(min(args.n_samples, len(new_data)))
        )
        text_start = f" starting from {args.start}" if args.start >= 0 else ""
        logger.critical(
            f"Running inference on {len(indexes)} samples{text_start} of {args.input_dataset} using clusters in {args.save_load_dir}."
        )
        queries = [new_data[i][args.input_content] for i in indexes]

        start_time = time.time()
        cluster_labels, embeddings = cc.infer(queries, top_k=1)

        if args.build_hf_ds:
            build_hf_datasets(cc, new_data, cluster_labels, embeddings, args)

        logger.info(f"Total time is {(time.time() - start_time)/60}min")
        
        ds_path = f"{args.save_load_dir}/{args.dataset_name}_{args.mode}_clusters"
        if args.topic_mode == "single_topic":
            plot_distributions(ds_path, args)
            logger.info("📊 Saved plots for demanding score and files distribution.")

    elif args.mode == "load":
        # Load existing pipeline
        if args.build_hf_ds:
            cc.load(args.save_load_dir)
            build_hf_datasets(cc, new_data, None, None, args)
        # Plotting
        ds_path = f"{args.save_load_dir}/{args.dataset_name}_{args.mode}_clusters"
        if os.path.exists(ds_path):
            if args.topic_mode == "single_topic":
                plot_distributions(ds_path, args)
                logger.info("📊 Saved plots for demanding score and files distribution.")
        else:
            logger.warn("Using mode=load but build_hf_ds is False and no cluster datsets existed, nothing to be done.")

    logger.info("Done 🎉")
