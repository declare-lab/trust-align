import json
import os
import random
import re
import string
from argparse import ArgumentParser
from itertools import combinations
from typing import List

import numpy as np
from nltk import sent_tokenize

REJECT_RESPONSES = [
    "I apologize, but I couldn't find an answer to your question in the search results."
]

def load_json_files_from_folder(folder_path):
    """
    Load JSON and JSONL files from the specified folder and categorize them into datasets.

    Args:
        folder_path (str): The path to the folder containing the JSON files.

    Returns:
        Tuple[List, List, List, List, List]: Lists containing data for ASQA, ELI5, QAMPARI datasets and their new answers.
    """
    asqa, eli5, qampari = [], [], []
    asqa_ans, eli5_ans = [], []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith('.json') and "error_instruction" in file_name:
            print(f"Loading JSON file: {file_path}")
            with open(file_path, 'r') as file:
                file_data = json.load(file)
                if "asqa" in file_name:
                    asqa.extend(file_data)
                elif "eli5" in file_name:
                    eli5.extend(file_data)
                elif "qampari" in file_name:
                    qampari.extend(file_data)
        elif file_name.endswith('.jsonl') and "new_answers" in file_name:
            print(f"Loading JSONL file: {file_path}")
            with open(file_path, 'r') as file:
                file_data = [json.loads(line) for line in file]
                if "asqa" in file_name:
                    asqa_ans.extend(file_data)
                elif "eli5" in file_name:
                    eli5_ans.extend(file_data)
    return asqa, eli5, qampari, asqa_ans, eli5_ans

def normalize_answer(s: str) -> str:
    """
    Normalize the answer string by removing articles, punctuation, and extra whitespace.

    Args:
        s (str): The input string.

    Returns:
        str: The normalized string.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_punctuation(text):
        return "".join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lower(s))))

def white_space_fix(text: str) -> str:
    """Fix whitespace issues in a string."""
    return " ".join(text.split())

def remove_citations(sentence: str) -> str:
    """Remove citation markers from a sentence."""
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sentence)).replace(" |", "").replace("]", "")

def insert_reference(sentence: str, references: List[int]) -> str:
    """
    Insert citation references into a sentence.

    Args:
        sentence (str): The sentence to insert references into.
        references (List[int]): A list of reference indices.

    Returns:
        str: The sentence with references inserted.
    """
    punctuations = "!.;)"
    pattern = fr"(.*)([{punctuations}]).*$"
    reference_str = "".join(f"[{ref}]" for ref in references)
    matched = re.search(pattern, sentence)

    if matched:
        return matched.group(1).strip(string.punctuation) + " " + reference_str + matched.group(2)
    else:
        return sentence.strip(string.punctuation) + " " + reference_str + "."

def extract_docs(ans_index: set, answers_found_list: List[List[int]], scores: List[float], max_cite: int = 1) -> List[int]:
    """
    Extract document indices that cover the answer indices.

    Args:
        ans_index (set): Set of answer indices.
        answers_found_list (List[List[int]]): List of documents' answers_found arrays.
        scores (List[float]): List of document scores.
        max_cite (int): Maximum number of citations.

    Returns:
        List[int]: List of document indices.
    """
    def calculate_coverage(answers_found, ans_index):
        coverage = {idx for idx in ans_index if idx < len(answers_found) and answers_found[idx] == 1}
        return coverage

    doc_coverage = [
        (i, calculate_coverage(answers_found, ans_index), scores[i])
        for i, answers_found in enumerate(answers_found_list)
    ]
    doc_coverage = [doc for doc in doc_coverage if doc[1]]
    doc_coverage.sort(key=lambda x: (-len(x[1]), -x[2]))

    cited_docs = set()
    for r in range(1, len(answers_found_list) + 1):
        for comb in combinations(doc_coverage, r):
            combined_coverage = set.union(*(doc[1] for doc in comb))
            if combined_coverage == ans_index:
                if len(cited_docs) < max_cite:
                    cited_docs.update(doc[0] for doc in comb)
        if cited_docs:
            break
    return sorted(cited_docs) if cited_docs else []

def process_data(data_list: List[dict], max_cite: int, score_key: str) -> List[dict]:
    """
    Process the dataset to generate positive data examples.

    Args:
        data_list (List[dict]): The dataset to process.
        max_cite (int): Maximum number of citations per answer.
        score_key (str): The key to use for document scores.

    Returns:
        List[dict]: The list of processed data examples.
    """
    positive_data = []
    for item in data_list:
        if item.get('new_answer'):
            sents = sent_tokenize(item['new_answer'])
            union_idx = np.nonzero(np.bitwise_or.reduce([doc['answers_found'] for doc in item['docs']]))[0].tolist()
            mapper = {str(i+1): union_idx[i] for i in range(len(union_idx))}

            try:
                citations = re.findall(r"\[.+?(\d+)", item['new_answer'])
                ans_idx_set = {mapper[r] for r in citations if r in mapper}
            except KeyError as e:
                print(f"KeyError: {e}, Question: {item['question']}")
                continue

            cited_sents = []
            for sent in sents:
                ans_idx = {mapper[r] for r in re.findall(r"\[.+?(\d+)", sent) if r in mapper}
                if ans_idx and ans_idx.issubset(union_idx):
                    doc_idxes = extract_docs(
                        ans_idx,
                        [doc['answers_found'] for doc in item['docs']],
                        [doc[score_key] for doc in item['docs']],
                        max_cite
                    )
                    if doc_idxes:
                        sent = remove_citations(sent).strip()
                        sent = insert_reference(sent, [idx+1 for idx in doc_idxes])
                        cited_sents.append(sent)
            response = " ".join(cited_sents)
            positive_data.append({"question": item['question'], "resp": response, "prompt": item['prompt']})
        else:
            response = random.choice(REJECT_RESPONSES)
            positive_data.append({"question": item['question'], "resp": response, "prompt": item['prompt']})
    return positive_data

def process_qampari_data(data_list: List[dict], max_cite: int) -> List[dict]:
    """
    Process the QAMPARI dataset to generate positive data examples.

    Args:
        data_list (List[dict]): The QAMPARI dataset to process.
        max_cite (int): Maximum number of citations per answer.

    Returns:
        List[dict]: The list of processed data examples.
    """
    positive_data = []
    for item in data_list:
        union_idx = np.nonzero(np.bitwise_or.reduce([doc['answers_found'] for doc in item['docs']]))[0].tolist()
        if union_idx:
            sents = [item['answers'][idx][0].strip() for idx in union_idx]
            cited_sents = []
            for i, sent in enumerate(sents):
                ans_idx = {union_idx[i]}
                doc_idxes = extract_docs(
                    ans_idx,
                    [doc['answers_found'] for doc in item['docs']],
                    [doc['score'] for doc in item['docs']],
                    max_cite
                )
                if doc_idxes:
                    sent = insert_reference(sent, [idx+1 for idx in doc_idxes]).strip('.').strip()
                    cited_sents.append(sent)
            response = ", ".join(cited_sents) + "."
            positive_data.append({"question": item['question'], "resp": response, "prompt": item['prompt']})
        else:
            response = random.choice(REJECT_RESPONSES)
            positive_data.append({"question": item['question'], "resp": response, "prompt": item['prompt']})
    return positive_data

def main():
    parser = ArgumentParser(description="Process datasets and generate positive data examples.")
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input data folder.')
    args = parser.parse_args()

    # Load datasets
    asqa_data, eli5_data, qampari_data, asqa_new_ans, eli5_new_ans = load_json_files_from_folder(args.input_folder)

    print(f'Number of ASQA data: {len(asqa_data)}')
    print(f'Number of ELI5 data: {len(eli5_data)}')
    print(f'Number of QAMPARI data: {len(qampari_data)}')
    print(f'Number of ASQA new answers: {len(asqa_new_ans)}')
    print(f'Number of ELI5 new answers: {len(eli5_new_ans)}')

    # Incorporate new answers
    for line in asqa_new_ans:
        asqa_data[line['index']]['new_answer'] = line['new_answer']
    for line in eli5_new_ans:
        eli5_data[line['index']]['new_answer'] = line['new_answer']

    # Process datasets
    asqa_positive_data = process_data(asqa_data, max_cite=2, score_key='score')
    eli5_positive_data = process_data(eli5_data, max_cite=2, score_key='rec_score')
    qampari_positive_data = process_qampari_data(qampari_data, max_cite=1)

    # Save processed data
    output_files = {
        'asqa_positive_data.json': asqa_positive_data,
        'eli5_positive_data.json': eli5_positive_data,
        'qampari_positive_data.json': qampari_positive_data
    }

    for filename, data in output_files.items():
        output_path = os.path.join(args.input_folder, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(data)} records to {output_path}")

if __name__ == '__main__':
    main()
