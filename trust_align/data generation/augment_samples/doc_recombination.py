import argparse
import matplotlib.pyplot as plt
from collections import Counter
import random
import pprint
import itertools
import copy
from tqdm import tqdm
import json
from json import loads, dumps
import os
import colorlog

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.nn.functional import normalize

# Configure logging
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

def doc_to_text_tfidf(doc):
    return doc['title'] + ' ' + doc['text']

def doc_to_text_dense(doc):
    return doc['title'] + '. ' + doc['text']

class SearcherWithinDocs:
    """
    A class to perform document retrieval using either TF-IDF or dense retrieval models.

    Attributes:
        docs (list): A list of documents to be searched.
        retriever (str): The type of retriever to use ('tfidf' or a dense retriever like 'gtr').
        model (object, optional): The dense retrieval model, required if using a dense retriever.
        device (str, optional): The device to run the model on, default is 'cuda'.
        tfidf (TfidfVectorizer): The TF-IDF vectorizer used when the retriever is 'tfidf'.
        tfidf_docs (sparse matrix): The TF-IDF representations of the documents.
        embeddings (tensor): The dense embeddings of the documents when using a dense retriever.
    """

    def __init__(self, docs, retriever, model=None, device="cuda"):
        """
        Initializes the SearcherWithinDocs with the given documents and retriever.

        Parameters:
            docs (list): A list of documents to be searched.
            retriever (str): The type of retriever to use ('tfidf' or a dense retriever like 'gtr').
            model (object, optional): The dense retrieval model, required if using a dense retriever.
            device (str, optional): The device to run the model on, default is 'cuda'.

        Raises:
            NotImplementedError: If the retriever is not 'tfidf' or a dense retriever.
        """
        self.retriever = retriever
        self.docs = docs
        self.device = device
        if retriever == "tfidf":
            self.tfidf = TfidfVectorizer()
            self.tfidf_docs = self.tfidf.fit_transform([doc_to_text_tfidf(doc) for doc in docs])
        elif "gtr" in retriever:
            self.model = model
            self.embeddings = self.model.encode([doc_to_text_dense(doc) for doc in docs], device=self.device, convert_to_numpy=False, convert_to_tensor=True, normalize_embeddings=True)
        else:
            raise NotImplementedError

    def search(self, query, k):
        """
        Searches the documents and returns the top-k result document IDs based on the query.

        Parameters:
            query (list[dict]): List of valid docs that we want our invalid docs to be similar to.
            k (int): The number of top results to return.

        Returns:
            list: The top-k result document IDs.

        Raises:
            NotImplementedError: If the retriever is not 'tfidf' or a dense retriever.
        """
        if self.retriever == "tfidf":
            tfidf_query = self.tfidf.transform([doc_to_text_tfidf(doc) for doc in query])[0]
            similarities = [cosine_similarity(tfidf_doc, tfidf_query) for tfidf_doc in self.tfidf_docs]
            similarities = [arr.item() for arr in similarities]
            best_docs_ids = np.argsort(similarities)[-k:][::-1]
            return best_docs_ids
        elif "gtr" in self.retriever:
            q_embed = self.model.encode([query], device=self.device, convert_to_numpy=False, convert_to_tensor=True, normalize_embeddings=True)
            score = torch.matmul(self.embeddings, q_embed.t()).squeeze(1).detach().cpu().numpy()
            best_doc_id = np.argmax(score)
            return best_doc_id
        else:
            raise NotImplementedError

def get_all_entailment_patterns(arrays):
    """
    Given set of docs (containing at least one vlid doc), generate all possible enatilment patterns 
    that can be formed from choosing subsets of 5 from this set. This is done by generating all unique combinations of 
    length 5 from the input arrays, where each combination is represented as a bitwise OR of its elements.
    
    Args:
        arrays (list[np.array]): List of entailment patterns of docs. 
    
    Returns:
        list: List of unique combinations with non-zero elements.
    """
    all_possible_combinations = list(itertools.combinations(arrays, 5))
    all_possible_entailments = [np.bitwise_or.reduce(combination) for combination in all_possible_combinations]
    unique_entailments = np.unique(np.array(all_possible_entailments), axis=0).tolist()
    filtered_entailments = [tuple(combination) for combination in unique_entailments if not all(x == 0 for x in combination)] # remove enatilment patterns where its all 0
    return filtered_entailments

def pad_documents(indices, max_padding=15):
    """
    Pad the list of valid document indices with random invalid document indices
    to reach the desired maximum padding length.
    
    Args:
        indices (list): List of indices for valid documents.
        max_padding (int): Maximum number of documents after padding.
    
    Returns:
        list: Combined list of valid and randomly chosen invalid document indices.
    """
    padding_size = max_padding - len(indices)
    padding_indices = [200 for i in range(padding_size)]
    combined_indices = indices + padding_indices
    return combined_indices

def select_elements(tuples_list, num_selections):
    """
    Given a set of patterns, create a final list of patterns that has num_selections number of patterns and where every valid pattern appears at least once.

    Args:
        tuples_list (list[tuple]): List of tuples where each tuple contains an entailment pattern
        num_selections (int): number of combinations to make
    
    Returns:
        result (list[tuples]): List of num_selections number of tuples containing entailment patterns.  
    """
    # Ensure each element is selected at least once
    num_additional_selections = num_selections - len(tuples_list)
    if num_additional_selections > 0:
        additional_selections = random.choices(tuples_list, k=num_additional_selections)
        result = tuples_list + additional_selections
    else:
        result = tuples_list[:num_selections]
    random.shuffle(result)
    return result

def get_valid_combinations(valid_docs, k=3):
    """
    Get k combinations of valid documents. Each combination must satisfy the following conditions:
    (1) Be unique in doc idx and also doc entailment pattern. 
    (2) k combinations of valid documents must also cover as diverse a pattern as possible. For example, 
    if the set of 15 valid docs can form [1,1,0], [1,0,0], [0,1,0], then these three patterns must appear at least once.
    
    Args:
        valid_docs (list[tuple(int, tuple)]): List of tuples where each tuple contains the index of the valid doc and also the entailment pattern of that doc.
        k (int): number of combinations to make
    
    Returns:
        valid_combinations (list[list]): List of lists containing indices of valid docs, padded with 200 i there are more than 4*k invalid docs.
        valid_metadata (dataframe): Rows of dataframe containing selected pattern. Dataframe contains Indices, Bitwise OR Result, Weight
    """

    combinations = list(itertools.combinations(valid_docs, 5))
    unique_combinations = list(set(combinations))
    formatted_combinations = [(tuple(idx for idx, _ in comb), [np.array(arr) for _, arr in comb]) for comb in unique_combinations] # Convert back to numpy arrays for future bitwise_or operations

    indices_list = []
    bitwise_or_list = []
    for indices, arrays in formatted_combinations:
        if all(index == 200 for index in indices):
            continue
        bitwise_or_result = tuple(np.bitwise_or.reduce(arrays))
        indices_list.append(indices)
        bitwise_or_list.append(bitwise_or_result)

    df = pd.DataFrame({
        'Indices': indices_list,
        'Bitwise OR Result': bitwise_or_list
    })

    # Add a 'Weights' column containing the number of valid elements in 'Indices'
    df['Weights'] = df['Indices'].apply(lambda indices: sum(1 for index in indices if index != 200))

    # Filter unique patterns and remove all-zero patterns
    unique_patterns = np.unique(np.array(bitwise_or_list), axis=0).tolist()
    valid_patterns = [tuple(pattern) for pattern in unique_patterns if not all(value == 0 for value in pattern)]

    # Select a subset of patterns ensuring each pattern is chosen at least once
    num_patterns_to_select = min(df.shape[0], k)
    chosen_patterns = select_elements(valid_patterns, num_patterns_to_select)

    # Map patterns to their corresponding indices in the DataFrame
    df['Row index'] = list(df.index)
    df_grouped = df[["Row index", "Bitwise OR Result"]].groupby("Bitwise OR Result")
    pattern_to_row_indices = {key: [x[0] for x in group.values] for key, group in df_grouped}

    # Select rows based on the chosen patterns and their weights
    selected_rows = []
    for pattern in chosen_patterns:
        if pattern not in pattern_to_row_indices or not pattern_to_row_indices[pattern]: # ran out of combinaions from tht particular pattern
            available_patterns = list(pattern_to_row_indices.keys())
            pattern = random.choice(available_patterns)
        
        selected_row_indices = pattern_to_row_indices[pattern]
        selected_row_index = random.choices(selected_row_indices, weights=df.loc[selected_row_indices, 'Weights'])[0]
        selected_row_indices.remove(selected_row_index)
        if not selected_row_indices:
            del pattern_to_row_indices[pattern]
        selected_rows.append(selected_row_index)

    valid_metadata = df.loc[selected_rows].copy()
    valid_combinations = df.loc[selected_rows, "Indices"].to_list()

    return valid_combinations, valid_metadata

def pad_invalid_docs(valid_combinations, invalid_doc_indices, documents):
    """
    Fill the remaining empty slots (represented by 200) with invalid documents. Filling strategy is as follows:
    (1) From the 50 docs most similar to question, fill each combination with docs most similar to existing valid docs. Fill without replacement
    (2) Repeat for every combination

    Args:
        valid_combinations (list[list]): List of list containing valid doc idx, padded with 200 representing slots for invalid docs
        invalid_doc_indices (list): List of index of top 50 invalid docs most similar to question
        documents (list[dict]): List of dicts containing actual document content
    
    Returns:
        final_combination (list[list]): List of lists containing document idxs
    """
    final_combination = []
    for combination in valid_combinations:
        invalid_documents = [documents[idx] for idx in invalid_doc_indices]
        searcher = SearcherWithinDocs(invalid_documents, "tfidf")

        num_invalid = combination.count(200)
        # if no invalid docs to pad, skip
        if num_invalid == 0:
            final_combination.append(combination)
            continue
        
        query_docs = [documents[i] for i in combination if i!=200]
        top_invalid_ids = searcher.search(query_docs, num_invalid)
        top_invalid = [invalid_doc_indices[i] for i in top_invalid_ids]
        [invalid_doc_indices.remove(x) for x in top_invalid]
        
        new_combination = []
        for el in combination:
            if el != 200:
                new_combination.append(el)
            else:
                new_combination.append(top_invalid.pop(0))
        final_combination.append(new_combination)
    return final_combination

def get_top50_invalid_docs(invalid_doc_indices, documents):
    """
    Get top 50 invalid docs most similar to question.

    Args:
        invalid_doc_indices (list): List containing idx of invalid documents
        documents (list[dict]): List of dict containing actual document content
    
    Returns:
        filtered_invalid_doc_indices (list): Indices of top 50 invalid docs most similar to question.
    """
    invalid_doc_indices_score = []
    for idx in invalid_doc_indices:
        invalid_doc_indices_score.append((idx, documents[idx]['score']))

    invalid_doc_indices_score.sort(reverse=True, key=lambda x: x[1])
    filtered_invalid_doc_indices = [invalid_doc_indices_score[i][0] for i in range(len(invalid_doc_indices_score))][:50]
    return filtered_invalid_doc_indices

def get_invalid_combinations(invalid_docs, k=3):
    """
    Get k combinations of docs that are all invalid.

    Args:
        invalid_docs (list): List of idx of invalid documents
    
    Returns:
        invalid_combination (list[list]): List of indices of invalid documents
    """
    invalid_combination = []
    if len(invalid_docs) >= 5*k:
        for i in range(k):
            combination = random.sample(invalid_docs, 5)
            invalid_combination.append(combination)
            [invalid_docs.remove(el) for el in combination]
    elif len(invalid_docs)//5 > 0:
        for i in range(len(invalid_docs)//5 ):
            combination = random.sample(invalid_docs, 5)
            invalid_combination.append(combination)
            [invalid_docs.remove(el) for el in combination]
    else:
        logger.warning(f"You have {len(invalid_docs)} invalid docs. This is insufficient to make invalid doc combination.")
        logger.debug(f'{invalid_docs=}')
    
    return invalid_combination

def process_questions(data, k = 3, label = None):
    """
    Process each question in the data to find new combination of documents.
    
    Args:
        data (list[dict]): List of 100 documents.
    
    Returns:
        dict: Dictionary containing the combinations of document indices for each question.
    """
    results = {}
    base_data_stats = {'invalid qns':0,
                       'less than 15 valid docs':0,
                       'more than 15 valid docs': 0}
    metadata = {}
    question_metadata = []
    less_than_k, eq_k = 0, 0

    for question_idx in tqdm(range(len(data)), desc=f"Processing {label} questions"):
        q_metadata = {}

        sample = data[question_idx]
        question = sample["question"]
        documents = sample["docs"]
        
        # Extract answers_found for each document in the current question
        all_answers_found = np.array([doc['answers_found'] for doc in documents])

        # Identify valid and invalid document indices based on answers_found
        valid_doc_indices = [idx for idx in np.where(~np.all(all_answers_found == 0, axis=1))[0]]
        invalid_doc_indices = [idx for idx in np.where(np.all(all_answers_found == 0, axis=1))[0]]
        
        if len(invalid_doc_indices) < 4*k:
            logger.warning(f"You have less than {4*k} invalid docs. # invalid docs: {len(invalid_doc_indices)}. No invalid docs will be added. Only pad with {len(invalid_doc_indices)//k} 200.")

        if len(valid_doc_indices) == 0:
            base_data_stats['invalid qns'] += 1
            continue
        elif len(valid_doc_indices) < 15:
            base_data_stats['less than 15 valid docs'] += 1
            doc_indices_list = valid_doc_indices.copy() 
        else:
            base_data_stats['more than 15 valid docs'] += 1
            doc_indices_list = random.sample(valid_doc_indices, 15)
        
        doc_indices_list += [200 for i in range(min(k, len(invalid_doc_indices)//k))] #  do not add any invalid docs in case there are less than 4*k invalid docs
        
        if len(invalid_doc_indices) < 4*k:
            logger.debug(f"{doc_indices_list=}")

        # Filter answers_found for the selected document indices
        filtered_answers_found = [all_answers_found[idx] if idx != 200 else np.zeros(len(all_answers_found[valid_doc_indices[0]]), dtype=int) for idx in doc_indices_list]
        doc_list = [(idx, tuple(arr)) for idx, arr in zip(doc_indices_list, filtered_answers_found)]
        
        valid_combinations, valid_metadata = get_valid_combinations(doc_list, k)
        
        if len(invalid_doc_indices) < 4*k:
            logger.debug(f"{valid_combinations=}")

        assert len(valid_combinations) <= k, "You have more combinations than specified."

        if (len(valid_doc_indices) == 1) or (len(valid_doc_indices) == 2):
            less_than_k+=1
            assert len(valid_combinations) < k, "error"
        else:
            eq_k+=1

        # select top 50 docs with highest similarity to question
        top50_invalid_doc_indices = get_top50_invalid_docs(invalid_doc_indices, documents)
        if len(invalid_doc_indices) < 4*k:
            logger.debug(f"{top50_invalid_doc_indices=}")
    
        final_combination_valid = pad_invalid_docs(valid_combinations, top50_invalid_doc_indices, documents)
        if len(invalid_doc_indices) < 4*k:
            logger.debug(f"{final_combination_valid=}")
        final_combination_invalid = get_invalid_combinations(top50_invalid_doc_indices, k)
        results[question_idx] = final_combination_valid + final_combination_invalid

        # log stats about data
        valid_metadata_str = valid_metadata.to_json(orient="table")
        valid_metadata = loads(valid_metadata_str)
        valid_answers_found = [x['Bitwise OR Result'] for x in valid_metadata['data']]

        invalid_answers_found = []
        for combination in final_combination_invalid:
            filtered_answers_found = np.array([all_answers_found[i] for i in combination])
            invalid_answers_found.append(np.bitwise_or.reduce(filtered_answers_found).tolist())

        q_metadata["question_idx"] = question_idx
        q_metadata["final_combination_valid"] = {'combination':convert_to_python_int(final_combination_valid), 'answers_found':convert_to_python_int(valid_answers_found)}
        q_metadata["final_combination_invalid"] = {'combination':convert_to_python_int(final_combination_invalid), 'answers_found':convert_to_python_int(invalid_answers_found)}
        q_metadata["valid_doc_indices"] = convert_to_python_int(valid_doc_indices)
        q_metadata["num_valid_doc"] = len(valid_doc_indices)
        q_metadata["invalid_doc_indices"] = convert_to_python_int(invalid_doc_indices)
        q_metadata["num_invalid_doc"] = len(invalid_doc_indices)
        question_metadata.append(q_metadata)
    
    metadata['base_data_stats'] = base_data_stats
    metadata['question_metadata'] = question_metadata
    metadata['extra_stats'] = {'num_less_than_k' : less_than_k, 'num_eq_k' : eq_k}

    # converted_metadata = convert_to_python_int(metadata)
    # print(json.dumps(converted_metadata, indent=4))
    # print(results)

    return results, metadata

def generate_combinations_data(data, output_idx_set, label):
    """
    Generate a new dataset with combinations of selected documents and their corresponding patterns.

    Args:
        data (list): List of questions and their corresponding documents.
        output_idx_set (dict): Dictionary containing the combinations of document indices for each question.

    Returns:
        list: New dataset with selected document combinations for each question.
    """
    combined_data = []

    for question_idx, doc_combinations in output_idx_set.items():
        original_sample = data[question_idx]
        
        for combination in doc_combinations:
            new_sample = copy.deepcopy(original_sample)
            selected_docs = [original_sample['docs'][index] for index in combination]
            pattern = np.bitwise_or.reduce([original_sample['docs'][index]['answers_found'] for index in combination])
            
            logger.debug(f'Pattern for Question {question_idx}: {pattern}')
            new_sample['docs'] = selected_docs
            combined_data.append(new_sample)
        
        # logger.info(f"Processed combinations for Question {question_idx}")
    
    logger.info(f'Total number of new samples: {len(combined_data)}')
    return combined_data

def save_data_to_json(data, directory_path, file_name, is_metadata = False):
    """
    Save the list of dictionaries to a JSON file in the specified directory.

    Args:
        data (list): List of dictionaries to be saved.
        directory_path (str): Path to the directory where the file will be saved.
        file_name (str): Name of the JSON file.

    Returns:
        str: The path to the saved JSON file.
    """
    os.makedirs(directory_path, exist_ok=True)
    file_path = os.path.join(directory_path, file_name)

    if is_metadata:
        # Convert data to be JSON serializable
        try:
            serializable_data = make_serializable(data)
        except (TypeError, OverflowError) as e:
            logger.error(f"Error serializing data: {e}")
            return False
        
        # Write data to file with error handling
        try:
            with open(file_path, 'w') as json_file:
                json.dump(serializable_data, json_file, indent=4, cls = NumpyEncoder)
        except (TypeError, IOError) as e:
            logger.error(f"Error writing to file: {e}")
            return False
        
        # Verify file integrity by reading it back
        try:
            with open(file_path, 'r') as json_file:
                loaded_data = json.load(json_file)
                # print(loaded_data == serializable_data) 
        except (IOError, json.JSONDecodeError, AssertionError) as e:
            logger.error(f"Error verifying file integrity: {e}")
            return False
        logger.info(f'Data saved to {file_path}')
        return True
    else:
        try:
            with open(file_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)
        except IOError as e:
            logger.error(f"Error writing to file: {e}")
            return False
        logger.info(f'Data saved to {file_path}')
        return True

def convert_to_python_int(data):
    if isinstance(data, list):
        return [convert_to_python_int(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_to_python_int(value) for key, value in data.items()}
    elif isinstance(data, (np.integer)):
        return int(data)
    elif isinstance(data, (np.floating)):
        return float(data)
    else:
        return data
    
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return super(NumpyEncoder, self).default(obj)

def make_serializable(data):
    if isinstance(data, dict):
        return {k: make_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_serializable(item) for item in data]
    elif isinstance(data, (np.integer, np.floating, np.ndarray, np.generic)):
        return json.loads(json.dumps(data, cls=NumpyEncoder))
    return data

def check_types(data, parent_key=''):
    """
    Recursively checks and prints the types of all items in a nested dictionary or list.
    
    Args:
        data: The nested dictionary or list to check.
        parent_key (str): The key leading to the current data (for nested structures).
    """
    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            print(f"Key: {full_key}, Type: {type(value)}")
            assert not isinstance(value, np.int64), "Variable should not be int64"
            check_types(value, full_key)
            
    elif isinstance(data, list):
        for index, item in enumerate(data):
            full_key = f"{parent_key}[{index}]"
            print(f"Key: {full_key}, Type: {type(item)}")
            assert not isinstance(item, np.int64), "Variable should not be int64"
            check_types(item, full_key)
    else:
        assert not isinstance(data, np.int64), "Variable should not be int64"
        print(f"Key: {parent_key}, Type: {type(data)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document selection")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset (ASQA, QAMPARI, ELI5)")
    parser.add_argument("--data_file", type=str, default=None, help="Path to data file")
    parser.add_argument("--output_file", type=str, default=None, help="Path to output file")
    parser.add_argument("--metadata_file", type=str, default=None, help="Path to metadata file")
    
    args = parser.parse_args()

    logger.info(f"Arguments received: {args}")

    random.seed(0)
    np.random.seed(0)
    
    with open(args.data_file) as f:
        data = json.load(f)

    final_combination, metadata = process_questions(data, k = 10, label = args.dataset_name)
    logger.info(f"# of invalid questions: {metadata['base_data_stats']['invalid qns']}, # of qns with less than 15 docs: {metadata['base_data_stats']['less than 15 valid docs']}, # of qns with more than 15 docs: {metadata['base_data_stats']['more than 15 valid docs']}")
    combined_data = generate_combinations_data(data, final_combination, args.dataset_name)
    # check_types(metadata)
    
    if args.output_file:
        directory_path = os.path.dirname(args.output_file)
        file_name = os.path.basename(args.output_file)
        save_data_to_json(combined_data, directory_path, file_name)
        metadata_file_name = os.path.basename(args.metadata_file)
        save_data_to_json(metadata, directory_path, metadata_file_name, is_metadata=True)