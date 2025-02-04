"""
Adapted with gratitude from https://github.com/princeton-nlp/ALCE/blob/main/retrieval.py
"""

import csv
import gc
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .logging_config import logger


def retrieve(queries: Union[str, List[str]], top_k: int=5) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    return gtr_wiki_query(queries, top_k=5)

def gtr_wiki_query(queries: Union[str, List[str]], top_k: int=5) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """
    Handle both single and batch queries for GTR Wikipedia search.
    
    Args:
        queries (str or list): A single query (str) or a list of queries (list).
        top_k (int): Number of top results to retrieve.
        
    Returns:
        list or dict: Results for the queries.
    """
    # Ensure queries is a list
    if isinstance(queries, str):
        queries = [queries]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading GTR encoder...")
    encoder = SentenceTransformer("sentence-transformers/gtr-t5-xxl", device=device)

    # Encode the queries
    with torch.inference_mode():
        query_embeddings_raw = encoder.encode(
            queries, batch_size=4, show_progress_bar=True, normalize_embeddings=True
        )
        query_embeddings = torch.tensor(query_embeddings_raw, dtype=torch.float16, device="cpu")
    
    # Load Wikipedia corpus
    DOCS_PICKLE = os.environ.get("DOCS_PICKLE", "docs.pkl")
    if os.path.exists(DOCS_PICKLE):
        print("Loading docs from pickle file...")
        with open(DOCS_PICKLE, "rb") as f:
            docs = pickle.load(f)
    else:
        print("Processing CSV file to construct docs...")
        docs = []
        DPR_WIKI_TSV = os.environ.get("DPR_WIKI_TSV", "psgs_w100.tsv")
        with open(DPR_WIKI_TSV) as f:
            total_rows = sum(1 for _ in f) - 1 

        with open(DPR_WIKI_TSV) as f:
            reader = csv.reader(f, delimiter="\t")
            _ = next(reader) # Skip the header row
            for row in tqdm(reader, desc="Processing wikipedia file", total=total_rows):
                docs.append(row[2] + "\n" + row[1])
        
        # Save `docs` for future use
        print("Saving docs to pickle file...")
        with open(DOCS_PICKLE, "wb") as f:
            pickle.dump(docs, f)

    # Load or build GTR embeddings
    GTR_EMB = os.environ.get("GTR_EMB", "gtr_wikipedia_index.pkl")
    if not os.path.exists(GTR_EMB):
        logger.info("gtr embeddings not found, building...")
        embs = gtr_build_index(encoder, docs)  
        with open(GTR_EMB, "wb") as f:
            pickle.dump(embs, f)
    else:
        logger.info("gtr embeddings found, loading...")
        with open(GTR_EMB, "rb") as f:
            embs = pickle.load(f)

    del encoder  # Free GPU memory

    gtr_emb = torch.tensor(embs, dtype=torch.float16, device=device)

    # Perform retrieval for each query
    logger.info("Running GTR retrieval...")
    query_embeddings = query_embeddings.to(device)
    results = []
    for query_idx in range(len(queries)):
        query_embedding = query_embeddings[query_idx]
        scores = torch.matmul(gtr_emb, query_embedding)  # Compare query with document embeddings
        score, idx = torch.topk(scores, top_k)  # Get top-k results

        # Prepare results for the current query
        query_results = []
        for i in range(idx.size(0)):
            title, text = docs[idx[i].item()].split("\n")
            query_results.append({
                "id": str(idx[i].item() + 1),
                "title": title,
                "text": text,
                "score": score[i].item()
            })
        results.append(query_results)
    
    del gtr_emb, query_embeddings # Free GPU memory
    gc.collect()
    torch.cuda.empty_cache()

    # Return a single result if input was a single query
    return results[0] if len(results) == 1 else results

def gtr_build_index(encoder: SentenceTransformer, docs: List[str]) -> np.ndarray:
    with torch.inference_mode():
        embs = encoder.encode(docs, batch_size=4, show_progress_bar=True, normalize_embeddings=True)
        embs = embs.astype("float16")

    GTR_EMB = os.environ.get("GTR_EMB", "gtr_wikipedia_index.pkl")
    with open(GTR_EMB, "wb") as f:
        pickle.dump(embs, f)
    return embs
