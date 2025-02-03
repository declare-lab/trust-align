import argparse
import json
import logging
import os
import re
import string
from typing import Any, Dict, List

import numpy as np
import torch
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .auto_ais_loader import get_autoais_model_and_tokenizer
from .logging_config import logger
from .searcher import SearcherWithinDocs
from .utils import get_max_memory, remove_citations


def post_hoc_cite(data: List[Dict[str, Any]], args: Any) -> List[Dict[str, Any]]:
    logger.info("Performing post hoc citation...")

    new_data = []
    external = json.load(open(args.external_docs)) if args.external_docs is not None else None
    
    # Load retrieval model
    if "gtr" in args.posthoc_retriever:
        logger.info("Loading gtr-t5-large...")
        gtr_model = SentenceTransformer(f'sentence-transformers/{args.posthoc_retriever}', device=args.posthoc_retriever_device)
    
    elif "nli" in args.posthoc_retriever:
        logger.info("Loading t5_xxl_true_nli_mixture...")
        autoais_model, autoais_tokenizer = get_autoais_model_and_tokenizer(args)
    
    for idx, item in enumerate(tqdm(data)):
        doc_list = item['docs']
        
        if args.external_docs is not None:
            assert external is not None
            assert external[idx]['question'] == item['question']
            doc_list = external[idx]['docs'][:args.ndoc]

        if "gtr" in args.posthoc_retriever:
            searcher = SearcherWithinDocs(doc_list, args.posthoc_retriever, model=gtr_model, device=args.posthoc_retriever_device)
        elif "nli" in args.posthoc_retriever:
            searcher = SearcherWithinDocs(doc_list, args.posthoc_retriever, model=autoais_model, tokenizer=autoais_tokenizer, device=args.posthoc_retriever_device)

        logger.debug(f'{item["output"]=}')
        
        # Pre-process output
        output = item["output"].strip().split("\n")[0] # Remove new lines and content after
        output = item["output"].replace("<|im_end|>", "")
        if "qampari" in args.data_type:
            sents = [item['question'] + ' ' + x.strip() for x in item['output'].rstrip(".").split(",")]
        else:
            sents = sent_tokenize(output)

        # Add citations
        new_output = ""
        for sent in sents:
            original_ref = [int(r[1:])-1 for r in re.findall(r"\[\d+", sent)] # Get the existing citations in the sentence

            if len(original_ref) == 0 or args.overwrite:
                logger.debug("\n-----")
                logger.debug("Original sentence:", sent)
                logger.debug("Original ref:", original_ref)

                sent = remove_citations(sent)
                best_doc_id = searcher.search(sent)
                sent = f"[{best_doc_id+1}] " + sent

                logger.debug("New ref:", best_doc_id)
                logger.debug("New sentence:", sent)

                if "qampari" in args.data_type:
                    new_output += sent.replace(item['question'], '').strip() + ", "
                else:
                    new_output += sent + " "
            else:
                if "qampari" in args.data_type:
                    new_output += sent.replace(item['question'], '').strip() + ", "
                else:
                    new_output += sent + " "
   
        item['output'] = new_output.rstrip().rstrip(",")
        logger.debug("Final output: " + item['output'])
        item['docs'] = doc_list
        new_data.append(item)

    return new_data