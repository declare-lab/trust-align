import gc
import json
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .auto_ais_loader import (
    delete_autoais_model_and_tokenizer,
    get_autoais_model_and_tokenizer,
)
from .logging_config import logger
from .metrics import _run_nli_autoais
from .utils import *


def construct_data(questions: List[str], answers: List[str], raw_docs: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]], args: Any) -> List[Dict[str, Any]]:
    processed_docs: List[List[Dict[str, Any]]]
    if raw_docs and isinstance(raw_docs[0], dict): 
        processed_docs = cast(List[List[Dict[str, Any]]], [raw_docs])
    elif raw_docs and isinstance(raw_docs[0], list):
        processed_docs = cast(List[List[Dict[str, Any]]], raw_docs) 
    else:
        raise ValueError("Invalid format for raw_docs. Expected List[Dict[str, Any]] or List[List[Dict[str, Any]]].")

    
    data = []
    for idx, q in enumerate(questions):
        docs = processed_docs[idx]
        docs = _annotate_docs(docs, [answers[idx]], args)
        item = {"question": q, "answers": [answers[idx]], "docs": docs}
        data.append(item)
    save_data(data, args.eval_file)
    delete_autoais_model_and_tokenizer()
    return data

def _annotate_docs(docs: List[Dict[str, Any]], answers: List[str], args: Any) -> List[Dict[str, Any]]:
    for idx, d in enumerate(docs):
        answers_found = []
        for a in answers:
            d_text = f"{d['title']} {d['text']}"
            support = _run_nli_autoais(d_text, a, args)
            answers_found.append(support)
        d['answers_found'] = answers_found
    return docs

def save_data(data: List[Dict[str, Any]], filename: str) -> None:
    with open(f'{filename}', "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Data saved to {filename}")