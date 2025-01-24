import gc
import json

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .auto_ais_loader import (
    delete_autoais_model_and_tokenizer,
    get_autoais_model_and_tokenizer,
)
from .logging_config import logger
from .metrics import _run_nli_autoais
from .utils import *


def construct_data(questions, answers, raw_docs, args):
    data = []
    for idx, q in enumerate(questions):
        docs = raw_docs[idx]
        docs = _annotate_docs(docs, [answers[idx]], args)
        item = {"question": q, "answers": [answers[idx]], "docs": docs}
        data.append(item)
    save_data(data, args.eval_file)
    delete_autoais_model_and_tokenizer()
    return data

def _annotate_docs(docs, answers, args):
    for idx, d in enumerate(docs):
        answers_found = []
        for a in answers:
            d_text = f"{d['title']} {d['text']}"
            support = _run_nli_autoais(d_text, a, args)
            answers_found.append(support)
        d['answers_found'] = answers_found
    return docs

def save_data(data, filename):
    with open(f'{filename}', "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Data saved to {filename}")