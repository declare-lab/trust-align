import logging

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from trust_eval.utils import get_max_memory

logger = logging.getLogger(__name__)

# Globals to store the shared instances
_autoais_model = None
_autoais_tokenizer = None

def get_autoais_model_and_tokenizer(args):
    global _autoais_model, _autoais_tokenizer

    if _autoais_model is None:
        logger.info("Loading AutoAIS model...")
        _autoais_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.autoais_model,
            torch_dtype=torch.bfloat16,
            max_memory=get_max_memory(),
            device_map="auto"
        )
        _autoais_tokenizer = AutoTokenizer.from_pretrained(
            args.autoais_model,
            use_fast=False
        )
        logger.info("AutoAIS model loaded successfully.")
    
    return _autoais_model, _autoais_tokenizer
