import gc
import logging
from typing import Any, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from trust_eval.utils import get_max_memory

logger = logging.getLogger(__name__)

# Globals to store the shared instances
_autoais_model = None
_autoais_tokenizer = None

def get_autoais_model_and_tokenizer(args: Any) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
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


def delete_autoais_model_and_tokenizer() -> None:
    """Delete the AutoAIS model and tokenizer to free GPU memory."""
    global _autoais_model, _autoais_tokenizer

    if _autoais_model is not None or _autoais_tokenizer is not None:
        logger.info("Deleting AutoAIS model and tokenizer...")

        _autoais_model = None
        _autoais_tokenizer = None

        gc.collect()
        torch.cuda.empty_cache()

        logger.info("AutoAIS model and tokenizer deleted successfully.")
    else:
        logger.info("AutoAIS model and tokenizer are already deleted.")