import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import json
import os
import re
import string
import time
from dataclasses import dataclass
from typing import List, Literal, Optional

import torch


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")


def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-3}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory


def make_doc_prompt(doc, doc_id, doc_prompt, use_shorter=None):
    # For doc prompt:
    # - {ID}: doc id (starting from 1)
    # - {T}: title
    # - {P}: text
    # use_shorter: None, "summary", or "extraction"

    text = doc['text']
    if use_shorter is not None:
        text = doc[use_shorter]
    return doc_prompt.replace("{T}", doc["title"]).replace("{P}", text).replace("{ID}", str(doc_id+1))


def get_shorter_text(item, docs, ndoc, key):
    doc_list = []
    for item_id, item in enumerate(docs):
        if key not in item:
            if len(doc_list) == 0:
                # If there aren't any document, at least provide one (using full text)
                item[key] = item['text']
                doc_list.append(item)
            logger.warn(f"No {key} found in document. It could be this data do not contain {key} or previous documents are not relevant. This is document {item_id}. This question will only have {len(doc_list)} documents.")
            break
        if "irrelevant" in item[key] or "Irrelevant" in item[key]:
            continue
        doc_list.append(item)
        if len(doc_list) >= ndoc:
            break
    return doc_list


def make_demo(item, prompt, ndoc=None, doc_prompt=None, instruction=None, use_shorter=None, test=False):
    # For demo prompt
    # - {INST}: the instruction
    # - {D}: the documents
    # - {Q}: the question
    # - {A}: the answers
    # ndoc: number of documents to put in context
    # use_shorter: None, "summary", or "extraction"

    prompt = prompt.replace("{INST}", instruction).replace("{Q}", item['question'])
    if "{D}" in prompt:
        if ndoc == 0:
            prompt = prompt.replace("{D}\n", "") # if there is no doc we also delete the empty line
        else:
            doc_list = get_shorter_text(item, item["docs"], ndoc, use_shorter) if use_shorter is not None else item["docs"][:ndoc]
            text = "".join([make_doc_prompt(doc, doc_id, doc_prompt, use_shorter=use_shorter) for doc_id, doc in enumerate(doc_list)])
            prompt = prompt.replace("{D}", text)

    if not test:
        answer = "\n" + "\n".join(item["answer"]) if isinstance(item["answer"], list) else item["answer"]
        prompt = prompt.replace("{A}", "").rstrip() + answer
    else:
        prompt = prompt.replace("{A}", "").rstrip() # remove any space or \n

    return prompt


def load_model(model_name_or_path, dtype=torch.float16, int8=False, lora_path=None, reserve_memory=10):
    # Load a huggingface model and tokenizer
    # dtype: torch.float16 or torch.bfloat16
    # int8: whether to use int8 quantization
    # reserve_memory: how much memory to reserve for the model on each gpu (in GB)

    # Load the FP16 model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info(f"Loading {model_name_or_path} in {dtype}...")
    if int8:
        logger.warn("Use LLM.int8")
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map='auto',
        torch_dtype=dtype,
        max_memory=get_max_memory(),
        load_in_8bit=int8,
    )
    
    if lora_path:
        from peft import PeftModel
        logger.info(f"Loading PeftModel {model_name_or_path} in {dtype}...")
        model = PeftModel.from_pretrained(model, model_id=lora_path)
    
    logger.info("Finish loading in %.2f sec." % (time.time() - start_time))

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

    # Fix OPT bos token problem in HF
    if "opt" in model_name_or_path:
        tokenizer.bos_token = "<s>"
    tokenizer.padding_side = "left"

    return model, tokenizer


def load_vllm(model_name_or_path, args, dtype=torch.bfloat16):
    from vllm import LLM, SamplingParams
    logger.info(f"Loading {model_name_or_path} in {dtype}...")
    start_time = time.time()
    model = LLM(
        model_name_or_path, 
        dtype=dtype,
        gpu_memory_utilization=0.9,
        seed=args.seed,
        max_seq_len_to_capture=args.max_length,
    )
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_new_tokens)
    logger.info("Finish loading in %.2f sec." % (time.time() - start_time))

    # Load the tokenizer
    tokenizer = model.get_tokenizer()
    
    tokenizer.padding_side = "left"
    
    return model, tokenizer, sampling_params


@dataclass
class RUN_Config:
    # General settings
    prompt_file: Optional[str] = None  # Path to the prompt file
    eval_file: Optional[str] = None  # Path to the evaluation file
    output_dir: Optional[str] = None  # Output directory for model's output
    quick_test: Optional[int] = None  # Number of examples for quick testing

    # ICL settings
    ndoc: int = 5  # Number of documents
    shot: int = 2  # Number of ICL demonstrations
    seed: int = 42  # Random seed
    no_doc_in_demo: bool = False  # Whether to remove documents in demo
    fewer_doc_in_demo: bool = False  # Whether to use fewer documents in demo
    ndoc_in_demo: Optional[int] = None  # Number of documents in demo when using fewer docs
    no_demo: bool = False  # Whether to disable demos

    # Model and naming
    eval_type: Literal["em", "em@5", "cm"] = None  # evaluation type for different dataset format
    model: str = "gpt2"  # Model to use
    openai_api: bool = False  # Whether to use OpenAI API
    azure: bool = False  # Whether to use Azure OpenAI API
    lora_path: Optional[str] = None  # Path to LoRA training checkpoint
    vllm: bool = False  # Whether to use vllm for acceleration

    # Decoding settings
    temperature: float = 0.5  # Temperature for decoding
    top_p: float = 1.0  # Nucleus sampling top-p
    max_new_tokens: int = 300  # Maximum number of new tokens to generate
    max_length: int = 2048  # Maximum length for model input
    num_samples: int = 1  # Number of samples for multiple answers
    
    def update_from_dict(self, config_dict: dict):
        """
        Update the Config dataclass fields from a dictionary.
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class EVAL_Config:
    # Eval settings
    output_path: str  # output file path for evaluation result (required)
    # Optional flags and settings
    citations: bool = False  # Evaluate using citation data
    at_most_citations: int = 3  # Maximum number of documents for citation evaluation
    claims_nli: bool = False  # Use claims for ELI5

    def update_from_dict(self, config_dict: dict):
        """
        Update the Config dataclass fields from a dictionary.
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)