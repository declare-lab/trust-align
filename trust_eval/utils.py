import re
import string
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizers import MistralTokenizer

from .logging_config import logger

AnyTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast,
                     MistralTokenizer]

def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def remove_citations(sent: str) -> str:
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")


def get_max_memory() -> Dict[int, str]:
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory_str = f"{free_in_GB - 3}GB"  
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory_str for i in range(n_gpus)}
    return max_memory


def load_model(
    model_name_or_path: str,
    dtype: torch.dtype = torch.float16,
    int8: bool = False,
    lora_path: Optional[str] = None,
    reserve_memory: int = 10
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    # Load a huggingface model and tokenizer
    # dtype: torch.float16 or torch.bfloat16
    # int8: whether to use int8 quantization
    # reserve_memory: how much memory to reserve for the model on each gpu (in GB)

    # Load the FP16 model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info(f"Loading {model_name_or_path} in {dtype}...")
    if int8:
        logger.warning("Use LLM.int8")
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


def load_vllm(model_name_or_path: str, args: Any, dtype: str="bfloat16") -> Tuple[LLM, AnyTokenizer, SamplingParams]:
    from vllm import LLM, SamplingParams
    logger.info(f"Loading {model_name_or_path} in {dtype}...")
    start_time = time.time()
    model = LLM(
        model_name_or_path, 
        dtype=dtype,
        gpu_memory_utilization=0.9,
        seed=args.seed,
        max_model_len=args.max_length,
    )
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_new_tokens)
    logger.info("Finish loading in %.2f sec." % (time.time() - start_time))

    # Load the tokenizer
    tokenizer = model.get_tokenizer()
    tokenizer.padding_side = "left" # type: ignore[union-attr]
    
    return model, tokenizer, sampling_params

def make_doc_prompt(doc:dict, doc_id: int, doc_prompt: str, use_shorter: Optional[str]=None) -> str:
    # For doc prompt:
    # - {ID}: doc id (starting from 1)
    # - {T}: title
    # - {P}: text
    # use_shorter: None, "summary", or "extraction"

    text = doc['text']
    if use_shorter is not None:
        text = doc[use_shorter]
    return doc_prompt.replace("{T}", doc["title"]).replace("{P}", text).replace("{ID}", str(doc_id+1))



def get_shorter_text(docs: List[Dict[str, Any]], ndoc: int, key: str) -> List[Dict[str, Any]]:
    doc_list: List[Dict[str, Any]] = []
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



def make_demo(item: Dict[str, Any], 
              prompt: str, 
              ndoc: int, 
              doc_prompt: str, 
              instruction: str="", 
              use_shorter: Optional[str]=None, 
              test: bool=False) -> str:
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
            doc_list = get_shorter_text(item["docs"], ndoc, use_shorter) if use_shorter is not None else item["docs"][:ndoc]
            text = "".join([make_doc_prompt(doc, doc_id, doc_prompt, use_shorter=use_shorter) for doc_id, doc in enumerate(doc_list)])
            prompt = prompt.replace("{D}", text)

    if not test:
        answer = "\n" + "\n".join(item["answer"]) if isinstance(item["answer"], list) else item["answer"]
        prompt = prompt.replace("{A}", "").rstrip() + answer
    else:
        prompt = prompt.replace("{A}", "").rstrip() # remove any space or \n

    return prompt

def format_document(doc: Dict[str, Any]) -> str:
        """Format document for AutoAIS."""

        if "sent" in doc:
            # QA-extracted docs
            return "Title: %s\n%s" % (doc['title'], doc['sent'])
        else:
            return "Title: %s\n%s" % (doc['title'], doc['text'])
