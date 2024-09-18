import argparse
import json
import os
import random
import re
import string
import time
from math import ceil
from pathlib import Path

import colorlog
import numpy as np
import openai
import tiktoken
import torch
import yaml
from datasets import load_from_disk
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureChatOpenAI
from nltk import sent_tokenize
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.trainer_utils import set_seed

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


def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-6}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory


def load_model(model_name_or_path, dtype=torch.bfloat16, int8=False):
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
    logger.info("Finish loading in %.2f sec." % (time.time() - start_time))

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

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


class LLM:

    def __init__(self, args):
        self.args = args
        
        if args.openai_api:
            os.environ["OPENAI_API_TYPE"] = "azure"
            os.environ["OPENAI_API_VERSION"] = "2024-02-01"
            model_engine = args.model
            
            if "gpt-4" in args.model or "gpt4" in args.model:
                model_name = "gpt-4"
            else:
                model_name = "gpt-3.5-turbo"
            
            self.chat_llm = AzureChatOpenAI(deployment_name=model_engine, max_retries=3, model_name=model_name)
            logger.info(f"Loading chat llm: {self.chat_llm.model_name}")
            self.tokenizer = tiktoken.encoding_for_model(self.chat_llm.model_name)
        
            # To keep track of how much the API costs
            self.prompt_tokens = 0
            self.completion_tokens = 0
        elif args.anthropic_api:
            self.chat_llm = ChatAnthropic(model=args.model, max_retries=3)
            logger.info(f"Loading chat llm: {self.chat_llm.model}")
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
            
            # To keep track of how much the API costs
            self.prompt_tokens = 0
            self.completion_tokens = 0
        elif args.vllm:
            self.chat_llm, self.tokenizer, self.sampling_params = load_vllm(args.model, args)
        else:
            self.chat_llm, self.tokenizer = load_model(args.model)
        
        self.prompt_exceed_max_length = 0
        self.fewer_than_50 = 0
        self.azure_filter_fail = 0
        
    def generate(self, prompt, max_tokens, stop=None):
        args = self.args
        if max_tokens <= 0:
            self.prompt_exceed_max_length += 1
            logger.warning("Prompt exceeds max length and return an empty string as answer. If this happens too many times, it is suggested to make the prompt shorter")
            return ""
        if max_tokens < 50:
            self.fewer_than_50 += 1
            logger.warning("The model can at most generate < 50 tokens. If this happens too many times, it is suggested to make the prompt shorter")
        
        if args.openai_api or args.anthropic_api:
            prompt = [
                SystemMessage(
                    content="You are a helpful assistant that answers the following questions with proper citations."
                ),
                HumanMessage(
                    content=prompt
                ),
            ]
            ok = False
            while not ok:
                try:
                    response = self.chat_llm.invoke(
                        prompt,
                        temperature = args.temperature,
                        max_tokens=max_tokens,
                        stop=stop,
                        top_p=args.top_p)
                    ok = True
                except Exception as e:
                    logger.error(f"OpenAI API Error: {repr(e)}")
                    if "triggering Azure OpenAI's content management policy" in repr(e):
                        # filtered by Azure 
                        self.azure_filter_fail += 1
                    if "rate limit" in repr(e):
                        wait_time =  re.search(r'retry after (\d+) second', repr(e)).group(1)
                        logger.critical("Sleeping for {} seconds.".format(wait_time))
                        time.sleep(int(wait_time))
                        continue
                    return ""
            
            if args.openai_api:
                self.prompt_tokens += response.response_metadata['token_usage']['prompt_tokens']
                self.completion_tokens += response.response_metadata['token_usage']['completion_tokens']
            
            if args.anthropic_api:
                self.prompt_tokens += response.response_metadata['usage']['input_tokens']
                self.completion_tokens += response.response_metadata['usage']['output_tokens']

            return response.content.strip()

        else:
            inputs = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.chat_llm.device)
            outputs = self.chat_llm.generate(
                **inputs,
                do_sample=True, temperature=args.temperature, top_p=args.top_p, 
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                eos_token_id=[self.chat_llm.config.eos_token_id]
            )
            generation = self.tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
            
            return generation.strip()
    

    def batch_generate(self, prompts, stop=None):
        args = self.args

        if args.vllm:
            inputs = [self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False) for prompt in prompts]
            self.sampling_params.n = 1  # Number of output sequences to return for the given prompt
            self.sampling_params.stop_token_ids = [self.chat_llm.llm_engine.get_model_config().hf_config.eos_token_id]
            outputs = self.chat_llm.generate(
                inputs,
                self.sampling_params,
                use_tqdm=True,
            )
            generation = [output.outputs[0].text.strip() for output in outputs]
            
            return generation
            
        else:
            NotImplementedError("No implemented batch generation method!")



def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")
    parser.add_argument("--prompt_file", type=str, help="Path to the prompt file")
    parser.add_argument("--infer_file", type=str, help="Path to the infer file")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    
    # ICL setting
    parser.add_argument("--ndoc", type=int, help="Number of documents")
    parser.add_argument("--shot", type=int, help="Number of ICL demonstrations")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")
    
    # Model and name
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset (for saving)")
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--openai_api", type=bool, default=False, help="Whether to use OpenAI API")
    parser.add_argument("--anthropic_api", type=bool, default=False, help="Whether to use Anthropic API")
    parser.add_argument("--vllm", type=bool, default=False, help="Whether to use vllm for acceleration")
    
    # Decoding
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--max_new_tokens", type=int, default=300, help="Max number of new tokens to generate in one step")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length the model can take. Should set properly wrt the model to avoid position overflow.")
    parser.add_argument("--num_samples", type=int, default=1, help="Sample multiple answers.")
    
    
    # Load config
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()
    for k in args.__dict__:
        print(f"{k}: {args.__dict__[k]}")
        
    logger.critical("Loading env variables and setting seed...")
    
    set_seed(args.seed)
    
    if "16k" in args.model:
        args.max_length = 16384
    elif "32k" in args.model:
        args.max_length = 32768
    elif "turbo" in args.model:
        args.max_length = 4096
    elif "gpt-4" in args.model or "gpt4" in args.model:
        args.max_length = 8192
    elif "llama-2" in args.model.lower() or "llama2" in args.model.lower():
        args.max_length = 4096
    elif "llama-3" in args.model.lower() or "llama3" in args.model.lower():
        args.max_length = 8192
    elif "claude" in args.model:
        args.max_length = 8192
    logger.info(f"Set the model max length to {args.max_length} (if not correct, check the code)")

    # Load the model or setup the API
    llm = LLM(args)
    
    # Load data
    prompt_data = json.load(open(args.prompt_file))
    infer_data = json.load(open(args.infer_file))
    
    # Generate the demonstration part
    head_prompt = ""
    if "rejection" in args.prompt_file:
        logger.warning("Using rejection head prompts...")
        pos_train_ids = np.random.choice(len(prompt_data["positive_demos"]), ceil(args.shot/2), replace=False)
        rej_train_ids = np.random.choice(len(prompt_data["reject_demos"]), args.shot//2, replace=False)

        train_items = []
        for pos_train_id in pos_train_ids:
            train_items.append(prompt_data["positive_demos"][pos_train_id])
        for rej_train_id in rej_train_ids:
            train_items.append(prompt_data["reject_demos"][rej_train_id])
        random.shuffle(train_items)
        for train_item in train_items:
            ndoc = args.ndoc
            head_prompt += make_demo(
                train_item, prompt=prompt_data["demo_prompt"], ndoc=ndoc, doc_prompt=prompt_data["doc_prompt"], 
                instruction=prompt_data["instruction"],
            )
            head_prompt += prompt_data["demo_sep"]
    
    else:
        train_ids = np.random.choice(len(prompt_data["demos"]), args.shot, replace=False)
        for train_id in train_ids:
            train_item = prompt_data["demos"][train_id]
            ndoc = args.ndoc
            head_prompt += make_demo(
                train_item, prompt=prompt_data["demo_prompt"], ndoc=ndoc, doc_prompt=prompt_data["doc_prompt"], 
                instruction=prompt_data["instruction"]
            )
            head_prompt += prompt_data["demo_sep"]
        
    # Generate full prompt
    logger.info("Generating prompts...") 
    incomplete_doc_list = 0 # For some questions there might be fewer than ndoc documents
    for idx, infer_item in enumerate(tqdm(infer_data)):
        infer_item["docs"] = infer_item["docs"][:args.ndoc]
        random.shuffle(infer_item["docs"])
        infer_data[idx]['prompt'] = head_prompt + make_demo(
            infer_item, prompt=prompt_data["demo_prompt"], ndoc=args.ndoc, doc_prompt=prompt_data["doc_prompt"],
            instruction=prompt_data["instruction"], 
            test=True
        )
        infer_data[idx]['docs'] = infer_item["docs"]
        if len(infer_data[idx]['docs']) < args.ndoc:
            incomplete_doc_list += 1
    logger.info("Done.")
    if incomplete_doc_list > 0:
        logger.warning(f"There are {incomplete_doc_list} questions that have incomplete document list (may due to a lot of them are filtered out by summary/extraction).")
    

    # Response generation: process a batch of items
    if args.vllm:
        prompts = [item['prompt'] for item in infer_data for _ in range(args.num_samples)]
        if args.openai_api:
            prompt_lengths = [len(llm.tokenizer.encode(prompt)) for prompt in prompts]
        elif args.anthropic_api:
            prompt_lengths = [len(llm.tokenizer.encode(prompt)) for prompt in prompts]
        else:
            prompt_lengths = [len(llm.tokenizer.tokenize(prompt)) for prompt in prompts]
        max_prompt_len = max(prompt_lengths)

        if idx == 0:
            print(prompts[0])
        
        # Generate outputs in batch
        logger.info(f"Max_N: {max_prompt_len}")
        batch_outputs = llm.batch_generate(prompts)
        
        # Post-process each output
        for i in range(len(infer_data)):
            output_array = []
            for j, output in enumerate(batch_outputs[i:i + args.num_samples]):
                output_array.append(output)
                output_array[-1] = output_array[-1].replace("<|im_end|>", "").rstrip()
                if output_array[-1].endswith("End."):
                    output_array[-1] = output_array[-1][:-len("End.")]

                logger.info(f"Prompt length={prompt_lengths[i + j]}")
                logger.info(f"Question: {infer_data[i]['question']}")
                logger.info(f"Gold answer: {infer_data[i]['answer']}")
                logger.info(f"Final model output: {output_array[-1]}")

            infer_data[i]['output'] = output_array if len(output_array) > 1 else output_array[0]

    else:
        for idx, item in enumerate(tqdm(infer_data)):
            prompt = item['prompt']
            if args.openai_api:
                prompt_len = len(llm.tokenizer.encode(prompt))
            elif args.anthropic_api:
                prompt_len = len(llm.tokenizer.encode(prompt))
            else:
                prompt_len = len(llm.tokenizer.tokenize(prompt))
            
            if idx == 0:
                print(prompt)

            output_array = []
            for _ in range(args.num_samples):
            
                logger.info(f"N: {prompt_len}")
                output_array.append(llm.generate(prompt, min(args.max_new_tokens, args.max_length-prompt_len)))
                item['prompt'] = prompt
                
                output_array[-1] = output_array[-1].replace("<|im_end|>", "").rstrip()
                if output_array[-1].endswith("End."):
                    output_array[-1] = output_array[-1][:-len("End.")]

                logger.info(f"Prompt length={prompt_len}")
                logger.info(f"Question: {item['question']}")
                logger.info(f"Gold answer: {item['answer']}")
                logger.info(f"Final model output: {output_array[-1]}") 

            item['output'] = output_array if len(output_array) > 1 else output_array[0]

    
    # Statistics
    logger.info(f"#Cases when prompts exceed max length: {llm.prompt_exceed_max_length}")
    logger.info(f"#Cases when max new tokens < 50: {llm.fewer_than_50}")

    # Save the result
    model_name = args.model
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    name = f"{args.dataset_name}-{model_name}-shot{args.shot}-ndoc{args.ndoc}-{args.seed}-temp{args.temperature}"
    
    if args.num_samples > 1:
        name += f"-sample{args.num_samples}"
        
    infer_data = {
        "args": args.__dict__,
        "data": infer_data,
    }
    
    if args.openai_api:
        logger.info(f"Token used: prompt {llm.prompt_tokens}; completion {llm.completion_tokens}")
        if "turbo" in args.model:
            p_price, c_price = 0.0015, 0.002
            if "16k" in args.model:
                p_price, c_price = 0.003, 0.004
        elif "gpt4" in args.model or "gpt-4" in args.model:
            p_price, c_price = 0.03, 0.06
            if "32k" in args.model:
                p_price, c_price = 0.06, 0.12
        else:
            logger.warn("Cannot find model price")
            p_price, c_price = 0, 0

        infer_data["total_cost"] = llm.prompt_tokens / 1000 * p_price + llm.completion_tokens / 1000 * c_price        

        logger.info(f"Unit price (Oct 16, 2023, prompt/completion) per KToken: {p_price}/{c_price}")
        logger.info(f"Total cost: %.1f" % (infer_data["total_cost"]))
        
        # azure failure
        infer_data["azure_filter_fail"] = llm.azure_filter_fail 
        logger.info(f"Azure failed case num: {llm.azure_filter_fail}")
    
    if args.anthropic_api:
        logger.info(f"Token used: prompt {llm.prompt_tokens}; completion {llm.completion_tokens}")
        if "claude-3-5" in args.model:
            if "sonnet" in args.model:
                p_price, c_price = 3.00, 15.00
        elif "claude-3" in args.model:
            if "haiku" in args.model:
                p_price, c_price = 0.25, 1.25
            elif "sonnet" in args.model:
                p_price, c_price = 3.00, 15.00
            elif "opus" in args.model:
                p_price, c_price = 15.00, 75.00
        elif "claude-2.0" in args.model or "claude-2.1" in args.model:
            p_price, c_price = 8.00, 24.00
        elif "claude-instant" in args.model:
            p_price, c_price = 0.80, 2.40
        else:
            logger.warn("Cannot find model price")
            p_price, c_price = 0, 0

        infer_data["total_cost"] = llm.prompt_tokens / 1000000 * p_price + llm.completion_tokens / 1000000 * c_price        

        logger.info(f"Unit price (July 16, 2024, prompt/completion) per MToken: {p_price}/{c_price}")
        logger.info(f"Total cost: %.1f" % (infer_data["total_cost"]))

    output_dir = Path(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    json.dump(infer_data, open(output_dir.joinpath(name + ".json"), "w"), indent=4)
    

if __name__ == "__main__":
    main()