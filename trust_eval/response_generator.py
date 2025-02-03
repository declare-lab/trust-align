import json
import os
import random
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from .config import ResponseGeneratorConfig
from .llm import LLM
from .logging_config import logger
from .post_hoc_cite import post_hoc_cite
from .utils import make_demo


class ResponseGenerator:
    def __init__(self, config: ResponseGeneratorConfig):
        self.config = config
        self.llm = LLM(config)
        self.max_length = self._set_max_length(config.model, config.max_length)
        logger.info(f"Set the model max length to {self.max_length}")
        self.prompt_file = self.config.prompt_file
        self.data_file = self.config.data_file
        self.output_path = self.config.eval_file
        self.load_data()

    def _set_max_length(self, model_name: str, max_len: int) -> int:
        """Determine max length based on the model type."""
        if max_len is not None:
            return max_len
        if "16k" in model_name:
            return 16384
        if "32k" in model_name:
            return 32768
        if "turbo" in model_name:
            return 4096
        if "gpt4" in model_name or "gpt-4" in model_name:
            return 8192
        if "llama-2" in model_name.lower() or "llama2" in model_name.lower():
            return 4096
        return 2048  # Default fallback

    def load_data(self, prompt_file: Optional[str] = None, data_file: Optional[str] = None) -> None:
        """Load prompt and evaluation data from JSON files."""
        if prompt_file is not None:
            self.prompt_data = json.load(open(prompt_file))
        else:
            assert self.prompt_file is not None
            self.prompt_data = json.load(open(self.prompt_file))
        if data_file is not None:
            self.eval_data = json.load(open(data_file))
        else:
            assert self.data_file is not None
            self.eval_data = json.load(open(self.data_file))[:10]

    def generate_responses(self, data: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Generate responses for evaluation data."""

        if data is None:
            data = self.eval_data
        
        np.random.seed(self.config.seed)

        # Add prompts to evaluation data
        incomplete_doc_list = 0
        for idx, eval_item in enumerate(tqdm(data)):
            eval_item['prompt'] = self.construct_prompt(eval_item)
            doc_list = eval_item["docs"][:self.config.ndoc]
            eval_item['docs'] = doc_list
            if len(doc_list) < self.config.ndoc:
                incomplete_doc_list += 1

        if incomplete_doc_list > 0:
            logger.warning(f"There are {incomplete_doc_list} questions with incomplete document lists.")

        # Generate responses
        if self.config.vllm:
            prompts = [item['prompt'] for item in data for _ in range(self.config.num_samples)]
            prompt_lengths = [len(self.llm.tokenizer.tokenize(prompt)) for prompt in prompts]
            max_prompt_len = max(prompt_lengths)

            if idx == 0:
                print(prompts[0])
            
            # Generate outputs in batch
            logger.info(f"Max prompt length: {max_prompt_len}")
            batch_outputs = self.llm.batch_generate(prompts)
            
            # release vllm
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            if hasattr(self.llm.chat_llm.llm_engine.model_executor, "driver_worker"):
                del self.llm.chat_llm.llm_engine.model_executor.driver_worker
            torch.cuda.empty_cache()
            
            # Post-process each output
            for idx, item in enumerate(tqdm(data)):
                output_array = []
                assert batch_outputs is not None
                for j, output in enumerate(batch_outputs[idx:idx + self.config.num_samples]):
                    output_array.append(output)
                    output_array[-1] = output_array[-1].replace("<|im_end|>", "").rstrip()
                    if output_array[-1].endswith("End."):
                        output_array[-1] = output_array[-1][:-len("End.")]

                item['output'] = output_array if len(output_array) > 1 else output_array[0]

        else:
            for idx, item in enumerate(tqdm(data)):
                prompt = item['prompt']
                item['output'] = self.generate_response_single_query(prompt)
        
        if self.config.posthoc:
            data = post_hoc_cite(data, self.config)

        self.eval_data = data
        return data

    def generate_response_single_query(self, prompt: str) -> Union[str, List[str]]:
        prompt_len = len(self.llm.tokenizer.tokenize(prompt))

        output_array = []
        for _ in range(self.config.num_samples):
        
            logger.info(f"Max prompt length: {prompt_len}")

            response = self.llm.generate(prompt, min(self.config.max_new_tokens, self.config.max_length - prompt_len))

            if response is None:
                raise ValueError("Generation failed: LLM returned None.")

            response = response.replace("<|im_end|>", "").rstrip()
            if response.endswith("End."):
                response = response[:-len("End.")]

            output_array.append(response)

        return output_array if len(output_array) > 1 else output_array[0]
    
    def construct_prompt(self, eval_item: Dict[str, Any]) -> str:
        head_prompt = self._generate_head_prompt()
        return head_prompt + make_demo(
            eval_item, 
            prompt=self.prompt_data["demo_prompt"], 
            ndoc=self.config.ndoc, 
            doc_prompt=self.prompt_data["doc_prompt"],
            instruction=self.prompt_data["instruction"], 
            test=True
        )
    
    def _generate_head_prompt(self) -> str:
        """Generate the head prompt based on demonstrations."""
        head_prompt = ""
        if not self.config.no_demo:
            assert self.config.prompt_file is not None
            if "rejection" in self.config.prompt_file:
                logger.warning("Using rejection head prompts...")
                pos_train_ids = np.random.choice(len(self.prompt_data["positive_demos"]), ceil(self.config.shot / 2), replace=False)
                rej_train_ids = np.random.choice(len(self.prompt_data["reject_demos"]), self.config.shot // 2, replace=False)

                train_items = [
                    *[self.prompt_data["positive_demos"][idx] for idx in pos_train_ids],
                    *[self.prompt_data["reject_demos"][idx] for idx in rej_train_ids]
                ]
                random.shuffle(train_items)

                for train_item in train_items:
                    head_prompt += self._make_demo_item(train_item)
                    head_prompt += self.prompt_data["demo_sep"]
            else:
                train_ids = np.random.choice(len(self.prompt_data["demos"]), self.config.shot, replace=False)
                for train_id in train_ids:
                    train_item = self.prompt_data["demos"][train_id]
                    head_prompt += self._make_demo_item(train_item)
                    head_prompt += self.prompt_data["demo_sep"]

        return head_prompt

    def _make_demo_item(self, train_item: Dict[str, Any]) -> str:
        """Helper to create demo items for the prompt."""
        ndoc = self.config.ndoc
        if self.config.no_doc_in_demo:
            ndoc = 0
        elif self.config.fewer_doc_in_demo:
            assert self.config.ndoc_in_demo is not None
            ndoc = self.config.ndoc_in_demo

        return make_demo(
            train_item, 
            prompt=self.prompt_data["demo_prompt"], 
            ndoc=ndoc, 
            doc_prompt=self.prompt_data["doc_prompt"], 
            instruction=self.prompt_data["instruction"], 
            use_shorter=None
        )

    def save_responses(self, output_path: Optional[str] = None) -> None:
        """Save evaluation data to a JSON file."""
        if output_path is not None:
            self.output_path = output_path
        assert self.output_path is not None
        with open(self.output_path, "w") as f:
            json.dump({"data": self.eval_data, "config": vars(self.config)}, f, indent=4)
        logger.info(f"Results saved to {self.output_path}")
