import json
import os
import random
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
from math import ceil

from .llm import LLM  
from .config import ResponseGeneratorConfig  
from .utils import make_demo 

logger = logging.getLogger(__name__)

class ResponseGenerator:
    def __init__(self, config: ResponseGeneratorConfig):
        self.config = config
        self.llm = LLM(config)
        self.max_length = self._set_max_length(config.model)

    def _set_max_length(self, model_name: str) -> int:
        """Determine max length based on the model type."""
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

    def load_data(self, prompt_file: str, eval_file: str):
        """Load prompt and evaluation data from JSON files."""
        self.prompt_data = json.load(open(prompt_file))
        self.eval_data = json.load(open(eval_file))

    def generate_responses(self):
        """Generate responses for evaluation data."""
        logger.info(f"Set the model max length to {self.max_length}")
        np.random.seed(self.config.seed)

        # Generate demonstration part
        head_prompt = self._generate_head_prompt()

        # Add prompts to evaluation data
        incomplete_doc_list = 0
        for idx, eval_item in enumerate(tqdm(self.eval_data)):
            eval_item['prompt'] = head_prompt + make_demo(
                eval_item, 
                prompt=self.prompt_data["demo_prompt"], 
                ndoc=self.config.ndoc, 
                doc_prompt=self.prompt_data["doc_prompt"],
                instruction=self.prompt_data["instruction"], 
                test=True
            )
            doc_list = eval_item["docs"][:self.config.ndoc]
            eval_item['docs'] = doc_list
            if len(doc_list) < self.config.ndoc:
                incomplete_doc_list += 1

        if incomplete_doc_list > 0:
            logger.warning(f"There are {incomplete_doc_list} questions with incomplete document lists.")

        # Generate responses
        logger.info("Generating responses...")
        for item in tqdm(self.eval_data):
            prompt = item['prompt']
            prompt_len = len(self.llm.tokenizer.tokenize(prompt))
            output_array = []
            for _ in range(self.config.num_samples):
                output = self.llm.generate(prompt, min(self.config.max_new_tokens, self.max_length - prompt_len))
                output = output.replace("<|im_end|>", "").rstrip()
                if output.endswith("End."):
                    output = output[:-len("End.")]
                output_array.append(output)
            item['output'] = output_array if len(output_array) > 1 else output_array[0]

        return self.eval_data

    def _generate_head_prompt(self):
        """Generate the head prompt based on demonstrations."""
        head_prompt = ""
        if not self.config.no_demo:
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

    def _make_demo_item(self, train_item):
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

    def save_results(self, output_dir: str, file_name: str):
        """Save evaluation data to a JSON file."""
        output_dir = Path(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = output_dir.joinpath(file_name)
        with open(file_path, "w") as f:
            json.dump({"data": self.eval_data, "config": vars(self.config)}, f, indent=4)
        logger.info(f"Results saved to {file_path}")
