

# trust_eval/llm.py
import json
import logging
import os
from typing import Any, List, Optional

import openai

from .utils import load_model, load_vllm

logger = logging.getLogger(__name__)

class LLM:
    def __init__(self, args: Any) -> None:
        self.args = args
        self.prompt_exceed_max_length = 0
        self.fewer_than_50 = 0
        self.azure_filter_fail = 0

        if args.openai_api:
            logger.info("Loading OpenAI model...")
            self._initialize_openai_api(args)
        elif args.vllm:
            logger.info("Loading VLLM model...")
            self._initialize_vllm(args)
        else:
            logger.info("Loading custom model...")
            self._initialize_custom_model(args)

    def _initialize_openai_api(self, args: Any) -> None:
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2", fast_tokenizer=False)
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _initialize_openai_client(self, args: Any) -> None:
        api_key = os.environ.get("OPENAI_API_KEY", "")

        if args.azure:
            client = openai.AzureOpenAI( # type: ignore[assignment]
                api_key=api_key,
                azure_endpoint=os.environ.get("OPENAI_ENDPOINT", ""),
                api_version="2023-05-15"
            )
        else:
            client = openai.OpenAI( # type: ignore[assignment]
                api_key=api_key,
                organization=os.environ.get("OPENAI_ORG_ID", "")
        )
        
        self.client = client

    def _initialize_vllm(self, args: Any) -> None:
        self.chat_llm, self.tokenizer, self.sampling_params = load_vllm(args.model, args)

    def _initialize_custom_model(self, args: Any) -> None:
        self.model, self.tokenizer = load_model(args.model, lora_path=args.lora_path)

    def _call_openai_api(self, is_chat: bool, formatted_prompt: Any, max_tokens: Optional[int], stop: Optional[List[str]]) -> Optional[str]:
        """
        Handles retries and API requests for both `ChatCompletion` and `Completion`.
        """
        retry_count = 0
        deploy_name = self.args.model if self.args.azure else None

        while retry_count < 3:
            try:
                if is_chat:
                    response = self.client.chat.completions.create( # type: ignore[assignment]
                        model=self.args.model,
                        messages=formatted_prompt,
                        temperature=self.args.temperature,
                        max_tokens=max_tokens,
                        stop=stop,
                        top_p=self.args.top_p
                    )
                    answer = response.choices[0].message.content
                else:
                    response = self.client.completions.create( # type: ignore[assignment]
                        model=self.args.model,
                        prompt=formatted_prompt,
                        temperature=self.args.temperature,
                        max_tokens=max_tokens,
                        top_p=self.args.top_p,
                        stop=["\n", "\n\n"] + (stop or [])
                    )
                    answer = response.choices[0].text # type: ignore[attr-defined]

                # Update token usage
                self.prompt_tokens += getattr(response.usage, "prompt_tokens", 0)
                self.completion_tokens += getattr(response.usage, "completion_tokens", 0)

                return answer

            except Exception as error:
                last_error = error
                retry_count += 1
                logger.warning(f"OpenAI API retry {retry_count} times ({error})")

                if "triggering Azure OpenAI's content management policy" in str(error):
                    self.azure_filter_fail += 1
                    return ""

        print(f"\n Fatal error: {last_error} \n")
        return ""

    def generate(self, prompt: str, max_tokens: int, stop: Optional[List[str]]=None) -> Optional[str]:
        args = self.args
        if max_tokens <= 0:
            self.prompt_exceed_max_length += 1
            logger.warning("Prompt exceeds max length and return an empty string as answer. If this happens too many times, it is suggested to make the prompt shorter")
            return ""
        if max_tokens < 50:
            self.fewer_than_50 += 1
            logger.warning("The model can at most generate < 50 tokens. If this happens too many times, it is suggested to make the prompt shorter")

        if args.openai_api:
            use_chat_api = ("turbo" in self.args.model and not self.args.azure) or (
                ("gpt4" in self.args.model or "gpt-4" in self.args.model) and self.args.azure
            )

            formatted_prompt = (
                [
                    {'role': 'system', 'content': "You are a helpful assistant that answers questions with proper citations."},
                    {'role': 'user', 'content': prompt}
                ] if use_chat_api else prompt
            )

            return self._call_openai_api(use_chat_api, formatted_prompt, max_tokens, stop)

        else:
            inputs = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                do_sample=True, temperature=args.temperature, top_p=args.top_p, 
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                eos_token_id=self.model.config.eos_token_id if isinstance(self.model.config.eos_token_id, list) else [self.model.config.eos_token_id]
            )
            generation = self.tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
            return generation.strip()


    def batch_generate(self, prompts: List[str]) -> Optional[List[str]]:
        args = self.args
        
        if args.vllm:
            inputs = [self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False) for prompt in prompts]
            self.sampling_params.n = 1  # Number of output sequences to return for the given prompt
            if isinstance(self.chat_llm.llm_engine.get_model_config().hf_config.eos_token_id, list):
                self.sampling_params.stop_token_ids = []
                self.sampling_params.stop_token_ids.extend(self.chat_llm.llm_engine.get_model_config().hf_config.eos_token_id)
            elif isinstance(self.chat_llm.llm_engine.get_model_config().hf_config.eos_token_id, int):
                self.sampling_params.stop_token_ids = [self.chat_llm.llm_engine.get_model_config().hf_config.eos_token_id]

            outputs = self.chat_llm.generate(
                inputs,
                self.sampling_params,
                use_tqdm=True,
            )
            generation = [output.outputs[0].text.strip() for output in outputs]
            return generation
            
        else:
            raise NotImplementedError("No implemented batch generation method!")
            
