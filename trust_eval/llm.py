

# trust_eval/llm.py
import json
import logging
import os

import openai

from .utils import load_model, load_vllm

logger = logging.getLogger(__name__)

class LLM:
    def __init__(self, args):
        self.args = args
        self.prompt_exceed_max_length = 0
        self.fewer_than_50 = 0
        self.azure_filter_fail = 0

        if args.openai_api:
            self._initialize_openai_api(args)
        elif args.vllm:
            logger.info("Loading VLLM model...")
            self._initialize_vllm(args)
        else:
            self._initialize_custom_model(args)

    def _initialize_openai_api(self, args):
        self._set_openai_api_credentials(args)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2", fast_tokenizer=False)
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _set_openai_api_credentials(self, args):
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        if args.azure:
            openai.api_base = os.environ.get("OPENAI_API_BASE")
            openai.api_type = "azure"
            openai.api_version = "2023-05-15"
        else:
            openai.organization = os.environ.get("OPENAI_ORG_ID")

    def _initialize_vllm(self, args):
        self.chat_llm, self.tokenizer, self.sampling_params = load_vllm(args.model, args)

    def _initialize_custom_model(self, args):
        self.model, self.tokenizer = load_model(args.model, lora_path=args.lora_path)

    def generate(self, prompt, max_tokens, stop=None):
        args = self.args
        if max_tokens <= 0:
            self.prompt_exceed_max_length += 1
            logger.warning("Prompt exceeds max length and return an empty string as answer. If this happens too many times, it is suggested to make the prompt shorter")
            return ""
        if max_tokens < 50:
            self.fewer_than_50 += 1
            logger.warning("The model can at most generate < 50 tokens. If this happens too many times, it is suggested to make the prompt shorter")

        if args.openai_api:
            use_chat_api = ("turbo" in args.model and not args.azure) or (("gpt4" in args.model or "gpt-4" in args.model) and args.azure)
            if use_chat_api:
                # For chat API, we need to convert text prompts to chat prompts
                prompt = [
                    {'role': 'system', 'content': "You are a helpful assistant that answers the following questions with proper citations."},
                    {'role': 'user', 'content': prompt}
                ]
            if args.azure:
                deploy_name = args.model

            if use_chat_api:
                is_ok = False
                retry_count = 0
                while not is_ok:
                    retry_count += 1
                    try:
                        response = openai.ChatCompletion.create(
                            engine=deploy_name if args.azure else None,
                            model=args.model,
                            messages=prompt,
                            temperature=args.temperature,
                            max_tokens=max_tokens,
                            stop=stop,
                            top_p=args.top_p,
                        )
                        is_ok = True
                    except Exception as error:
                        if retry_count <= 3:
                            logger.warning(f"OpenAI API retry for {retry_count} times ({error})")
                            if "triggering Azure OpenAI's content management policy" in str(error):
                                # filtered by Azure 
                                self.azure_filter_fail += 1
                                return ""
                            continue
                        print(f"\n Here is a fatal error: {error} \n")
                        return ""
                        # import pdb; pdb.set_trace()
                self.prompt_tokens += response['usage']['prompt_tokens']
                self.completion_tokens += response['usage']['completion_tokens']
                try:
                    answer = response["choices"][0]["message"]["content"]
                except KeyError:
                    print("Error in message chat completions.")
                    print(json.dumps(response) + "\n")
                    answer = ""
                return answer
            else:
                is_ok = False
                retry_count = 0
                while not is_ok:
                    retry_count += 1
                    try:
                        response = openai.Completion.create(
                            engine=deploy_name if args.azure else None,
                            model=args.model,
                            prompt=prompt,
                            temperature=args.temperature,
                            max_tokens=max_tokens,
                            top_p=args.top_p,
                            stop=["\n", "\n\n"] + (stop if stop is not None else [])
                        )    
                        is_ok = True
                    except Exception as error:
                        if retry_count <= 3:
                            logger.warning(f"OpenAI API retry for {retry_count} times ({error})")
                            if "triggering Azure OpenAI's content management policy" in str(error):
                                # filtered by Azure 
                                self.azure_filter_fail += 1
                                return ""
                            continue
                        print(f"\n Here is a fatal error: {error} \n")
                        return ""
                        # import pdb; pdb.set_trace()
                self.prompt_tokens += response['usage']['prompt_tokens']
                self.completion_tokens += response['usage']['completion_tokens']
                return response['choices'][0].get('text') or ""

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


    def batch_generate(self, prompts, stop=None):
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
            NotImplementedError("No implemented batch generation method!")
            
