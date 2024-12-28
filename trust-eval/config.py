from dataclasses import dataclass
from typing import List, Literal, Optional

@dataclass
class ResponseGeneratorConfig:
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

