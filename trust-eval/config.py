from dataclasses import dataclass
from typing import List, Literal, Optional

import yaml


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
class EvaluationConfig:
    # Eval settings
    eval_type: Literal["em", "em@5", "cm"] = None  # evaluation type for different dataset format

    eval_file: Optional[str] = None 
    output_path: str  # output file path for evaluation result (required)
    citations: bool = False  # Evaluate using citation data
    at_most_citations: int = 3  # Maximum number of documents for citation evaluation
    claims_nli: bool = False  # Use claims for ELI5
    rejection_flag: str = "I apologize, but I couldn't find an answer"
    rejection_threshold: int = 85
    autoais_model: str = "google/t5_xxl_true_nli_mixture"

    calib: bool = True
    parametric: bool = True
    is_qampari: bool = False

    def __post_init__(self):
        """
        Automatically set default configurations based on the evaluation type after initialization.
        """
        self.set_defaults_based_on_eval_type()

    def set_defaults_based_on_eval_type(self):
        """
        Set default configurations based on the evaluation type.
        """
        default_configs = {
            "em": {"citations": True, "claims_nli": False, "is_qampari": False},
            "em@5": {"citations": True, "claims_nli": False, "is_qampari": True},
            "cm": {"citations": True, "claims_nli": True, "is_qampari": False},
        }

        if self.eval_type in default_configs:
            for key, value in default_configs[self.eval_type].items():
                setattr(self, key, value)
        else:
            raise ValueError(f"Unknown eval_type: {self.eval_type}")

    def update_from_dict(self, config_dict: dict):
        """
        Update the Config dataclass fields from a dictionary.
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "EvaluationConfig":
        """
        Load configuration from a YAML file and update the default instance.
        """
        with open(yaml_path, "r") as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
        config = cls()  # Initialize with defaults
        config.update_from_dict(yaml_data)  # Update from YAML
        return config

