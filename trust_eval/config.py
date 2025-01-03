import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Literal, Optional

import yaml

from .logging_config import logger


@dataclass
class BaseConfig(ABC):
    yaml_path: str
    data_type: Literal["asqa", "qampari", "eli5"] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BaseConfig":
        """
        Load configuration from a YAML file and update the default instance.
        """
        with open(yaml_path, "r") as file:
            yaml_data = yaml.safe_load(file)
        
        data_type = yaml_data.get("data_type")
        if not data_type:
            raise ValueError("The YAML file must specify 'data_type'.")

        config = cls(yaml_path=yaml_path, data_type=data_type)
        config.set_defaults()
        config.update_from_dict(yaml_data)
        return config
    
    def update_from_dict(self, config_dict: dict):
        """
        Update the Config dataclass fields from a dictionary.
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def __post_init__(self):
        """
        Automatically set default configurations based on the evaluation type after initialization.
        """
        self.set_defaults()

    @abstractmethod
    def set_defaults(self):
        """This method must be implemented by subclasses."""
        pass
  

@dataclass
class ResponseGeneratorConfig(BaseConfig):
    
    # General settings
    prompt_file: Optional[str] = None  # Path to the prompt file
    eval_file: Optional[str] = None  # Path to the evaluation file
    output_path: Optional[str] = None  # Output directory for model's output
    quick_test: Optional[int] = None  # Number of examples for quick testing

    # ICL settings
    ndoc: int = 5  # Number of documents
    shot: int = 2  # Number of ICL demonstrations
    seed: int = 42  # Random seed
    no_doc_in_demo: bool = False  # Whether to remove documents in demo
    fewer_doc_in_demo: bool = False  # Whether to use fewer documents in demo
    ndoc_in_demo: Optional[int] = None  # Number of documents in demo when using fewer docs
    no_demo: bool = False  # Whether to disable demos
    rejection: bool = True  # Whether to use rejection demos

    # Model and naming
    model: str = None  # Model to use
    openai_api: bool = False  # Whether to use OpenAI API
    azure: bool = False  # Whether to use Azure OpenAI API
    lora_path: Optional[str] = None  # Path to LoRA training checkpoint
    vllm: bool = True  # Whether to use vllm for acceleration

    # Decoding settings
    temperature: float = 0.5  # Temperature for decoding
    top_p: float = 0.95  # Nucleus sampling top-p
    max_new_tokens: int = 300  # Maximum number of new tokens to generate
    max_length: int = 2048  # Maximum length for model input
    num_samples: int = 1  # Number of samples for multiple answers

    
    def _generate_default_output_path(self):
        base_dir = Path("output")  
        base_dir.mkdir(parents=True, exist_ok=True)  
        file_name = f"{self.data_type}_eval_top100_calibrated.json"
        return str(base_dir / file_name)
    
    def _generate_default_eval_file_path(self):
        base_dir = Path("eval_data")
        base_dir.mkdir(parents=True, exist_ok=True)  
        file_name = f"{self.data_type}_eval_top100_calibrated.json"
        return str(base_dir / file_name)

    def _generate_prompt_path(self):
        base_dir = Path("prompts")
        base_dir.mkdir(parents=True, exist_ok=True)  
        
        closedbook, rejection = "", "_rejection"
        if not self.rejection:
            rejection = "_default"
        if self.ndoc == 0:
            closedbook = "_closedbook"
        file_name = f"{self.data_type}{closedbook}{rejection}.json"

        return str(base_dir / file_name)


    def set_defaults(self):
        """
        Set default configurations based on the evaluation type.
        """
        if self.output_path is None:
            self.output_path = self._generate_default_output_path()
        
        if self.eval_file is None:
            self.eval_file = self._generate_default_eval_file_path()

        if self.prompt_file is None:
            self.prompt_file = self._generate_prompt_path()


@dataclass
class EvaluationConfig(BaseConfig):

    # Eval settings
    eval_file: Optional[str] = None
    output_path: str = None  # output file path for evaluation result (required)
    eval_type: Literal["em", "em@5", "cm"] = None  # evaluation type for different dataset format

    # correctness configs
    compute_correctness: bool = True
    parametric: bool = True
    is_qampari: bool = False

    # citation configs
    compute_citations: bool = True  # Evaluate using citation data
    at_most_citations: int = 3  # Maximum number of documents for citation evaluation
    calib: bool = True
    
    rejection_flag: str = "I apologize, but I couldn't find an answer"
    rejection_threshold: int = 85
    autoais_model: str = "google/t5_xxl_true_nli_mixture"
    
    def _generate_default_output_path(self):
        base_dir = Path("results")  
        base_dir.mkdir(parents=True, exist_ok=True)  
        file_name = f"{self.eval_type}_evaluation.json"
        return str(base_dir / file_name)
    
    def _generate_default_eval_file_path(self):
        base_dir = Path("output")  
        file_name = f"{self.data_type}_eval_top100_calibrated.json"
        return str(base_dir / file_name)
    
    def set_defaults(self):
        """
        Set default configurations based on the evaluation type.
        """
        data2eval = {"asqa": "em",
                     "qampari": "em5",
                     "eli5": "cm"}
       
        if self.data_type in data2eval:
            self.eval_type = data2eval[self.data_type]
            logger.debug("self.eval_type", data2eval[self.data_type])
        else:
            raise ValueError(f"Unknown data_type: {self.data_type}. Please select either asqa, qampari or eli5.")

        default_configs = {
            "em": {"is_qampari": False},
            "em5": {"is_qampari": True},
            "cm": {"is_qampari": False},
        }

        for key, value in default_configs[self.eval_type].items():
            setattr(self, key, value)
        
        if self.output_path is None:
            self.output_path = self._generate_default_output_path()
        
        if self.eval_file is None:
            self.eval_file = self._generate_default_eval_file_path()