import os
from abc import ABC
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal, Optional, Type, TypeVar

import yaml

from .logging_config import logger

T = TypeVar("T", bound="BaseConfig")

@dataclass
class BaseConfig(ABC):
    @classmethod
    def from_yaml(cls: Type[T], yaml_path: str) -> T:
        """
        Load configuration from a YAML file and update the default instance.
        """
        with open(yaml_path, "r") as file:
            yaml_data = yaml.safe_load(file)

        valid_fields = {f.name for f in fields(cls)}
        valid_data = {key: value for key, value in yaml_data.items() if key in valid_fields}
        config = cls(yaml_path=yaml_path, **valid_data) # type: ignore[call-arg]
        return config
    
    def _generate_path(self, dir: str, file_name: str) -> str:
        base_dir = Path(dir)
        base_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        return str(base_dir / file_name)
    
@dataclass
class ResponseGeneratorConfig(BaseConfig):

    # General settings
    yaml_path: str
    model: str = "Qwen/Qwen2.5-3B-Instruct" # Model to use
    data_type: Literal["asqa", "qampari", "eli5"] = "eli5"
    prompt_file: Optional[str] = None  # Path to the prompt file
    data_file: Optional[str] = None  # Path to the evaluation file
    eval_file: Optional[str] = None  # Output directory for model's output
    quick_test: Optional[int] = None  # Number of examples for quick testing

    # ICL settings
    ndoc: int = 5  # Number of documents
    shot: int = 2  # Number of ICL demonstrations
    seed: int = 42  # Random seed
    no_doc_in_demo: bool = False  # Whether to remove documents in demo
    fewer_doc_in_demo: bool = False  # Whether to use fewer documents in demo
    ndoc_in_demo: Optional[int] = None  # Number of documents in demo when using fewer docs
    no_demo: bool = False  # Whether to disable demos; same effect as shot=0
    rejection: bool = True  # Whether to use rejection demos

    # Model and naming
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

    rejection_flag: str = "I apologize, but I couldn't find an answer"
    rejection_threshold: int = 85
    autoais_model: str = "google/t5_xxl_true_nli_mixture"

    # Posthoc settings
    posthoc: bool = False
    posthoc_retriever: str = "gtr-t5-large" # Retriever to use. Options: `gtr-t5-large` for PostCite, `nli` for PostAttr
    posthoc_retriever_device: str = "cuda" # Where to put the dense retriever if using. Options: `cuda`, `cpu`
    overwrite: bool = True # Overwrite existing citations
    external_docs: Optional[str] = None # Use external documents

    def __post_init__(self) -> None:
        self.data_file = self.data_file or self._generate_path(
            dir="data", file_name=f"{self.data_type}_eval_top100_calibrated.json"
        )
        
        self.eval_file = self.eval_file or self._generate_path(
            dir="eval_data", file_name=f"{self.data_type}_eval_top100_calibrated.json"
        )
        
        if self.prompt_file is None:
            closedbook = "_closedbook" if self.ndoc == 0 else ""
            rejection = "_default" if not self.rejection else "_rejection"
            file_name = f"{self.data_type}{closedbook}{rejection}.json"
            self.prompt_file = self._generate_path(dir="prompts", file_name=file_name)


@dataclass
class EvaluationConfig(BaseConfig):

    # Eval settings
    yaml_path: str
    data_type: Literal["asqa", "qampari", "eli5"] = "eli5"
    eval_file: Optional[str] = None
    result_path: Optional[str] = None  # output file path for evaluation result (required)
    eval_type: str = "cm"  # evaluation type for different dataset format ["em", "em5", "cm"]

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
    
    def __post_init__(self) -> None:
        """
        Set default configurations based on the evaluation type.
        """
        data2eval = {
            "asqa": {"eval_type": "em", "is_qampari": False},
            "qampari": {"eval_type": "em5", "is_qampari": True},
            "eli5": {"eval_type": "cm", "is_qampari": False},
        }
        
        if self.data_type in data2eval:
            eval_config = data2eval[self.data_type]
            assert isinstance(eval_config["eval_type"], str) and eval_config["eval_type"] in ["em", "em5", "cm"]
            self.eval_type = eval_config["eval_type"]
            for key, value in eval_config.items():
                setattr(self, key, value)
        logger.debug("self.eval_type: %s", self.eval_type)

        self.eval_file = self.eval_file or self._generate_path(dir="eval_data", file_name=f"{self.data_type}_eval_top100_calibrated.json")
        self.result_path = self.result_path or self._generate_path(dir="results", file_name=f"{self.eval_type}_evaluation.json")