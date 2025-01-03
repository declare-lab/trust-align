# trust_eval/__init__.py

__version__ = "0.1.0"
__author__ = "Shang Hong Sim"

# Importing core components for easy access
from .config import EvaluationConfig, ResponseGeneratorConfig
from .evaluator import Evaluator

# Setting up a default logger
from .logging_config import logger
from .metrics import (
    compute_citation_metrics,
    compute_exact_match,
    compute_len,
    compute_macro_metrics,
    compute_trust_score,
    get_all_cm_scores,
    get_all_em5_scores,
    get_all_em_scores,
    get_citation_scores,
    get_macro_scores,
)
from .response_generator import ResponseGenerator
from .utils import load_model, load_vllm, make_demo, normalize_answer, remove_citations

logger.info(f"trust_eval version {__version__} by {__author__} initialized.")
