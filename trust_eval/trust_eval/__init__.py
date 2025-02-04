# trust_eval/__init__.py

__version__ = "0.1.1"
__author__ = "Shang Hong Sim"

from .auto_ais_loader import (
    delete_autoais_model_and_tokenizer,
    get_autoais_model_and_tokenizer,
)
from .config import EvaluationConfig, ResponseGeneratorConfig
from .data import construct_data, save_data
from .evaluator import Evaluator
from .llm import LLM
from .logging_config import logger
from .metrics import (
    compute_answered_ratio,
    compute_citation_metrics,
    compute_claim_match,
    compute_exact_match,
    compute_len,
    compute_macro_metrics,
    compute_qampari_f1,
    compute_trust_score,
    get_all_cm_scores,
    get_all_em5_scores,
    get_all_em_scores,
    get_basic_stats,
    get_citation_scores,
    get_macro_scores,
)
from .response_generator import ResponseGenerator
from .retrieval import gtr_build_index, gtr_wiki_query
from .utils import load_model, load_vllm, make_demo, normalize_answer, remove_citations

logger.info(f"trust_eval version {__version__} by {__author__} initialized.")
