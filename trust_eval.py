import collections
import copy
import json
import os
import random
import re
import string
from math import ceil
from pathlib import Path

import colorlog
import numpy as np
import openai
import torch
import yaml
from fuzzywuzzy import fuzz
from nltk import sent_tokenize
from rouge_score import rouge_scorer, scoring
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from utils import *

fmt_string = '%(log_color)s %(asctime)s - %(levelname)s - %(message)s'
log_colors = {
        'DEBUG': 'white',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'purple'
        }
colorlog.basicConfig(log_colors=log_colors, format=fmt_string, level=colorlog.INFO)
logger = colorlog.getLogger(__name__)
logger.setLevel(colorlog.INFO)

AUTOAIS_MODEL="google/t5_xxl_true_nli_mixture"
global autoais_model, autoais_tokenizer
autoais_model, autoais_tokenizer = None, None

REJECTION_FUZZ_THRESHOLD=85
REJECTION_FLAG="I apologize, but I couldn't find an answer"


def compute_f1(a_gold, a_pred):
    """Compute F1 score between two strings."""

    def _get_tokens(s):
        if not s:
            return []
        return normalize_answer(s).split()

    gold_toks = _get_tokens(a_gold)
    pred_toks = _get_tokens(a_pred)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_exact(a_gold, a_pred):
    """Check whether two strings are equal up to normalization."""

    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def exact_presence(short_answers, context):
    """Verify if any of the answers is present in the given context.
    Args:
        short_answers: list of short answers to look for in the context
        context: a paragraph to search for short answers
    Returns:
        true if any of the short answers is present in the context
    """

    n_short_answers = [normalize_answer(sa) for sa in short_answers]
    n_context = normalize_answer(context)

    for ans in n_short_answers:
        if ans in n_context:
            return True

    return False


def compute_str_em(data, calib=False, parametric=False):
    """Compute STR-EM metric (only for ASQA)
    Args:
        data: requires field `qa_pairs/short_answers` and `output`
    Returns:
        STR-EM and STR-EM-HIT ()
    """
    
    if len(data) == 0:
        logger.warn("Warning: data should not be zero")
        return 0, 0

    if 'qa_pairs' not in data[0] or data[0]['qa_pairs'] is None:
        logger.warn("Warning: no QA pairs found in data")
        return 0, 0

    acc = []
    hit = []

    for item in tqdm(data):
        loc_acc = []
        if calib:
            # at least answerable
            union_ans_set = np.bitwise_or.reduce([doc["answers_found"] for doc in item['docs']]).tolist()
            if any(union_ans_set) and (not fuzz.partial_ratio(normalize_answer(REJECTION_FLAG), normalize_answer(item['output'])) > REJECTION_FUZZ_THRESHOLD):
                for i, qa_pair in enumerate(item['qa_pairs']):
                    # ignore golden answers that are not recalled by given docs
                    if union_ans_set[i] == 1:
                        loc_acc.append(exact_presence(qa_pair['short_answers'], item["output"]))
            else:
                loc_acc.append(False)
        else:
            for qa_pair in item['qa_pairs']:
                loc_acc.append(exact_presence(qa_pair['short_answers'], item["output"]))
                
        if calib and parametric:
            acc.append(np.sum(a=loc_acc)/len(union_ans_set))
            hit.append( int(np.sum(loc_acc)/len(union_ans_set) == 1) )
        else:
            acc.append(np.mean(loc_acc))
            hit.append( int(np.mean(loc_acc) == 1) )

    return 100 * np.mean(acc), 100 * np.mean(hit)


def compute_len(data):
    """Compute average length of predictions."""

    if len(data) == 0:
        logger.warn("Warning: data should not be zero")
        return 0
    
    res, cntr = 0, 0
    for item in data:
        res += len(item["output"].split())
        cntr += 1
    return res / cntr


def _run_nli_autoais(passage, claim):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    global autoais_model, autoais_tokenizer
    input_text = "premise: {} hypothesis: {}".format(passage, claim)
    input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(autoais_model.device)
    with torch.inference_mode():
        outputs = autoais_model.generate(input_ids, max_new_tokens=10)
    result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    inference = 1 if result == "1" else 0
    return inference


def compute_claims(data, calib=False, parametric=False):
    
    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto")
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)

    logger.info("Computing claims...")
    regular_scores = []
    answered_scores = []
    if calib:
        calib_answered_scores = []
        calib_answerable_scores = []
        if parametric:
            parametric_answered_scores = []
    for item in tqdm(data):
        normalized_output = remove_citations(item['output'])
        entail = 0
        claims = [claim_list[0] for claim_list in item["answers"]]
        
        if calib:
            calib_entail = 0
            union_ans_set = np.bitwise_or.reduce([doc["answers_found"] for doc in item['docs']]).tolist()
        
        for i, claim in enumerate(claims):
            ais_score = _run_nli_autoais(normalized_output, claim)
            entail += ais_score
            if calib:
                # ignore golden answers that are not recalled by given docs
                if union_ans_set[i] == 1:
                    calib_entail += ais_score
        
        # answered/answerable condition
        rejection = fuzz.partial_ratio(normalize_answer(REJECTION_FLAG), normalize_answer(item['output'])) > REJECTION_FUZZ_THRESHOLD

        if not rejection:
            answered_scores.append(entail / len(claims))

        if calib:
            if not rejection:    
                if any(union_ans_set):
                    calib_answered_scores.append(calib_entail / sum(union_ans_set))
                    if parametric:
                        parametric_answered_scores.append(calib_entail / len(union_ans_set))
                else:
                    calib_answered_scores.append(0)
                    if parametric:
                        parametric_answered_scores.append(0)
            
            if any(union_ans_set):
                if not rejection:
                    calib_answerable_scores.append(calib_entail / sum(union_ans_set))
                else:
                    calib_answerable_scores.append(0)            

        regular_scores.append(entail / len(claims))
    
    if calib and parametric:
        return {
            "regular_claims_nli": 100 * np.mean(regular_scores),
            "answered_claims_nli": 100 * np.mean(answered_scores if len(answered_scores) != 0 else 0),
            "calib_answered_claims_nli": 100 * np.mean(calib_answered_scores if len(calib_answered_scores) != 0 else 0),
            "calib_answerable_claims_nli": 100 * np.mean(calib_answerable_scores if len(calib_answerable_scores) != 0 else 0),
            "parametric_answered_claims_nli": 100 * np.mean(answered_scores if len(answered_scores) != 0 else 0) - 100 * np.mean(parametric_answered_scores if len(parametric_answered_scores) != 0 else 0),
        }
    elif calib:
        return {
            "regular_claims_nli": 100 * np.mean(regular_scores),
            "answered_claims_nli": 100 * np.mean(answered_scores if len(answered_scores) != 0 else 0),
            "calib_answered_claims_nli": 100 * np.mean(calib_answered_scores if len(calib_answered_scores) != 0 else 0),
            "calib_answerable_claims_nli": 100 * np.mean(calib_answerable_scores if len(calib_answerable_scores) != 0 else 0 ),
        }
    else:
        return {
            "regular_claims_nli": 100 * np.mean(regular_scores),
            "answered_claims_nli": 100 * np.mean(answered_scores if len(answered_scores) != 0 else 0),
        }


def compute_autoais(data,
                    decontext=False,
                    concat=False,
                    qampari=False,
                    at_most_citations=None):
    """
    Compute AutoAIS score.

    Args:
        data: requires field `output` and `docs`
              - docs should be a list of items with fields `title` and `text` (or `phrase` and `sent` for QA-extracted docs)
        citation: check citations and use the corresponding references.
        decontext: decontextualize the output
    """

    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto")
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)

    logger.info(f"Running AutoAIS...")

    def _format_document(doc):
        """Format document for AutoAIS."""

        if "sent" in doc:
            # QA-extracted docs
            return "Title: %s\n%s" % (doc['title'], doc['sent'])
        else:
            return "Title: %s\n%s" % (doc['title'], doc['text'])


    regular_ais_scores = []
    regular_ais_scores_prec = []
    answered_ais_scores = []
    answered_ais_scores_prec = []

    sent_total = 0
    sent_mcite = 0
    sent_mcite_support = 0
    sent_mcite_overcite = 0
    autoais_log = []
    for item in tqdm(data):
        # Get sentences by using NLTK
        if qampari:
            sents = [item['question'] + " " + x.strip() for x in item['output'].rstrip().rstrip(".").rstrip(",").split(",")]
        else:
            sents = sent_tokenize(item['output'])
        if len(sents) == 0:
            continue

        target_sents = [remove_citations(sent).strip() for sent in sents]
        
        entail = 0
        entail_prec = 0
        total_citations = 0
        for sent_id, sent in enumerate(sents):
            target_sent = target_sents[sent_id] # Citation removed and (if opted for) decontextualized
            joint_entail = -1 # Undecided

            # Find references
            ref = [int(r[1:])-1 for r in re.findall(r"\[\d+", sent)] # In text citation id starts from 1
            logger.info(f"For `{sent}`, find citations {ref}")
            if len(ref) == 0:
                # No citations
                joint_entail = 0
            elif any([ref_id >= len(item['docs']) for ref_id in ref]):
                # Citations out of range
                joint_entail = 0
            else:
                if at_most_citations is not None:
                    ref = ref[:at_most_citations]
                total_citations += len(ref)
                joint_passage = '\n'.join([_format_document(item['docs'][psgs_id]) for psgs_id in ref])

            # If not directly rejected by citation format error, calculate the recall score
            if joint_entail == -1: 
                joint_entail = _run_nli_autoais(joint_passage, target_sent)
                autoais_log.append({
                    "question": item['question'],
                    "output": item['output'],
                    "claim": sent,
                    "passage": [joint_passage],
                    "model_type": "NLI",
                    "model_output": joint_entail,
                })

            entail += joint_entail
            if len(ref) > 1:
                sent_mcite += 1

            # calculate the precision score if applicable
            if joint_entail and len(ref) > 1:
                sent_mcite_support += 1
                # Precision check: did the model cite any unnecessary documents?
                for psgs_id in ref:
                    # condition A
                    passage = _format_document(item['docs'][psgs_id]) 
                    nli_result = _run_nli_autoais(passage, target_sent)

                    # condition B
                    if not nli_result:
                        subset_exclude = copy.deepcopy(ref)
                        subset_exclude.remove(psgs_id)
                        passage = '\n'.join([_format_document(item['docs'][pid]) for pid in subset_exclude])
                        nli_result = _run_nli_autoais(passage, target_sent)
                        # check if it could support any claims within the subset_exclude
                        subset_coverage = np.bitwise_or.reduce([item['docs'][pid]['answers_found'] for pid in subset_exclude])
                        contained = False
                        for i in range(len(subset_coverage)):
                            if subset_coverage[i] == 1 and item['docs'][psgs_id]['answers_found'][i] == 1:
                                contained = True
                                break
                                
                        if nli_result and (not contained): # psgs_id is not necessary
                            flag = 0
                            sent_mcite_overcite += 1 
                            logger.info(f"For `{sent}`, exclude citation {psgs_id}")
                        else:
                            entail_prec += 1
                    else:
                        entail_prec += 1
            else:
                entail_prec += joint_entail 

        sent_total += len(sents)
        regular_ais_scores.append(entail / len(sents))
        regular_ais_scores_prec.append(entail_prec / total_citations if total_citations > 0 else 0) # len(sents))

        # answered data
        rejection = fuzz.partial_ratio(normalize_answer(REJECTION_FLAG), normalize_answer(item['output'])) > REJECTION_FUZZ_THRESHOLD
        if not rejection:
            answered_ais_scores.append(entail / len(sents))
            answered_ais_scores_prec.append(entail_prec / total_citations if total_citations > 0 else 0) # len(sents))

    if sent_mcite > 0 and sent_mcite_support > 0:
        print("Among all sentences, %.2f%% have multiple citations, among which %.2f%% are supported by the joint set, among which %.2f%% overcite." % (
            100 * sent_mcite / sent_total, 
            100 * sent_mcite_support / sent_mcite, 
            100 * sent_mcite_overcite / sent_mcite_support
        ))

    regular_recall = 100 * np.mean(regular_ais_scores)
    regular_precision = 100 * np.mean(regular_ais_scores_prec)
    regular_f1_score = 2 * (regular_precision * regular_recall) / (regular_precision + regular_recall) if (regular_precision + regular_recall) > 0 else 0
    
    answered_recall = 100 * np.mean(answered_ais_scores if len(answered_ais_scores) != 0 else 0)
    answered_precision = 100 * np.mean(answered_ais_scores_prec if len(answered_ais_scores_prec) != 0 else 0)
    answered_f1_score = 2 * (answered_precision * answered_recall) / (answered_precision + answered_recall) if (answered_precision + answered_recall) > 0 else 0

    return {
        "regular_" + "citation_rec": regular_recall,
        "regular_" + "citation_prec": regular_precision,
        "regular_" + "citation_f1": regular_f1_score,
        "answered_" + "citation_rec": answered_recall,
        "answered_" + "citation_prec": answered_precision,
        "answered_" + "citation_f1": answered_f1_score,
    }


def compute_qampari_f1(data, cot=False, prefix="", calib=False, parametric=False):
    if len(data) == 0:
        logger.warn("Warning: data should not be zero")
        return {
            prefix + "num_preds": 0,
            prefix + "qampari_prec": 0,
            prefix + "qampari_rec": 0,
            prefix + "qampari_rec_top5": 0,
            prefix + "qampari_f1": 0,
            prefix + "qampari_f1_top5": 0,
        }
    
    prec = []
    rec = []
    rec_top5 = []
    f1 = []
    f1_top5 = []

    num_preds = []
    for item in tqdm(data):
        if cot:
            if ":" in item['output']:
                o = ':'.join(item['output'].split(":")[1:]) # try to separate the COT part and the answer list part.
            else:
                o = ""
        else:
            o = item['output']
        preds = [normalize_answer(x.strip()) for x in o.rstrip().rstrip(".").rstrip(",").split(",")]
        preds = [p for p in preds if len(p) > 0] # delete empty answers
        num_preds.append(len(preds))
        
        if calib:
            # at least answerable
            union_ans_set = np.bitwise_or.reduce([doc["answers_found"] for doc in item['docs']]).tolist()
            if any(union_ans_set) and (not fuzz.partial_ratio(normalize_answer(REJECTION_FLAG), normalize_answer(item['output'])) > REJECTION_FUZZ_THRESHOLD):
                # ignore golden answers that are not recalled by given docs
                answers = [[normalize_answer(x) for x in ans] for i, ans in enumerate(item['answers']) if union_ans_set[i] == 1]
            else:
                answers = [['']]
        else:
            answers = [[normalize_answer(x) for x in ans] for ans in item['answers']]
            
        flat_answers = [item for sublist in answers for item in sublist]
        
        if calib and parametric:
            prec.append(sum([p in flat_answers for p in preds]) / len(preds) if len(preds) > 0 else 0)
            rec.append(sum([any([x in preds for x in a]) for a in answers]) / len(union_ans_set))
            rec_top5.append(min(5, sum([any([x in preds for x in a]) for a in answers])) / min(5, len(union_ans_set)))
        else:
            prec.append(sum([p in flat_answers for p in preds]) / len(preds) if len(preds) > 0 else 0)
            rec.append(sum([any([x in preds for x in a]) for a in answers]) / len(answers))
            rec_top5.append(min(5, sum([any([x in preds for x in a]) for a in answers])) / min(5, len(answers)))
            
        if (prec[-1] + rec[-1]) == 0:
            f1.append(0)
        else:
            f1.append(2 * prec[-1] * rec[-1] / (prec[-1] + rec[-1]))
        if (prec[-1] + rec_top5[-1]) == 0:
            f1_top5.append(0) 
        else:
            f1_top5.append(2 * prec[-1] * rec_top5[-1] / (prec[-1] + rec_top5[-1]))

    return {
        prefix + "num_preds": np.mean(num_preds),
        prefix + "qampari_prec": 100 * np.mean(prec),
        prefix + "qampari_rec": 100 * np.mean(rec),
        prefix + "qampari_rec_top5": 100 * np.mean(rec_top5),
        prefix + "qampari_f1": 100 * np.mean(f1),
        prefix + "qampari_f1_top5": 100 * np.mean(f1_top5),
    }
    
    
def calculate_macro_metrics(data):    
    reject_rec_num = 0
    reject_rec = 0
    reject_prec_num = 0
    reject_prec = 0

    ans_rec_num = 0
    ans_rec = 0
    ans_prec_num = 0
    ans_prec = 0
    
    for item in data:
        answerable = any(np.bitwise_or.reduce([doc["answers_found"] for doc in item['docs']]))
        rejection = fuzz.partial_ratio(normalize_answer(REJECTION_FLAG), normalize_answer(item['output'])) > REJECTION_FUZZ_THRESHOLD
        
        # Rejection recall
        if not answerable:
            reject_rec_num += 1
            if rejection:
                reject_rec += 1
        
        # Rejection precision
        if rejection:
            reject_prec_num += 1
            if not answerable:
                reject_prec += 1

        # Answerable recall
        if answerable:
            ans_rec_num += 1
            if not rejection:
                ans_rec += 1

        # Answerable precision
        if not rejection:
            ans_prec_num += 1
            if answerable:
                ans_prec += 1
    
    reject_recall = 100 * reject_rec / reject_rec_num if reject_rec_num > 0 else 0
    reject_precision = 100 * reject_prec / reject_prec_num if reject_prec_num > 0 else 0
    reject_f1_score = 2 * (reject_precision * reject_recall) / (reject_precision + reject_recall) if (reject_precision + reject_recall) > 0 else 0

    ans_recall = 100 * ans_rec / ans_rec_num if ans_rec_num > 0 else 0
    ans_precision = 100 * ans_prec / ans_prec_num if ans_prec_num > 0 else 0
    ans_f1_score = 2 * (ans_precision * ans_recall) / (ans_precision + ans_recall) if (ans_precision + ans_recall) > 0 else 0
    
    return {
        "reject_rec": reject_recall, 
        "reject_prec": reject_precision, 
        "reject_f1": reject_f1_score,
        "answerable_rec": ans_recall,
        "answerable_prec": ans_precision,
        "answerable_f1": ans_f1_score,
        "macro_avg": np.mean([reject_recall, ans_recall]),
        "macro_f1": np.mean([reject_f1_score, ans_f1_score])
    }
    

def calculate_incorrect_frequency(answered_data):
    
    if len(answered_data) == 0:
        logger.warn("Warning: answered_data should not be zero")
        return {
            "qampari_presence": 0.0,
            "qampari_absence": 0.0,
        }
    
    presence_list = []
    absence_list = []
    for item in answered_data:
        union_ans_set = np.bitwise_or.reduce([doc["answers_found"] for doc in item['docs']]).tolist()
        calib_ground_truths = []  
        for i, ans in enumerate(item['answers']):
            # ignore golden answers that are not recalled by given docs
            if union_ans_set[i] == 1:
                calib_ground_truths.append(ans)
        
        o = item['output']        
        preds = [x.strip() for x in o.rstrip().rstrip(".").rstrip(",").split(",")]
        preds = [p for p in preds if len(p) > 0] # delete empty answers
        
        # detect correct/incorrect
        ans_correctness = []
        for p in preds:
            ans_correctness.append(any([exact_presence(gts, p) for gts in calib_ground_truths]))
                    
        # detect in/not in docs
        ans_existence = []
        for p in preds:
            ans_existence.append(any([exact_presence([p], doc['text']) for doc in item['docs']]))      

        ans_correctness = np.array(ans_correctness)
        ans_existence = np.array(ans_existence)
        if any(ans_correctness == 0):
            presence_list.append(np.sum((ans_correctness == 0) & (ans_existence == 1)) / sum(ans_correctness == 0))
            absence_list.append(np.sum((ans_correctness == 0) & (ans_existence == 0)) / sum(ans_correctness == 0))
        
    return {
        "qampari_presence": 100 * np.mean(presence_list),
        "qampari_absence": 100 * np.mean(absence_list),
    }
    
    
class LLM:

    def __init__(self, args):
        self.args = args

        if args.openai_api:
            import openai 
            OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
            OPENAI_ORG_ID = os.environ.get("OPENAI_ORG_ID")
            OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")

            if args.azure:
                openai.api_key = OPENAI_API_KEY
                openai.api_base = OPENAI_API_BASE
                openai.api_type = 'azure'
                openai.api_version = '2023-05-15' 
            else: 
                openai.api_key = OPENAI_API_KEY
                openai.organization = OPENAI_ORG_ID

            self.tokenizer = AutoTokenizer.from_pretrained("gpt2", fast_tokenizer=False) # TODO: For ChatGPT we should use a different one
            # To keep track of how much the API costs
            self.prompt_tokens = 0
            self.completion_tokens = 0
        elif args.vllm:
            self.chat_llm, self.tokenizer, self.sampling_params = load_vllm(args.model, args)
        else:
            self.model, self.tokenizer = load_model(args.model, lora_path=args.lora_path)
        
        self.prompt_exceed_max_length = 0
        self.fewer_than_50 = 0
        self.azure_filter_fail = 0


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
            # inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
            inputs = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.model.device)
            # stop = [] if stop is None else stop
            # stop = list(set(stop + ["\n", "Ċ", "ĊĊ", "<0x0A>"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
            # stop_token_ids = list(set([self.tokenizer._convert_token_to_id(stop_token) for stop_token in stop] + [self.model.config.eos_token_id]))
            # if "llama" in args.model.lower():
            #     stop_token_ids.remove(self.tokenizer.unk_token_id)
            outputs = self.model.generate(
                **inputs,
                do_sample=True, temperature=args.temperature, top_p=args.top_p, 
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                eos_token_id=[self.model.config.eos_token_id]
            )
            generation = self.tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
            return generation.strip()


    def batch_generate(self, prompts, stop=None):
        args = self.args
        
        if args.vllm:
            inputs = [self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False) for prompt in prompts]
            self.sampling_params.n = 1  # Number of output sequences to return for the given prompt
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
            

def run_model(args: RUN_Config):
    if "turbo" in args.model:
        # ChatGPT has a longer max length
        args.max_length = 4096
    if "16k" in args.model:
        args.max_length = 16384
    elif "32k" in args.model:
        args.max_length = 32768
    elif "turbo" in args.model:
        args.max_length = 4096
    elif "gpt4" in args.model or "gpt-4" in args.model:
        args.max_length = 8192
    elif "llama-2" in args.model.lower() or "llama2" in args.model.lower():
        args.max_length = 4096


    logger.info(f"Set the model max length to {args.max_length} (if not correct, check the code)")
    
    # Load the model or setup the API
    llm = LLM(args)
    
    # Generate prompts
    np.random.seed(args.seed)
    
    # Load data
    prompt_data = json.load(open(args.prompt_file))
    eval_data = json.load(open(args.eval_file))
    
    # Generate the demonstration part
    head_prompt = ""
    if not args.no_demo:
        if "rejection" in args.prompt_file:
            logger.warning("Using rejection head prompts...")
            pos_train_ids = np.random.choice(len(prompt_data["positive_demos"]), ceil(args.shot/2), replace=False)
            rej_train_ids = np.random.choice(len(prompt_data["reject_demos"]), args.shot//2, replace=False)

            train_items = []
            for pos_train_id in pos_train_ids:
                train_items.append(prompt_data["positive_demos"][pos_train_id])
            for rej_train_id in rej_train_ids:
                train_items.append(prompt_data["reject_demos"][rej_train_id])
            random.shuffle(train_items)
            for train_item in train_items:
                ndoc = args.ndoc
                if args.no_doc_in_demo:
                    ndoc = 0
                elif args.fewer_doc_in_demo:
                    assert args.ndoc_in_demo is not None
                    ndoc = args.ndoc_in_demo
                head_prompt += make_demo(
                    train_item, prompt=prompt_data["demo_prompt"], ndoc=ndoc, doc_prompt=prompt_data["doc_prompt"], 
                    instruction=prompt_data["instruction"], use_shorter=None
                )
                head_prompt += prompt_data["demo_sep"]
                
        else:    
            train_ids = np.random.choice(len(prompt_data["demos"]), args.shot, replace=False)
            for train_id in train_ids:
                train_item = prompt_data["demos"][train_id]
                ndoc = args.ndoc
                if args.no_doc_in_demo:
                    ndoc = 0
                elif args.fewer_doc_in_demo:
                    assert args.ndoc_in_demo is not None
                    ndoc = args.ndoc_in_demo
                head_prompt += make_demo(
                    train_item, prompt=prompt_data["demo_prompt"], ndoc=ndoc, doc_prompt=prompt_data["doc_prompt"], 
                    instruction=prompt_data["instruction"], use_shorter=None
                )
                head_prompt += prompt_data["demo_sep"]
            
    # Sample quick test
    if args.quick_test is not None:
        eval_ids = np.random.choice(len(eval_data), args.quick_test, replace=False)
        eval_data = [eval_data[int(idx)] for idx in eval_ids]

    logger.info("Generating prompts...") 
    incomplete_doc_list = 0 # For some questions there might be fewer than ndoc documents
    for idx, eval_item in enumerate(tqdm(eval_data)):
        eval_data[idx]['prompt'] = head_prompt + make_demo(
            eval_item, prompt=prompt_data["demo_prompt"], ndoc=args.ndoc, doc_prompt=prompt_data["doc_prompt"],
            instruction=prompt_data["instruction"], use_shorter=None, 
            test=True
        )
        doc_list = eval_item["docs"][:args.ndoc]
        # Trim original docs by ndoc /and filtered if using summary/extraction for saving
        eval_data[idx]['docs'] = doc_list
        if len(doc_list) < args.ndoc:
            incomplete_doc_list += 1
    logger.info("Done.")
    if incomplete_doc_list > 0:
        logger.warning(f"There are {incomplete_doc_list} questions that have incomplete document list (may due to a lot of them are filtered out by summary/extraction).")


    # Response generation: process a batch of items
    if args.vllm:
        prompts = [item['prompt'] for item in eval_data for _ in range(args.num_samples)]
        prompt_lengths = [len(llm.tokenizer.tokenize(prompt)) for prompt in prompts]
        max_prompt_len = max(prompt_lengths)

        if idx == 0:
            print(prompts[0])
        
        # Generate outputs in batch
        logger.info(f"Max_N: {max_prompt_len}")
        batch_outputs = llm.batch_generate(prompts)
        
        # release vllm
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
        del llm.chat_llm.llm_engine.model_executor.driver_worker
        torch.cuda.empty_cache()
        
        # Post-process each output
        for i in range(len(eval_data)):
            output_array = []
            for j, output in enumerate(batch_outputs[i:i + args.num_samples]):
                output_array.append(output)
                output_array[-1] = output_array[-1].replace("<|im_end|>", "").rstrip()
                if output_array[-1].endswith("End."):
                    output_array[-1] = output_array[-1][:-len("End.")]

            eval_data[i]['output'] = output_array if len(output_array) > 1 else output_array[0]

    else:
        for idx, item in enumerate(tqdm(eval_data)):
            prompt = item['prompt']
            prompt_len = len(llm.tokenizer.tokenize(prompt))

            if idx == 0:
                print(prompt)

            output_array = []
            for _ in range(args.num_samples):
            
                logger.info(f"N: {prompt_len}")
                output_array.append(llm.generate(prompt, min(args.max_new_tokens, args.max_length-prompt_len)))
                item['prompt'] = prompt
                
                output_array[-1] = output_array[-1].replace("<|im_end|>", "").rstrip()
                if output_array[-1].endswith("End."):
                    output_array[-1] = output_array[-1][:-len("End.")]
            
            item['output'] = output_array if len(output_array) > 1 else output_array[0]
    
    
    # Statistics    
    logger.info(f"#Cases when prompts exceed max length: {llm.prompt_exceed_max_length}")
    logger.info(f"#Cases when max new tokens < 50: {llm.fewer_than_50}")
    
    # Save the result
    model_name = args.model
    if args.lora_path:
        model_name = args.lora_path
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    name = f"{model_name}-{args.eval_type}-shot{args.shot}-ndoc{args.ndoc}-{args.seed}-temp{args.temperature}"
    if args.azure:
        name += "-azure"
    if args.quick_test is not None:
        name += f"-quick_test{args.quick_test}"
    if args.no_doc_in_demo:
        name += "-no_doc_in_demo"
    if args.fewer_doc_in_demo:
        name += f"-{args.ndoc_in_demo}_doc_in_demo"
    if args.num_samples > 1:
        name += f"-sample{args.num_samples}"
    if args.no_demo:
        name += f"-no_demo"

    
    eval_data = {
        "args": args.__dict__,
        "data": eval_data,
    }
    if args.openai_api:
        logger.info(f"Token used: prompt {llm.prompt_tokens}; completion {llm.completion_tokens}")
        if "turbo" in args.model:
            p_price, c_price = 0.0015, 0.002
            if "16k" in args.model:
                p_price, c_price = 0.003, 0.004
        elif "gpt4" in args.model or "gpt-4" in args.model:
            p_price, c_price = 0.03, 0.06
            if "32k" in args.model:
                p_price, c_price = 0.06, 0.12
        else:
            logger.warn("Cannot find model price")
            p_price, c_price = 0, 0

        eval_data["total_cost"] = llm.prompt_tokens / 1000 * p_price + llm.completion_tokens / 1000 * c_price        

        logger.info(f"Unit price (Oct 16, 2023, prompt/completion): {p_price}/{c_price}")
        logger.info(f"Total cost: %.1f" % (eval_data["total_cost"]))

        if args.azure:
            eval_data["azure_filter_fail"] = llm.azure_filter_fail 

    output_dir = Path(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    json.dump(eval_data, open(output_dir.joinpath(name + ".json"), "w"), indent=4)
    
    return eval_data


def eval_model(eval_data, args: EVAL_Config):
    data = eval_data['data']
    
    if "em@5" in args.output_path:
        qampari = True
    else:
        qampari = False

    # Truncate by newline and remove on the fly search result
    logger.warning("We remove all the pre/appended space/newlines and we truncate the answer by the first newline.")
    logger.warning("We replace any on the fly search result to standard bracket citation format.")
    
    answered_data = []
    answerable_data = []
    for idx, item in enumerate(data):
        rejection = fuzz.partial_ratio(normalize_answer(REJECTION_FLAG), normalize_answer(item['output'])) > REJECTION_FUZZ_THRESHOLD
        answerable = any(np.bitwise_or.reduce([doc["answers_found"] for doc in item['docs']]))

        if not rejection:
            answered_data.append(copy.deepcopy(item))

        if answerable:
            answerable_data.append(copy.deepcopy(item))
    
    # Remove all citations for all non-AutoAIS evaluation
    normalized_data = copy.deepcopy(data)
    normalized_answered_data = copy.deepcopy(answered_data)
    normalized_answerable_data = copy.deepcopy(answerable_data)
    for i in range(len(normalized_data)):
        normalized_data[i]['output'] = remove_citations(normalized_data[i]['output'])
    for i in range(len(normalized_answered_data)):
        normalized_answered_data[i]['output'] = remove_citations(normalized_answered_data[i]['output'])
    for i in range(len(normalized_answerable_data)):
        normalized_answerable_data[i]['output'] = remove_citations(normalized_answerable_data[i]['output'])


    result = {}
    # all data points
    result['answered_num'] = len(normalized_answered_data)
    result['answerable_num'] = len(normalized_answerable_data)
    result['overlapped_num'] = len([item for item in normalized_answered_data if any(np.bitwise_or.reduce([doc["answers_found"] for doc in item['docs']]))])
    result['regular_length'] = compute_len(normalized_data)
    result['answered_length'] = compute_len(normalized_answered_data)
    # for answerable and answered
    result['regular_str_em'], result['regular_str_hit'] = compute_str_em(normalized_data)
    result['answered_str_em'], result['answered_str_hit'] = compute_str_em(normalized_answered_data)
    result['calib_answered_str_em'], result['calib_answered_str_hit'] = compute_str_em(normalized_answered_data, calib=True)
    result['calib_answerable_str_em'], result['calib_answerable_str_hit'] = compute_str_em(normalized_answerable_data, calib=True)
    result['parametric_str_em'], result['parametric_str_hit'] = compute_str_em(normalized_answered_data, calib=True, parametric=True)
    result['parametric_str_em'], result['parametric_str_hit'] = result['answered_str_em'] - result['parametric_str_em'], result['answered_str_hit'] - result['parametric_str_hit']

    # show rejection rate
    macro_dict = calculate_macro_metrics(data)
    logger.critical(f"Macro metrics: {repr(macro_dict)}")
    args.output_path = args.output_path.replace(".score", "-reject.score")
    result.update(macro_dict)
    
    if qampari:
        result.update(calculate_incorrect_frequency(normalized_answered_data))
        result.update(compute_qampari_f1(normalized_data, cot=False, prefix="regular_"))
        result.update(compute_qampari_f1(normalized_answered_data, cot=False, prefix="answered_"))
        result.update(compute_qampari_f1(normalized_answered_data, cot=False, prefix="calib_answered_", calib=True))
        result.update(compute_qampari_f1(normalized_answerable_data, cot=False, prefix="calib_answerable_", calib=True))
        result['parametric_qampari_rec_top5'] = result['answered_qampari_rec_top5'] - compute_qampari_f1(normalized_answered_data, cot=False, prefix="parametric_", calib=True, parametric=True)['parametric_qampari_rec_top5']

    if args.citations: 
        result.update(compute_autoais(data, qampari=qampari, at_most_citations=args.at_most_citations))
    if args.claims_nli:
        result.update(compute_claims(normalized_data, calib=True, parametric=True))
        # result.update(compute_claims(normalized_data, calib=True))
    
    print(result)
    
    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=4)
    
    return result
    

def TRUST_SCORE(run_config: RUN_Config):
    
    eval_data = run_model(run_config)
    
    model = run_config.model.split("/")[-1]
    eval_config = EVAL_Config(output_path=f"{run_config.output_dir}/{model}-{run_config.eval_type}-temp{run_config.temperature}.score")
    if run_config.eval_type == "em":
        eval_config.update_from_dict(
            {
                "citations": True,
            }
        )
    elif run_config.eval_type == "em@5":
        eval_config.update_from_dict(
            {
                "citations": True,
            }
        )
    elif run_config.eval_type == "cm":
        eval_config.update_from_dict(
            {
                "citations": True,
                "claims_nli": True,
            }
        )
        
    result = eval_model(eval_data, eval_config)
    
    return result