import collections
import copy
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.stats as stats
import torch
from fuzzywuzzy import fuzz
from nltk import sent_tokenize
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .logging_config import logger
from .utils import *

global autoais_model, autoais_tokenizer
autoais_model, autoais_tokenizer = None, None

def compute_len(data):
    """Compute average length of predictions."""

    if len(data) == 0:
        logger.warning("Warning: data should not be zero")
        return 0
    
    res, cntr = 0, 0
    for item in data:
        res += len(item["output"].split())
        cntr += 1
    return res / cntr


def compute_exact_match(data, args, calib=False, parametric=False):
    """
    Computes exact match (STR-EM) and hit metrics (STR-EM-HIT) (only for ASQA) for given data.

    Evaluates how well model predictions match gold answers, optionally filtering answers 
    based on document recall (calibration). Returns metrics for regular, calibrated, and 
    parametric evaluations.

    Parameters:
    ----------
    data : list of dict
        Dataset to evaluate, where each item includes:
        - `answers`: List of gold answer lists.
        - `output`: Model's prediction as a string.
        - `docs` (optional): Document metadata for calibration, containing `answers_found`.

    calib : bool, optional
        If True, considers only answers recalled in the documents for evaluation.

    parametric : bool, optional
        If True, uses stricter conditions for calibrated evaluation.

    Returns:
    -------
    dict
        Metrics for exact match and hits:
        - `regular_str_em` and `regular_str_hit` (all data).
        - `calib_answered_str_em` and `calib_answered_str_hit` (if `calib=True`).
        - `parametric_str_em` and `parametric_str_hit` (if `parametric=True`).

    """
    
    if len(data) == 0:
        logger.warning("Warning: data should not be zero")
        return 0, 0

    if 'answers' not in data[0] or data[0]['answers'] is None:
        logger.warning("Warning: no answers found in data")
        return 0, 0

    acc = []
    hit = []

    for item in tqdm(data):
        loc_acc = []
        if calib:
            # at least answerable
            union_ans_set = np.bitwise_or.reduce([doc["answers_found"] for doc in item['docs']]).tolist()
            if any(union_ans_set) and (not fuzz.partial_ratio(normalize_answer(args.rejection_flag), normalize_answer(item['output'])) > args.rejection_threshold):
                for i, ans_list in enumerate(item['answers']):
                    # ignore golden answers that are not recalled by given docs
                    if union_ans_set[i] == 1:
                        loc_acc.append(_exact_presence(ans_list, item["output"]))
            else:
                loc_acc.append(False)
        else:
            for ans_list in item['answers']:
                loc_acc.append(_exact_presence(ans_list, item["output"]))
                
        if calib and parametric:
            acc.append(np.sum(a=loc_acc)/len(union_ans_set))
            hit.append( int(np.sum(loc_acc)/len(union_ans_set) == 1) )
        else:
            acc.append(np.mean(loc_acc))
            hit.append( int(np.mean(loc_acc) == 1) )

    return 100 * np.mean(acc), 100 * np.mean(hit)


def get_all_em_scores(data: List[Dict], normalized_data: List[Dict], normalized_answered_data: List[Dict], normalized_answerable_data: List[Dict], args):
    result = {}
    result['regular_str_em'], result['regular_str_hit'] = compute_exact_match(normalized_data, args)
    result['answered_str_em'], result['answered_str_hit'] = compute_exact_match(normalized_answered_data, args)

    if args.calib:
        result['calib_answered_str_em'], result['calib_answered_str_hit'] = compute_exact_match(normalized_answered_data, args, calib=True)
        result['calib_answerable_str_em'], result['calib_answerable_str_hit'] = compute_exact_match(normalized_answerable_data, args, calib=True)
        result['calib_str_em_f1'] = stats.hmean([result['calib_answered_str_em'], result['calib_answerable_str_em']])

    if args.parametric:
        result['parametric_str_em'], result['parametric_str_hit'] = compute_exact_match(normalized_answered_data, args, calib=True, parametric=True)
        result['parametric_str_em'], result['parametric_str_hit'] = result['answered_str_em'] - result['parametric_str_em'], result['answered_str_hit'] - result['parametric_str_hit']
    
    return result


def compute_claim_match(data, args, calib=False, parametric=False):
    
    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(args.autoais_model, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto")
        autoais_tokenizer = AutoTokenizer.from_pretrained(args.autoais_model, use_fast=False)

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
        rejection = fuzz.partial_ratio(normalize_answer(args.rejection_flag), normalize_answer(item['output'])) > args.rejection_threshold

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
            "calib_claims_nli_f1": stats.hmean([100 * np.mean(calib_answered_scores if len(calib_answered_scores) != 0 else 0), 100 * np.mean(calib_answerable_scores if len(calib_answerable_scores) != 0 else 0)])
        }
    elif calib:
        return {
            "regular_claims_nli": 100 * np.mean(regular_scores),
            "answered_claims_nli": 100 * np.mean(answered_scores if len(answered_scores) != 0 else 0),
            "calib_answered_claims_nli": 100 * np.mean(calib_answered_scores if len(calib_answered_scores) != 0 else 0),
            "calib_answerable_claims_nli": 100 * np.mean(calib_answerable_scores if len(calib_answerable_scores) != 0 else 0 ),
            "calib_claims_nli_f1": stats.hmean([100 * np.mean(calib_answered_scores if len(calib_answered_scores) != 0 else 0), 100 * np.mean(calib_answerable_scores if len(calib_answerable_scores) != 0 else 0)])
        }
    else:
        return {
            "regular_claims_nli": 100 * np.mean(regular_scores),
            "answered_claims_nli": 100 * np.mean(answered_scores if len(answered_scores) != 0 else 0),
        }


def get_all_cm_scores(data: List[Dict], normalized_data: List[Dict], normalized_answered_data: List[Dict], normalized_answerable_data: List[Dict], args):
    return compute_claim_match(normalized_data, args, calib=args.calib, parametric=args.parametric)


def compute_qampari_f1(data, args, cot=False, prefix="", calib=False, parametric=False):
    if len(data) == 0:
        logger.warning("Warning: data should not be zero")
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
            if any(union_ans_set) and (not fuzz.partial_ratio(normalize_answer(args.rejection_flag), normalize_answer(item['output'])) > args.rejection_threshold):
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


def get_all_em5_scores(data: List[Dict], normalized_data: List[Dict], normalized_answered_data: List[Dict], normalized_answerable_data: List[Dict], args):
    result = {}
    result.update(_compute_incorrect_frequency(normalized_answered_data))
    result.update(compute_qampari_f1(normalized_data, args, prefix="regular_"))
    result.update(compute_qampari_f1(normalized_answered_data, args, prefix="answered_"))
    
    if args.calib:
        result.update(compute_qampari_f1(normalized_answered_data, args, prefix="calib_answered_", calib=True))
        result.update(compute_qampari_f1(normalized_answerable_data, args, prefix="calib_answerable_", calib=True))
        result["calib_qampari_em_f1"] = stats.hmean([result['calib_answered_qampari_f1_top5'], result['calib_answerable_qampari_f1_top5']])

    if args.parametric:
        result['parametric_qampari_rec_top5'] = result['answered_qampari_rec_top5'] - compute_qampari_f1(normalized_answered_data, args, prefix="parametric_", calib=True, parametric=True)['parametric_qampari_rec_top5']
    return result


def compute_citation_metrics(data,
                             args,
                             decontext=False,
                             concat=False,
                             is_qampari=False,
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
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(args.autoais_model, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto")
        autoais_tokenizer = AutoTokenizer.from_pretrained(args.autoais_model, use_fast=False)

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
        if is_qampari:
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
        rejection = fuzz.partial_ratio(normalize_answer(args.rejection_flag), normalize_answer(item['output'])) > args.rejection_threshold
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


def get_citation_scores(data: List[Dict], normalized_data: List[Dict], normalized_answered_data: List[Dict], normalized_answerable_data: List[Dict], args):
    result = {}
    result.update(compute_citation_metrics(data, args, is_qampari=args.is_qampari, at_most_citations=args.at_most_citations))
    return result

    
def compute_macro_metrics(data, args):    
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
        rejection = fuzz.partial_ratio(normalize_answer(args.rejection_flag), normalize_answer(item['output'])) > args.rejection_threshold
        
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


def get_macro_scores(data: List[Dict], normalized_data: List[Dict], normalized_answered_data: List[Dict], normalized_answerable_data: List[Dict], args):
    return compute_macro_metrics(data, args)
    

def _compute_incorrect_frequency(answered_data):
    
    if len(answered_data) == 0:
        logger.warning("Warning: answered_data should not be zero")
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
            ans_correctness.append(any([_exact_presence(gts, p) for gts in calib_ground_truths]))
                    
        # detect in/not in docs
        ans_existence = []
        for p in preds:
            ans_existence.append(any([_exact_presence([p], doc['text']) for doc in item['docs']]))      

        ans_correctness = np.array(ans_correctness)
        ans_existence = np.array(ans_existence)
        if any(ans_correctness == 0):
            presence_list.append(np.sum((ans_correctness == 0) & (ans_existence == 1)) / sum(ans_correctness == 0))
            absence_list.append(np.sum((ans_correctness == 0) & (ans_existence == 0)) / sum(ans_correctness == 0))
        
    return {
        "qampari_presence": 100 * np.mean(presence_list),
        "qampari_absence": 100 * np.mean(absence_list),
    }
    

def _exact_presence(short_answers, context):
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


def _compute_f1(a_gold, a_pred):
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


def _compute_exact(a_gold, a_pred):
    """Check whether two strings are equal up to normalization."""

    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


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

def compute_answered_ratio(data: List[Dict], normalized_data: List[Dict], normalized_answered_data: List[Dict], normalized_answerable_data: List[Dict], args):
    return round(100 * len(normalized_answered_data) / len(normalized_data), 2)

def get_basic_stats(data: List[Dict], normalized_data: List[Dict], normalized_answered_data: List[Dict], normalized_answerable_data: List[Dict], args):
    result = {}
    result['answered_ratio'] = round(100 * len(normalized_answered_data) / len(normalized_data), 2)
    result['answered_num'] = len(normalized_answered_data)
    result['answerable_num'] = len(normalized_answerable_data)
    result['overlapped_num'] = len([item for item in normalized_answered_data if any(np.bitwise_or.reduce([doc["answers_found"] for doc in item['docs']]))])
    result['regular_length'] = compute_len(normalized_data)
    result['answered_length'] = compute_len(normalized_answered_data)
    return result

def compute_trust_score(result, args):
    if args.eval_type == "em":
        result['trust_score'] = np.mean([result['calib_str_em_f1'], result['macro_f1'], result['answered_citation_f1']])
    elif args.eval_type == "em@5":
        result['trust_score'] = np.mean([result['calib_qampari_em_f1'], result['macro_f1'], result['answered_citation_f1']])
    elif args.eval_type == "cm":
        result['trust_score'] = np.mean([result['calib_claims_nli_f1'], result['macro_f1'], result['answered_citation_f1']])
    
    return result