import argparse
import random
import numpy as np
import copy
from tqdm import tqdm
import json
import os
import colorlog
import re
from fuzzywuzzy import fuzz

import torch
from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from utils import get_max_memory, normalize_answer, remove_citations, PunktLanguageVars, save_data_to_json

# Set up the Punkt tokenizer with custom abbreviations
abbreviations = [
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Rev", "Fr", "Sr", "Jr", "St", "Co", "Corp", "Inc", "Ltd", 
    "Mt", "Blvd", "Ave", "Rd", "Ln", "Ct", "Pl", "Sq", "Ste", "Bldg", "Apt", "Dept", "No", "Nos", 
    "vs", "etc", "eg", "ie", "cf", "al", "a.m", "p.m", "U.S", "U.K", "Gen", "Maj", "Col", "Lt", 
    "Cmdr", "Capt", "Sgt", "Cpl", "Pvt"
]
punkt_param = PunktParameters()
punkt_param.abbrev_types = set(abbrev.lower() for abbrev in abbreviations)
sentence_splitter = PunktSentenceTokenizer(punkt_param, lang_vars=PunktLanguageVars())

# Logger
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

# Define globals
QA_MODEL="gaotianyu1350/roberta-large-squad"
AUTOAIS_MODEL="google/t5_xxl_true_nli_mixture"

global autoais_model, autoais_tokenizer
autoais_model, autoais_tokenizer = None, None

REJECTION_FUZZ_THRESHOLD=85
REJECTION_FLAG="I apologize, but I couldn't find an answer"

# rejection errors
def is_rejection_prec_error(answer, pattern):
    rejection = fuzz.partial_ratio(normalize_answer(REJECTION_FLAG), normalize_answer(answer)) > REJECTION_FUZZ_THRESHOLD
    answerable = not(np.all(pattern == 0))
    if answerable and rejection:
        return True
    return False

def is_rejection_recall_error(answer, pattern):
    rejection = fuzz.partial_ratio(normalize_answer(REJECTION_FLAG), normalize_answer(answer)) > REJECTION_FUZZ_THRESHOLD
    answerable = not(np.all(pattern == 0))
    if not answerable and not rejection:
        return True
    return False

def not_invalid_qns(pattern):
    return not np.all(pattern == 0)

def not_reject(answer):
    return not fuzz.partial_ratio(normalize_answer(REJECTION_FLAG), normalize_answer(answer)) > REJECTION_FUZZ_THRESHOLD

# asqa correctness
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
            return True, ans

    return False, ""

def compute_str_em(sample, pattern):
    """Compute STR-EM metric (only for ASQA)
    Args:
        data: requires field `qa_pairs/short_answers` and `output`
    Returns:
        STR-EM and STR-EM-HIT ()
    """
    upvoted_content_labels, downvoted_content_labels = [], []

    if 'qa_pairs' not in sample or sample['qa_pairs'] is None:
        return 0, 0

    matched = [0 for i in range(len(pattern))]
    sents = sentence_splitter.tokenize(sample['output'])
    
    for sent_idx, sent in enumerate(sents):
        normalized_sent = remove_citations(sent)
        for idx, qa_pair in enumerate(sample['qa_pairs']):
            contained, ans = exact_presence(qa_pair['short_answers'], normalized_sent)
            doc_support = (pattern[idx] == 1)
            logger.debug(f'{contained=}, {doc_support=}')
            if contained and doc_support:
                matched[idx] = 1
                upvoted_content_labels.append([sent_idx, sent, ans])
                break
        if not contained or not doc_support:
            downvoted_content_labels.append([sent_idx, sent])

    coverage = sum(matched) / sum(pattern)
    return coverage, upvoted_content_labels, downvoted_content_labels

# eli5 citation recall and precision
def compute_claims(sample, pattern):

    upvoted_content_labels, downvoted_content_labels = [], []

    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto")
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)

    logger.info("Computing claims...")
    entail = [0 for i in range(len(pattern))]
    sents = sentence_splitter.tokenize(sample['output'])
    claims = sample["claims"]
    
    for sent_idx, sent in enumerate(sents):
        normalized_sent = remove_citations(sent)
        logger.debug(f'{sent=}')
        for idx, claim in enumerate(claims):
            logger.debug(f'{claim=}')
            contained = _run_nli_autoais(normalized_sent, claim)
            doc_support = (pattern[idx] == 1)
            logger.debug(f'{contained=}, {doc_support=}')
            if contained and doc_support:
                entail[idx] = 1
                upvoted_content_labels.append([sent_idx, sent, claim])
                break
        if not contained or not doc_support:
            downvoted_content_labels.append([sent_idx, sent])
    
    score = sum(entail) / sum(pattern)
    logger.debug(f'{entail=}, {pattern=}, {score=}')
        
    return score, upvoted_content_labels, downvoted_content_labels

# qampari correctness
def compute_qampari_f1(sample, pattern):
    prec = 0
    rec = 0
    rec_top5 = 0
    f1 = 0
    f1_top5 = 0
    num_preds = 0

    preds = [x.strip() for x in sample['output'].rstrip().rstrip(".").rstrip(",").split(",")]
    preds = [p for p in preds if len(p) > 0] # delete empty answers
    normalized_preds = [normalize_answer(remove_citations(p)) for p in preds]
    num_preds = len(normalized_preds)
    upvoted_content_labels, downvoted_content_labels = [], []

    answers = [[normalize_answer(x) for x in ans] for ans in sample['answers']]

    matched = [0 for i in range(len(pattern))]
    for p_idx, p in enumerate(preds):
        normalized_p = normalized_preds[p_idx]
        for idx, ans_subls in enumerate(answers):
            contained = normalized_p in ans_subls
            doc_support = (pattern[idx] == 1)
            if contained and doc_support:
                matched[idx] = 1
                upvoted_content_labels.append([p_idx, normalized_p, None])
                break
        if not contained or not doc_support:
            downvoted_content_labels.append([p_idx, p])
    
    rec= sum(matched)/ sum(pattern) # how many correct was captured
    rec_top5 = min(5, sum(matched))/ min(5, sum(pattern))
    logger.debug(f'{normalized_preds=}, {answers=}, {pattern=}, {matched=}, {rec_top5=}')
    
    # only select for answers supported by docs
    filtered_answers = []
    for idx, a in enumerate(answers):
        if pattern[idx] == 1:
            filtered_answers.append(a)
    filtered_flat_answers = [sample for sublist in filtered_answers for sample in sublist]
    p_matched = []
    for p in normalized_preds:
        if p in filtered_flat_answers:
            p_matched.append(1)
        else:
            p_matched.append(0)
    prec= sum(p_matched) / len(normalized_preds) if len(normalized_preds) > 0 else 0 # how many of output is correct

    if (prec + rec) == 0:
        f1 = 0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    if (prec + rec_top5) == 0:
        f1_top5 = 0
    else:
        f1_top5 = 2 * prec * rec_top5 / (prec + rec_top5)
    
    return {
        "num_preds": num_preds,
        "qampari_prec":  prec,
        "qampari_rec":  rec,
        "qampari_rec_top5":  rec_top5,
        "qampari_f1":  f1,
        "qampari_f1_top5":  f1_top5,
    }, upvoted_content_labels, downvoted_content_labels

# asqa and eli5 and qampari citation recall and precision
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

def compute_autoais(sample,
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

     # Get sentences by using NLTK
    
    if qampari:
        sents = [sample['question'] + " " + x.strip() for x in sample['output'].rstrip().rstrip(".").rstrip(",").split(",")]
        # print(f'{sents=}')
    else:
        sents = sentence_splitter.tokenize(sample['output'])
    if len(sents) == 0:
        raise NotImplementedError

    target_sents = [remove_citations(sent).strip() for sent in sents]

    ais_scores = 0 
    ais_scores_prec = 0
   
    sent_total = 0
    sent_mcite = 0
    sent_mcite_support = 0
    sent_mcite_overcite = 0
    autoais_log = []

    entail = 0
    entail_prec = 0
    total_citations = 0

    upvoted_cite_labels, downvoted_cite_labels = [], []

    for sent_id, sent in enumerate(sents):
        candidate_upvote, candidate_downvote = [], []

        target_sent = target_sents[sent_id] # Citation removed and (if opted for) decontextualized
        joint_entail = -1 # Undecided

        # Find references
        ref = [int(r[1:])-1 for r in re.findall(r"\[\d+", sent)] # In text citation id starts from 1
        logger.info(f"For `{sent}`, find citations {[i+1 for i in ref]}")
        if len(ref) == 0:
            # No citations
            joint_entail = 0
        elif any([ref_id >= len(sample['docs']) for ref_id in ref]):
            # Citations out of range
            joint_entail = 0
            candidate_downvote.extend([i+1 for i in ref])
        else:
            if at_most_citations is not None:
                ref = ref[:at_most_citations]
            total_citations += len(ref)
            joint_passage = '\n'.join([_format_document(sample['docs'][psgs_id]) for psgs_id in ref])

        # If not directly rejected by citation format error, calculate the recall score
        if joint_entail == -1: 
            joint_entail = _run_nli_autoais(joint_passage, target_sent)
            autoais_log.append({
                "question": sample['question'],
                "output": sample['output'],
                "claim": sent,
                "passage": [joint_passage],
                "model_type": "NLI",
                "model_output": joint_entail,
                "purpose": "citation recall"
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
                passage = _format_document(sample['docs'][psgs_id]) 
                nli_result = _run_nli_autoais(passage, target_sent)
                autoais_log.append({
                    "question": sample['question'],
                    "output": sample['output'],
                    "claim": sent,
                    "passage": [passage],
                    "model_type": "NLI",
                    "model_output": nli_result,
                    "purpose": "citation precision, does this doc support the claim?"
                })

                # condition B
                if not nli_result:
                    subset_exclude = copy.deepcopy(ref)
                    subset_exclude.remove(psgs_id)
                    passage = '\n'.join([_format_document(sample['docs'][pid]) for pid in subset_exclude])
                    nli_result = _run_nli_autoais(passage, target_sent)
                    autoais_log.append({
                        "question": sample['question'],
                        "output": sample['output'],
                        "claim": sent,
                        "passage": [passage],
                        "model_type": "NLI",
                        "model_output": nli_result,
                        "purpose": "citation precision, can the remaining docs support claim?"
                    })
                    
                    # check if it could support any claims within the subset_exclude
                    subset_coverage = np.bitwise_or.reduce([sample['docs'][pid]['answers_found'] for pid in subset_exclude])
                    contained = False
                    for i in range(len(subset_coverage)):
                        if subset_coverage[i] == 1 and sample['docs'][psgs_id]['answers_found'][i] == 1:
                            contained = True
                            break
                            
                    if nli_result and (not contained): # psgs_id is not necessary
                        flag = 0
                        sent_mcite_overcite += 1 # why only overcite if not contained?
                        logger.info(f"For `{sent}`, exclude citation {psgs_id+1}")
                        candidate_downvote.append(psgs_id+1)
                    else:
                        entail_prec += 1
                        candidate_upvote.append(psgs_id+1)
                else:
                    entail_prec += 1
                    candidate_upvote.append(psgs_id+1)
        else:
            entail_prec += joint_entail # 0
            if joint_entail == 0:
                candidate_downvote.extend([i+1 for i in ref])
            if joint_entail == 1:
                candidate_upvote.extend([i+1 for i in ref])

        upvoted_cite_labels.append(candidate_upvote)
        downvoted_cite_labels.append(candidate_downvote)

    sent_total += len(sents)
    ais_scores = entail / len(sents)
    ais_scores_prec = entail_prec / total_citations if total_citations > 0 else 0 # len(sents))

    if sent_mcite > 0 and sent_mcite_support > 0:
        sent_cited = sent_mcite / sent_total
        sent_supported = sent_mcite_support / sent_mcite
        overcite = sent_mcite_overcite / sent_mcite_support
        logger.info("Among all sentences, %.2f%% have multiple citations, among which %.2f%% are supported by the joint set, among which %.2f%% overcite." % (
            100 * sent_cited, 
            100 * sent_supported, 
            100 * overcite
        ))
    else:
        sent_cited = 0
        sent_supported = 0
        overcite = 0

    output = {
        "sent_cited" : sent_cited,
        "sent_supported" : sent_supported,
        "overcite" : overcite,
        "citation_rec": ais_scores,
        "citation_prec": ais_scores_prec,
        "autoais_log": autoais_log,
    }
    return output, upvoted_cite_labels, downvoted_cite_labels

def process_samples(data, dataset_name=None):
    logger.info(f"Starting to process {dataset_name} data...")
    
    labelled_data = []
    samples_to_errors = {}
    errors_to_samples = {'rejection_precision':0, 
                         'rejection_recall':0, 
                         'coverage':0, 
                         'citation_rec':0, 
                         'citation_prec': 0, 
                         'overcite':0, 
                         'no_error':0, 
                         'total_num_samples':len(data)}

    for sample_idx, sample in tqdm(enumerate(data), desc=f"Processing {dataset_name} questions", total=len(data)):
        question = sample['question']
        docs = sample['docs']
        pattern = np.bitwise_or.reduce([(doc['answers_found']) for doc in docs])
        answer = sample['output']
        single_sample_to_errors = {}

        #added for prelim study due to different data format
        dataset_name = sample['label']

        # Remove all citations for all length calculation
        normalized_sample = copy.deepcopy(sample['output'])
        length = len(normalized_sample)

        # first check if its a rejection recall or rejection precision error
        if not_invalid_qns(pattern) and not_reject(answer):
            # calculate eval metrics
            # coverage, upvoted_content_labels, downvoted_content_labels
            if dataset_name == "asqa":
                coverage, upvoted_content_labels, downvoted_content_labels = compute_str_em(sample, pattern)
            elif dataset_name == "eli5":
                coverage, upvoted_content_labels, downvoted_content_labels = compute_claims(sample, pattern)
            elif dataset_name == "qampari":
                coverage_stats, upvoted_content_labels, downvoted_content_labels = compute_qampari_f1(sample, pattern)
                coverage = coverage_stats['qampari_f1_top5']
            else:
                raise NotImplementedError("Please put a valid dataset name")
            
            # citation quality,upvoted_cite_labels, downvoted_cite_labels
            citation, upvoted_cite_labels, downvoted_cite_labels = compute_autoais(sample, qampari=(dataset_name == "qampari"))
            citation_rec = citation['citation_rec']
            citation_prec = citation['citation_prec']

            # store data
            error_free_flag = True
            if coverage != 1:
                single_sample_to_errors['coverage'] = float(coverage)
                errors_to_samples['coverage'] += 1
                error_free_flag = False
            if citation_rec != 1:
                single_sample_to_errors['citation_rec'] = float(citation_rec)
                errors_to_samples['citation_rec'] += 1
                error_free_flag = False
            if citation_prec != 1:
                single_sample_to_errors['citation_prec'] = float(citation_prec)
                errors_to_samples['citation_prec'] += 1
                error_free_flag = False
            if error_free_flag:
                errors_to_samples['no_error'] += 1
        else:
            if is_rejection_prec_error(answer, pattern):
                single_sample_to_errors['rejection_precision'] = 0
                errors_to_samples['rejection_precision'] += 1

                sents = sentence_splitter.tokenize(answer)
                upvoted_content_labels, downvoted_content_labels = [], [[i, sent] for i,sent in enumerate(sents)]
                upvoted_cite_labels, downvoted_cite_labels = [], []
            elif is_rejection_recall_error(answer, pattern):
                single_sample_to_errors['rejection_recall'] = 0
                errors_to_samples['rejection_recall'] += 1

                sents = sentence_splitter.tokenize(answer)
                upvoted_content_labels, downvoted_content_labels = [], [[i, sent] for i,sent in enumerate(sents)]
                citation, upvoted_cite_labels, downvoted_cite_labels = compute_autoais(sample, qampari=(dataset_name == "qampari"))
            else:
                upvoted_content_labels, downvoted_content_labels = [], []
                upvoted_cite_labels, downvoted_cite_labels = [], []
                errors_to_samples['no_error'] += 1
        
        samples_to_errors[sample_idx] = single_sample_to_errors
        labelled_sample = json.loads(json.dumps(sample)) # make copy
        labelled_sample['error_type'] = single_sample_to_errors

        # add in all eval stats calculated
        labelled_sample['eval_metrics'] = {}
        labelled_sample['eval_metrics']['length'] = length
        labelled_sample['eval_metrics']['critic'] = {
            'upvoted_content_labels': upvoted_content_labels,
            'downvoted_content_labels': downvoted_content_labels,
            'upvoted_cite_labels': upvoted_cite_labels,
            'downvoted_cite_labels': downvoted_cite_labels,
        }
        if not_invalid_qns(pattern) and not_reject(answer):
            if dataset_name != "qampari":
                labelled_sample['eval_metrics']['coverage'] = coverage
            else:
                labelled_sample['eval_metrics']['coverage'] = coverage_stats
            labelled_sample['eval_metrics']['citation'] = citation
        else:
            if is_rejection_prec_error(answer, pattern):
                labelled_sample['eval_metrics']['rejection_precision'] = 0 
            elif is_rejection_recall_error(answer, pattern):
                labelled_sample['eval_metrics']['rejection_recall'] = 0 
                labelled_sample['eval_metrics']['citation'] = citation
        labelled_data.append(labelled_sample)
        
    logger.info(f'Processed {len(data)} questions')
    return samples_to_errors, errors_to_samples, labelled_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document selection")
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset (asqa, qamapri, eli5)")
    parser.add_argument("--data_file", type=str, default=None, help="Path to data file")
    parser.add_argument("--output_dir", type=str, help="Path to the output dir")
    
    args = parser.parse_args()

    logger.info(f"Arguments received: {args}")

    random.seed(0)
    np.random.seed(0)
    
    with open(args.data_file) as f:
        data = json.load(f) #['infer_data']


    samples_to_errors, errors_to_samples, labelled_data = process_samples(data, dataset_name= args.dataset_name)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    base_name = os.path.basename(args.data_file)
    
    file_name = f"{base_name}-errortypes.json"
    save_data_to_json(labelled_data, args.output_dir, file_name)

    file_name = f"{base_name}-metadata.json"
    save_data_to_json([samples_to_errors, errors_to_samples], args.output_dir, file_name)

