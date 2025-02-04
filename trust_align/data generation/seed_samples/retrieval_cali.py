import argparse
import csv
import json
import os
import pickle
import re
import string
import threading
import time
from datetime import timedelta

import colorlog
import numpy as np
import torch
import torch.distributed
from accelerate import PartialState
from accelerate.utils import broadcast_object_list, gather_object, broadcast
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

distributed_state = PartialState(timeout=timedelta(minutes=30))

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


def bm25_sphere_retrieval(data, args):
    from pyserini.search import LuceneSearcher
    index_path = os.environ.get("BM25_SPHERE_PATH")
    print("loading bm25 index, this may take a while...")
    searcher = LuceneSearcher(index_path)

    print("running bm25 retrieval...")
    for d in tqdm(data):
        query = d["question"]
        try:
            hits = searcher.search(query, args.top_k)
        except Exception as e:
            #https://github.com/castorini/pyserini/blob/1bc0bc11da919c20b4738fccc020eee1704369eb/scripts/kilt/anserini_retriever.py#L100
            if "maxClauseCount" in str(e):
                query = " ".join(query.split())[:950]
                hits = searcher.search(query, args.top_k)
            else:
                raise e

        docs = []
        for hit in hits:
            score = hit.score
            h = json.loads(str(hit.docid).strip())
            docs.append({
                "title": h["title"],
                "text": hit.lucene_document.get('raw'),
                "url": h["url"],
                "score": round(score, 4)
            })
        d["docs"] = docs



def gtr_wiki_retrieval(data, args):
    device = f"cuda:{distributed_state.process_index}" if torch.cuda.is_available() else "cpu"
    print("loading GTR encoder...")
    encoder = SentenceTransformer("sentence-transformers/gtr-t5-xxl", device = device)

    questions = [d["question"] for d in data]
    with torch.inference_mode():
        queries = encoder.encode(questions, batch_size=120, show_progress_bar=True, normalize_embeddings=True)
        queries = torch.tensor(queries, dtype=torch.float16, device="cpu")

    # the wikipedia split from DPR repo: https://github.com/facebookresearch/DPR
    DPR_WIKI_TSV = os.environ.get("DPR_WIKI_TSV")
    docs = []
    print("loading wikipedia file...")
    with open(DPR_WIKI_TSV) as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(tqdm(reader, total=21015325)):
            if i == 0:
                continue
            docs.append(row[2] + "\n" + row[1])

    GTR_EMB = os.environ.get("GTR_EMB")
    if not os.path.exists(GTR_EMB):
        print("gtr embeddings not found, building...")
        embs = gtr_build_index(encoder, docs)
    else:
        print("gtr embeddings found, loading...")
        with open(GTR_EMB, "rb") as f:
            embs = pickle.load(f)

    del(encoder) # save gpu mem

    gtr_emb = torch.tensor(embs, dtype=torch.float16, device=device)

    print("running GTR retrieval...")
    for qi, q in enumerate(tqdm(queries)):
        q = q.to(device)
        scores = torch.matmul(gtr_emb, q)
        score, idx = torch.topk(scores, args.top_k)
        ret = []
        for i in range(idx.size(0)):
            title, text = docs[idx[i].item()].split("\n")
            ret.append({"id": str(idx[i].item()+1),"title": title, "text": text, "score": score[i].item()})
        data[qi]["docs"] = ret
        q = q.to("cpu")
        

def gtr_build_index(encoder, docs):
    with torch.inference_mode():
        embs = encoder.encode(docs, batch_size=4, show_progress_bar=True, normalize_embeddings=True)
        embs = embs.astype("float16")

    GTR_EMB = os.environ.get("GTR_EMB")
    with open(GTR_EMB, "wb") as f:
        pickle.dump(embs, f)
    return embs


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def format_document(doc):
    """Format document for AutoAIS."""

    if "sent" in doc:
        # QA-extracted docs
        return "Title: %s\n%s" % (doc['title'], doc['sent'])
    else:
        return "Title: %s\n%s" % (doc['title'], doc['text'])


def compute_str_em(short_answers, doc, is_synonym):

    # Convert to list
    if not isinstance(short_answers, list):   
        short_answers = [short_answers]
        
    n_short_answers = [normalize_answer(sa) for sa in short_answers]
    n_doc = normalize_answer(doc)
    
    flag = [n_sa in n_doc for n_sa in n_short_answers]
    
    # Verify if any of the answers is present in the given context.
    if is_synonym:
        return int(any(flag))
        
    return int(all(flag))  #hit


def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-6}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory


def compute_entailment(claims, docs):
    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        # logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map=f"cuda:{distributed_state.process_index}")
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)
        
    # Convert to list
    if not isinstance(claims, list):   
        claims = [claims]
    if not isinstance(docs, list):   
        docs = [docs]
        
    # run claim eval
    input_texts = ["premise: {} hypothesis: {}".format(doc, claim) for doc, claim in zip(docs, claims)]
    input_ids = autoais_tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=autoais_tokenizer.model_max_length).input_ids
    batch_size = 25 # 5
    with torch.inference_mode():
        outputs = []
        for start_index in tqdm(range(0, len(input_ids), batch_size), desc="Compute entailment", disable=True):
            input_batch = input_ids[start_index:start_index + batch_size].to(autoais_model.device)
            output_batch = autoais_model.generate(input_batch, max_new_tokens=10)
            outputs.extend(autoais_tokenizer.batch_decode(output_batch, skip_special_tokens=True))

    torch.cuda.empty_cache()
    return [1 if result == "1" else 0 for result in outputs]


def calculate_recall_score(ans_set, top_k=None, calib=False):
    # recall score of the given set
    if not all(isinstance(i, list) for i in ans_set):   
        ans_set = [ans_set]
        
    union_set = np.bitwise_or.reduce(ans_set).tolist()

    if calib:
        return 100 if any(union_set) else 0

    else:
        if top_k:
            return 100 * min(top_k, sum(union_set)) / min(top_k, len(union_set))
        
        return 100 * sum(union_set) / len(union_set)


def nli_verify(short_answers, text, answers_found):
    if any(answers_found):
        selected_answers = [short_answers[i].strip() for i in range(len(answers_found)) if answers_found[i] == 1]
        nli_result = compute_entailment(selected_answers, len(selected_answers) * [text])
        nli_index = 0
        new_answers_found = answers_found.copy()
        for i in range(len(new_answers_found)):
            if new_answers_found[i] == 1:
                new_answers_found[i] = nli_result[nli_index]
                nli_index += 1
        return new_answers_found
    else:
        return answers_found


def found_answers(item, args):
    if args.dataset_name == "asqa":
        for doc in item['docs']:
            doc['answers_found'] = []
            for qa_pair in item['qa_pairs']:
                doc['answers_found'].append(compute_str_em(qa_pair['short_answers'], doc['text'], is_synonym=True))
            
            # NLI verification
            qa_pair_ans_found = nli_verify([qa_pair['question'] + " " + qa_pair['short_answers'][0].strip() for qa_pair in item['qa_pairs']], format_document(doc), doc['answers_found'])
            overall_ans_found = nli_verify([item['question'] + " " + qa_pair['short_answers'][0].strip() for qa_pair in item['qa_pairs']], format_document(doc), doc['answers_found'])
            doc['answers_found'] = np.bitwise_or.reduce([qa_pair_ans_found, overall_ans_found]).tolist()
            
            # calculate the recall score for each passage
            doc['rec_score'] = calculate_recall_score([doc['answers_found']])
            
    elif args.dataset_name == "qampari":
        for doc in item['docs']:
            doc['answers_found'] = []
            for ans in item['answers']:
                doc['answers_found'].append(compute_str_em(ans, doc['text'], is_synonym=True))
                
            # NLI verification
            doc['answers_found'] = nli_verify([item['question'] + " " + answer[0].strip() for answer in item['answers']], format_document(doc), doc['answers_found'])
            
            # calculate the recall-5 score for each passage
            doc['rec_score'] = calculate_recall_score([doc['answers_found']], top_k=5)
        
    elif args.dataset_name == "eli5":
        length = len(item["claims"])
        docs = []
        claims = []
        for doc in item['docs']:
            for claim in item["claims"]:
                docs.append(format_document(doc))
                claims.append(claim)

        results = compute_entailment(claims, docs)
                 
        for i, doc in enumerate(item['docs']):
            doc['answers_found'] = results[i*length:i*length+length]
            doc['rec_score'] = calculate_recall_score([doc['answers_found']])
                    
    elif args.dataset_name == "expertqa":
        length = len(item["claims"])
        docs = []
        claims = []
        for doc in item['docs']:
            for claim in item["claims"]:
                docs.append(format_document(doc))
                claims.append(claim)

        results = compute_entailment(claims, docs)
                 
        for i, doc in enumerate(item['docs']):
            doc['answers_found'] = results[i*length:i*length+length]
            doc['rec_score'] = calculate_recall_score([doc['answers_found']])
    
    else:
        raise ValueError("No corresponding dataset format")
    

def create_oracle_set(data, args, top=5):
    for item in tqdm(data):
    # Retrieval recall@k on ASQA(EM recall), QAMPARI(recall-5), and ELI5(claim recall).
        
        # take the top 5 passages as initial oracle set.
        item['docs'].sort(key=lambda x: x['rec_score'], reverse=True)
        oracle_set = item['docs'][:top]
        remaining_set = item['docs'][top:]
        
        # calculate the largest recall improvement of the oracle set
        for i in range(len(oracle_set)):
            best_replacement_idx = None
            best_improvement_score = 0
            
            for j, passage in enumerate(remaining_set):
                current_recall = calculate_recall_score([doc['answers_found'] for doc in oracle_set], top_k = 5 if args.dataset_name == "qampari" else None)
                
                # Calculate the recall score of the potential new oracle set with replacement
                new_oracle_set = oracle_set[:i] + [passage] + oracle_set[i+1:]
                new_recall = calculate_recall_score([doc['answers_found'] for doc in new_oracle_set], top_k = 5 if args.dataset_name == "qampari" else None)
                improvement_score = new_recall - current_recall
                
                if improvement_score > best_improvement_score:
                    best_replacement_idx = j
                    best_improvement_score = improvement_score
            
            # Perform the replacement if it improves recall
            if best_replacement_idx is not None and best_improvement_score > 0:
                oracle_set[i] = remaining_set.pop(best_replacement_idx)
        
        # save oracle set per sample
        item['docs'] = oracle_set
    
    # calculate final mean recall
    rec = []
    # show answerable num
    answerable_num = 0
    for item in data:
        rec.append(calculate_recall_score([doc['answers_found'] for doc in item['docs']], top_k = 5 if args.dataset_name == "qampari" else None))
        answerable_num += any(np.bitwise_or.reduce([doc["answers_found"] for doc in item['docs'][:5]]))
    
    logger.info(f"Oracle averaged recall score: {np.mean(rec)}")
    logger.info(f"Oracle answerable num: {answerable_num}")


def send_heartbeat(rank, world_size, stop_event):
    flag = torch.tensor(0, device=distributed_state.process_index)
    while not stop_event.is_set():
        broadcast(flag, 0)
        time.sleep(10)  # Send heartbeat every second
    flag = torch.tensor(1, device=distributed_state.process_index)
    broadcast(flag, 0)
    time.sleep(1)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Passage retrieval.")
    parser.add_argument("--retriever", type=str, default=None, help="options: bm25/gtr")
    parser.add_argument("--data_file", type=str, default=None, help="path to the data file")
    parser.add_argument("--output_file", type=str, default=None, help="same format as the data file but with the retrieved docs.")
    parser.add_argument("--top_k", type=int, default=100, help="retrieve top-k docs per question")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset (for recall score)")
    parser.add_argument("--oracle", action="store_true", help="get oracle set")
    parser.add_argument("--answer_found", action="store_true", help="get answer_found for each doc")

    args = parser.parse_args()

    with open(args.data_file) as f:
        data = json.load(f)
    
    if args.retriever and args.answer_found:
        if distributed_state.is_main_process:
            stop_event = threading.Event()
            heartbeat_thread = threading.Thread(target=send_heartbeat, args=(distributed_state.process_index, distributed_state.num_processes, stop_event))
            heartbeat_thread.start()
            
            if args.retriever == "bm25":
                # Using Sphere as corpus
                bm25_sphere_retrieval(data, args)
                pass
            elif args.retriever == "gtr":
                gtr_wiki_retrieval(data, args)
            else:
                raise NotImplementedError
            
            # save retreived docs only
            with open(args.output_file, "w") as f:
                json.dump(data, f, indent=4)
            
            stop_event.set()  # Stop the heartbeat thread
            heartbeat_thread.join()
            broadcast_object_list(data, 0)
        else:
            # wait for main process to get data
            flag = torch.tensor(0, device=distributed_state.process_index)
            while True:
                broadcast(flag, 0)
                if flag:
                    broadcast_object_list(data, 0)
                    break
                time.sleep(10)
        
    if args.answer_found:
        # find answer list
        tmp_splits = []
        with distributed_state.split_between_processes(data) as inputs:
            print(f"{distributed_state.process_index}: {len(inputs)}")
            for item in tqdm(inputs):
                found_answers(item, args)
            tmp_splits.extend(inputs)
        # wait for other process to process data in case of timeout
        data = gather_object(tmp_splits)  

        if distributed_state.is_main_process:
            # save retreived docs and answer_found
            with open(args.output_file, "w") as f:
                json.dump(data, f, indent=4)
            
            # calculate final mean recall
            rec = []
            rec_5 = []
            calib_rec = []
            calib_rec_5 = []
            # show answerable num
            answerable_num = 0
            for item in data:
                rec.append(calculate_recall_score([doc['answers_found'] for doc in item['docs']], top_k = 5 if args.dataset_name == "qampari" else None))
                rec_5.append(calculate_recall_score([doc['answers_found'] for doc in item['docs'][:5]], top_k = 5 if args.dataset_name == "qampari" else None))
                calib_rec.append(calculate_recall_score([doc['answers_found'] for doc in item['docs']], top_k = 5 if args.dataset_name == "qampari" else None, calib=True))
                calib_rec_5.append(calculate_recall_score([doc['answers_found'] for doc in item['docs'][:5]], top_k = 5 if args.dataset_name == "qampari" else None, calib=True))
                answerable_num += any(np.bitwise_or.reduce([doc["answers_found"] for doc in item['docs'][:5]]))

            logger.info(f"Top-{args.top_k} averaged recall score: {np.mean(rec)}")
            logger.info(f"Top-5 averaged recall score: {np.mean(rec_5)}")
            logger.info(f"Top-5 answerable num: {answerable_num}")
            logger.info(f"Top-{args.top_k} averaged calibrated recall score: {np.mean(calib_rec)}")
            logger.info(f"Top-5 averaged calibrated recall score: {np.mean(calib_rec_5)}")

    if args.oracle:
        if distributed_state.is_main_process:
            # show statistics
            rec = []
            rec_5 = []
            calib_rec = []
            calib_rec_5 = []
            # show answerable num
            answerable_num = 0
            for item in data:
                rec.append(calculate_recall_score([doc['answers_found'] for doc in item['docs']], top_k = 5 if args.dataset_name == "qampari" else None))
                rec_5.append(calculate_recall_score([doc['answers_found'] for doc in item['docs'][:5]], top_k = 5 if args.dataset_name == "qampari" else None))
                calib_rec.append(calculate_recall_score([doc['answers_found'] for doc in item['docs']], top_k = 5 if args.dataset_name == "qampari" else None, calib=True))
                calib_rec_5.append(calculate_recall_score([doc['answers_found'] for doc in item['docs'][:5]], top_k = 5 if args.dataset_name == "qampari" else None, calib=True))
                answerable_num += any(np.bitwise_or.reduce([doc["answers_found"] for doc in item['docs'][:5]]))

            logger.info(f"Top-{args.top_k} averaged recall score: {np.mean(rec)}")
            logger.info(f"Top-5 averaged recall score: {np.mean(rec_5)}")
            logger.info(f"Top-5 answerable num: {answerable_num}")
            logger.info(f"Top-{args.top_k} averaged calibrated recall score: {np.mean(calib_rec)}")
            logger.info(f"Top-5 averaged calibrated recall score: {np.mean(calib_rec_5)}")

            create_oracle_set(data, args, top=5)
            # save oracle docs
            with open(args.output_file.split(".json")[0] + "_oracle.json", "w") as f:
                json.dump(data, f, indent=4)
