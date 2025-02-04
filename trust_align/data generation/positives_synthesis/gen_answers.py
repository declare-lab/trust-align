import argparse
import os
import re
import json
import time
import string
import threading
import colorlog
import numpy as np
import concurrent.futures
from tqdm.auto import tqdm
from textwrap import dedent
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

stop_event = threading.Event()

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

load_dotenv()
model = AzureChatOpenAI(
    deployment_name = os.getenv("DEPLOY_NAME"),
    openai_api_version = "2024-02-01",
    max_tokens=200,
    temperature=0.3
)


def save_as_jsonl(records, filename):
    records.sort(key=lambda x: x[0])
    with open(filename, "w") as f:
        for r in records:
            line = {
                "index": r[0],
                "new_answer": r[1],
            }
            f.write(json.dumps(line) + "\n")
            
            
def count_non_empty_sublists(double_list):
    count = 0
    for sublist in double_list:
        if sublist:
            count += 1
    return count
 
 
def format_document(doc):
    """Format document for AutoAIS."""

    if "sent" in doc:
        # QA-extracted docs
        return "Title: %s\n%s" % (doc['title'], doc['sent'])
    else:
        return "Title: %s\n%s" % (doc['title'], doc['text'])
    

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
    

def eli5_task(i, template, q, c, bar):
    while stop_event.is_set():
        time.sleep(1)

    assert len(c) > 0, f"No claims provided for Q: {q}"
    claims = "\n".join([f"Claim {idx+1}: {claim}" for idx, claim in enumerate(c)])
    prompt = template.format_prompt(question = q, claims = claims)
    # logger.critical(prompt)
    while True:
        try:
            output = model.invoke(prompt)
            # process output
            logger.warning(f"Q: {q}")
            resp = output.content.split("Generated Response:")[-1].strip()
            logger.warning(f"A: {resp}")
            bar.update(1)
            return i, resp
        except Exception as e:
            if "rate limit" in repr(e):
                wait_time =  re.search(r'retry after (\d+) second', repr(e)).group(1)
                logger.critical("Sleeping for {} seconds.".format(wait_time))
                time.sleep(int(wait_time))
                continue
            logger.critical(f"Cannot get answer for: '{q}', due to {e}")
            bar.update(1)
            return i, ""  
    

def eli5_new_answers(data, args):
    system_template = dedent("""
    Given a problem and some claims as answer tags, please generate a high-quality response. The response needs to follow the following requirements:

    1. Use only all of the claims: Ensure that the response contains and only contains information from the given claims, without introducing any new information. Guarantee covering all claims in the reponse.
    2. Each sentence must contain valuable information: Every sentence must either directly originate from the claims or infer from the claims, avoiding any irrelevant and unuseful information included in the response. You can use each claim only for one time.
    3. Condense and combine: If there are similarities between claims, merge them into a comprehensive statement to make the response more concise. For example, if two claims both mention similar aspect of health benefits, they can be merged into one sentence.
    4. Fluent and natural: Ensure that the sentences in the response are coherent and natural, using connecting words and maintaining logical order between sentences.
    5. Answer tags in response: Indicate each claim immediately after the corresponding content in the response with the format [Claim X], where X is the index of the claim in the provided list starting from 1. For example, [Claim 1].
                             
    Example:

    Question: What are the impacts of regular reading on cognitive function?
    Claim 1: Regular reading enhances vocabulary and language skills.
    Claim 2: It stimulates mental processes and improves focus.
    Claim 3: Reading can slow down cognitive decline related to aging.
    Generated Response: Regular reading positively impacts cognitive function by enriching vocabulary and language abilities [Claim 1], sharpening mental processes [Claim 2], and boosting concentration [Claim 2]. Additionally, cognitive decline associated with aging is also slowed [Claim 3].
    """).strip()

    input_template = dedent("""
    Question: {question}
    {claims}
    Generated Response:
    """).strip()

    template = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", input_template)
    ])
    
    
    questions = []
    claims = []
    for item in data:
        # only use topk docs for answer generating
        item['docs'] = item['docs'][:args.topk_doc]
        questions.append(item['question'])
        union_ans_set = np.bitwise_or.reduce([doc['answers_found'] for doc in item['docs']]).tolist()
        selected_answers = [item['claims'][i].strip() for i in range(len(union_ans_set)) if union_ans_set[i] == 1]
        # can be null
        claims.append(selected_answers)
    
    return template, questions, claims


def asqa_task(i, template, q, p, c, bar):
    while stop_event.is_set():
        time.sleep(1)

    assert len(c) > 0, f"No claims provided for Q: {q}"
    answers = ", ".join([f'Answer Label {idx+1}: "{ans}"' for idx, ans in enumerate(c)])
    passages = "\n".join(p)
    prompt = template.format_prompt(question = q, passage = passages, answers = answers)
    # logger.critical(prompt)
    while True:
        try:
            output = model.invoke(prompt)
            # process output
            logger.warning(f"Q: {q}")
            resp = output.content.split("Output:")[-1].strip()
            logger.warning(f"A: {resp}")
            bar.update(1)
            return i, resp
        except Exception as e:
            if "rate limit" in repr(e):
                wait_time =  re.search(r'retry after (\d+) second', repr(e)).group(1)
                logger.critical("Sleeping for {} seconds.".format(wait_time))
                time.sleep(int(wait_time))
                continue
            logger.critical(f"Cannot get answer for: '{q}', due to {e}")
            bar.update(1)
            return i, ""  


def asqa_new_answers(data, args):
    system_template = dedent("""
    Please provide a high-quality answer to the given question using the provided document. The answer must include all the answer labels, and each answer label used should be marked with its index immediately after it in the format [Answer Label X], where X is the index of the answer label in the provided list starting from 1. For example, [Answer Label 1].  Ensure the answer is coherent and natural, and does not exceed four sentences. You cannot make up any factual information based on your imagination: The additional information added from the given document should be relevant to the question and grounded by the document, but must not contain any factual information that cannot be inferred from the given answer labels. (e.g., if the answer label does not mention a specific year, you cannot introduce a specific year in the final answer).
    
    Example:

    Question: Where are the Chargers playing their home games?
    
    Document: The Los Angeles Chargers were founded as a Los Angeles-based team on August 14, 1959, and began play on September 10, 1960. They spent their first season in Los Angeles, playing at the Los Angeles Memorial Coliseum, before relocating to San Diego in 1961, where they played at Balboa Stadium until 1966 and SDCCU Stadium from 1967-2016. The Chargers returned to Los Angeles prior to the 2017 season and played at Dignity Health Sports Park from 2017-2019, while SoFi Stadium was under construction. The Chargers currently play their home games at SoFi Stadium in Inglewood, California, which the club shares with the Los Angeles Rams.
    
    Answer Label 1: "Los Angeles Memorial Coliseum"
    Answer Label 2: "Balboa Stadium"
    Answer Label 3: "SDCCU Stadium"
    Answer Label 4: "Dignity Health Sports Park"
    
    Output: The Los Angeles Chargers originally played at the Los Angeles Memorial Coliseum [Answer Label 1], then moved to Balboa Stadium [Answer Label 2] in San Diego. They later played at SDCCU Stadium [Answer Label 3] before returning to Los Angeles to play at Dignity Health Sports Park [Answer Label 4].
    """).strip()

    input_template = dedent("""
    Question: {question}
    
    Document: {passage}
    
    {answers}
    
    Output:
    """).strip()
    
    template = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", input_template)
    ])
    
    questions = []
    claims = []
    passages = []
    for item in data:
        # only use topk docs for answer generating
        item['docs'] = item['docs'][:args.topk_doc]
        questions.append(item['question'])
        facts = item['docs']
        union_ans_set = np.bitwise_or.reduce([doc['answers_found'] for doc in item['docs']]).tolist()
        short_answers = [qa_pair['short_answers'][0].strip() for qa_pair in item['qa_pairs']]
        # remove duplicate & unsupport short answers
        seen = set()
        selected_answers = []
        selected_facts = []
        for i in range(len(union_ans_set)):
            if union_ans_set[i] == 1:
                norm_st_ans = normalize_answer(short_answers[i])
                if norm_st_ans not in seen:
                    seen.add(norm_st_ans)
                    selected_answers.append(short_answers[i].strip())
                    for fact in facts:
                        if fact['answers_found'][i] == 1:
                            selected_facts.append(format_document(fact))
        # can be null
        passages.append(selected_facts)
        claims.append(selected_answers)
        
    return template, questions, passages, claims
        
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generat golden answers")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset (for recall score)")
    parser.add_argument("--data_file", type=str, default=None, help="path to the data file")
    parser.add_argument("--output_file", type=str, default=None, help="same format as the data file but with the retrieved docs.")
    parser.add_argument("--topk_doc", type=int, default=5, help="use topk docs for answer generating")
    
    args = parser.parse_args()
    
    with open(args.data_file) as f:
        data = json.load(f)
        
    if args.dataset_name == "asqa":
        # Using Sphere as corpus
        template, questions, passages, claims = asqa_new_answers(data, args)
    elif args.dataset_name == "eli5":
        template, questions, claims = eli5_new_answers(data, args)
    else:
        raise NotImplementedError
    
    new_answers = []
    bar = tqdm(total=count_non_empty_sublists(claims))
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        if args.dataset_name == "asqa":
            futures = [executor.submit(asqa_task, i, template, q, p, c, bar) for i, (q, p, c) in enumerate(zip(questions, passages, claims)) if len(c) > 0]
        elif args.dataset_name == "eli5":
            futures = [executor.submit(eli5_task, i, template, q, c, bar) for i, (q, c) in enumerate(zip(questions, claims)) if len(c) > 0]
        else:
            raise NotImplementedError
        
        start_time = time.time()
        
        for future in concurrent.futures.as_completed(futures):
            i, output = future.result()
            new_answers.append([i, output])

            if len(new_answers) % 100 == 0:
                save_as_jsonl(new_answers, args.output_file)
                
            if len(new_answers) % 50 == 0 and (time.time() - start_time) < 60:
                stop_event.set()
                logger.critical("Sleeping for {} seconds.".format(round(60-(time.time() - start_time), 2)))
                time.sleep(60-(time.time() - start_time))
                start_time = time.time()
                stop_event.clear()

    save_as_jsonl(new_answers, args.output_file)