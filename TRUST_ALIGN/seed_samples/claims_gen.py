import re
import os
import json
import time
import threading
import colorlog
import concurrent.futures
from tqdm.auto import tqdm
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
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
    deployment_name = os.getenv("DEPLOY_NAME"),   # gpt-35-turbo
    openai_api_version = "2024-02-01",
    max_tokens=500,
    temperature=0.3
)


input_template = """Read the original question and passage, and generate 3 additional claims that are supported by the passage and answer the question.

Original question: What's the difference between Shia vs. Sunni Islam?
Passage: The main difference between Shia and Sunni Muslim is related to ideological heritage and
issues of leadership. This difference is first formed after the death of the Prophet Muhammad in 632
A.D. The ideological practice of the Sunni branch strictly follows Prophet Muhammad and his
teachings, while the Shia branch follows Prophet Muhammad's son-in-law Ali. Nowadays, Sunni and
Shia are the major branches of Islam.
Claim 1: The major branches of Islam are Sunni and Shia.
Claim 2: Prophet Muhammad died in 632 A.D.
Claim 3: The ideological practice of the Sunni branch strictly follows Prophet Muhammad and his
teachings.

Original question: What causes Bi-polar disorder?
Passage: Bipolar disorder is an emotional disorder that causes extreme mood swings between
excitement and depression. The spectrum of mood swing may span from days to months. We are still not
certain of the exact factors that cause such disorder, but genetics is considered a major factor.
Claim 1: One symptom of Bi-polar disorder is extreme mood swings between excitement and depression.
Claim 2: Genetics could be one of the major factors that causes Bi-polar disorder.
Claim 3: The mood swing from Bi-polar disorder can last days to months.

Original question: How do we hear differences in sound besides volume and pitch?
Passage: Pitch refers to the frequency of soundwave, and volumn refers to the amplitude of the
soundwave. Besides volumn and pitch, we can also tell the difference between sounds based on the
tone of sound. For example, we can differentiate the sound of different instruments based on the
tone of the sounds.
Claim 1: Volume of sound is the amplitude of the soundwave.
Claim 2: Pitch is the frequency of soundwave.
Claim 3: We can use the tone of the sounds to differentiate the sound of different instruments.

Original question: {question}
Passage: {passage}
"""

template = PromptTemplate.from_template(input_template)


def task(i, template, q, p, bar):
    while stop_event.is_set():
        time.sleep(1)
    prompt = template.format_prompt(question = q, passage = p)
    while True:
        try:
            output = model.invoke(prompt)
            # logger.warning(output)
            claims = re.findall(r'Claim \d+: (.*?)(?=\n|$)', output.content)
            claims = [claim.strip() for claim in claims]
            bar.update(1)
            logger.critical(f"Q: {q}")
            logger.critical(claims)
            return i, claims 
        except Exception as e:
            logger.error(repr(e))
            if "rate limit" in repr(e):
                wait_time =  re.search(r'retry after (\d+) second', repr(e)).group(1)
                logger.warning("Sleeping for {} seconds.".format(wait_time))
                time.sleep(int(wait_time))
                continue
            bar.update(1)
            return i, []  


def save_as_jsonl(records, filename):
    records.sort(key=lambda x: x[0])
    with open(filename, "w") as f:
        for r in records:
            line = {
                "index": r[0],
                "claims": r[1],
            }
            f.write(json.dumps(line) + "\n")


def generate_claims(questions, passages):
    claims = []
    bar = tqdm(total=len(questions))
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(task, i, template, q, p, bar) for i, (q, p) in enumerate(zip(questions, passages))]
        start_time = time.time()
        
        for furture in concurrent.futures.as_completed(futures):
            i, output = furture.result()
            claims.append([i, output])

            # if len(claims) % 500 == 0:
            #     save_as_jsonl(claims, f"claims_num_{len(questions)}.jsonl")
                
            if len(claims) % 500 == 0 and (time.time() - start_time) < 60:
                stop_event.set()
                logger.warning("Sleeping for {} seconds.".format(round(60-(time.time() - start_time), 2)))
                time.sleep(60-(time.time() - start_time))
                start_time = time.time()
                stop_event.clear()
    return claims

if __name__ == "__main__":
    with open("eli5_run.json", "r") as f:
        new_data = json.load(f)
    questions = []
    passages = []
    for item in new_data:
        questions.append(item['question'])
        passages.append(item['answer'])
        
    claims = generate_claims(questions, passages)
    save_as_jsonl(claims, "claims.jsonl")