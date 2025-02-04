import json
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.functional import normalize

from .utils import format_document


def doc_to_text_tfidf(doc: Dict[str, Any]) -> str:
    return doc['title'] + ' ' + doc['text']

def doc_to_text_dense(doc: Dict[str, Any]) -> str:
    return doc['title'] + '. ' + doc['text']


class SearcherWithinDocs:

    def __init__(self, docs: List[Dict[str, Any]], retriever: str, model: Any=None, tokenizer: Any=None, device: str="cuda") -> None:
        self.retriever = retriever
        self.docs = docs
        self.device = device
        if retriever == "tfidf":
            self.tfidf = TfidfVectorizer()
            self.tfidf_docs = self.tfidf.fit_transform([doc_to_text_tfidf(doc) for doc in docs])
        elif "gtr" in retriever: 
            self.model = model
            self.embeddings = self.model.encode([doc_to_text_dense(doc) for doc in docs], device=self.device, convert_to_numpy=False, convert_to_tensor=True, normalize_embeddings=True)
        elif "nli" in retriever:
            self.model = model
            self.tokenizer = tokenizer
        else:
            raise NotImplementedError

    def search(self, query:str) -> int:
        # Return the top-1 result doc id

        if self.retriever == "tfidf":
            tfidf_query = self.tfidf.transform([query])[0]
            similarities = [cosine_similarity(tfidf_doc, tfidf_query) for tfidf_doc in self.tfidf_docs]
            best_doc_id = np.argmax(similarities)
        elif "gtr" in self.retriever:
            q_embed = self.model.encode([query], device=self.device, convert_to_numpy=False, convert_to_tensor=True, normalize_embeddings=True)
            score = torch.matmul(self.embeddings, q_embed.t()).squeeze(1).detach().cpu().numpy()
            best_doc_id = np.argmax(score)
        elif "nli" in self.retriever:
            claim = query
            score = np.array([self.get_entailment_score(format_document(doc), claim) for doc in self.docs])
            print(f'{score=}')
            best_doc_id = np.argmax(score)
            # print(f'{best_doc_id=}')
        else:
            raise NotImplementedError
        return int(best_doc_id)

        
    def get_entailment_score(self, passage: str, claim: str) -> float:
        input_text = "premise: {} hypothesis: {}".format(passage, claim)
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(input_ids, max_new_tokens=10, output_scores=True, return_dict_in_generate=True)
            generated_ids = outputs.sequences
            scores = outputs.scores  # list of logits for each generated token

        log_probs = []
        for i, score in enumerate(scores):
            log_prob = torch.nn.functional.log_softmax(score, dim=-1)
            token_id = generated_ids[0, i + 1]  # +1 because the first token is the input token
            log_probs.append(log_prob[0, token_id].item())

        entailment_score = sum(log_probs)
        return entailment_score