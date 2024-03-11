import json
import math
import re
from collections import Counter
import pandas as pd
from google.cloud import storage
from nltk.stem import PorterStemmer

from inverted_index_gcp import InvertedIndex
from nltk.corpus import stopwords
import hashlib

stemmer = PorterStemmer()



english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
all_stopwords = set(all_stopwords)

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

NUM_BUCKETS = 124


def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


def token2bucket_id(token):
    return int(_hash(token), 16) % NUM_BUCKETS


colnames_pr = ['doc_id', 'pr']  # pagerank
colnames_pv = ['doc_id', 'views']  # pageviews
colnames_id_len = ['doc_id', 'len']  # dict_id_len
import re


def preprocess_query(query):
    # Extract phrases in quotation marks
    pattern = r'"([^"]+)"'
    phrases_in_quotes = re.findall(pattern, query)
    query_without_quotes = re.sub(pattern, '', query)

    # Tokenize and stem the rest of the query
    stemmed_tokens = []
    unstemed_tokens = []
    for token in re.finditer(r'\w+', query_without_quotes.lower()):
        if token.group() not in all_stopwords:
            stemmed_tokens.append(stemmer.stem(token.group()))
            unstemed_tokens.append(token.group())

    # Handle phrases within quotes: add them as single tokens
    for phrase in phrases_in_quotes:
        # Add the full phrase as a single token for stemmed version
        stemmed_phrase = ' '.join([stemmer.stem(word) for word in phrase.split() if word not in all_stopwords])
        stemmed_tokens.append(stemmed_phrase)
        # Add the full phrase as a single token for unstemmed version
        unstemed_tokens.append(phrase)
        print("stemmed_tokens",stemmed_tokens)
        print("unstemed_tokens",unstemed_tokens)

    return stemmed_tokens, unstemed_tokens


def classify_query(query):
    # Keywords for factual questions
    if re.search(r'who|what|when|where|why|how|which', query, re.IGNORECASE):
        return 'factual'
    # Keywords or patterns for detailed explanations
    elif re.search(r'describe|explain|process|structure', query, re.IGNORECASE):
        return 'detailed_explanation'
    # If none match, consider it a broad topic
    else:
        return 'broad_topic'

def open_json(gcs_path):
    client = storage.Client()
    bucket_name, blob_name = gcs_path.split("//")[1].split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    content = blob.download_as_text()
    result_dict = json.loads(content)
    return result_dict
def convert_pr_scores_to_dict(pr_scores_df):
    return pd.Series(pr_scores_df.pr.values, index=pr_scores_df.doc_id).to_dict()


def normalize_scores(scores_dict):
    """Normalize scores to a 0-1 range."""
    if not scores_dict:
        return {}
    min_score = min(scores_dict.values())
    max_score = max(scores_dict.values())
    if max_score == min_score:
        return {k: 1 for k in scores_dict}  # Normalize to 1 if all scores are the same
    return {k: (v - min_score) / (max_score - min_score) for k, v in scores_dict.items()}


def calculate_tf_idf(self, query):
    tf_idf_scores = {}
    # Tokenize and filter out stopwords
    query_tokens_stem = [stemmer.stem(token.group()) for token in re.finditer(r'\w+', query.lower()) if
                         token.group() not in all_stopwords]
    # Calculate TF and aggregate TF-IDF for each document
    for term in query_tokens_stem:
        postings = self.index_text.get_posting_list(term, "text", "with_stemming")
        for doc_id, freq in postings:
            tf = freq
            tf_idf = tf * self.idf_scores[term]
            tf_idf_scores[doc_id] = tf_idf_scores.get(doc_id, 0) + tf_idf

    # Combine TF-IDF scores with PageRank and fetch Wikipedia titles
    res = []
    for doc_id, tf_idf_score in sorted(tf_idf_scores.items(), key=lambda item: item[1], reverse=True)[:100]:
        page_rank_score = self.pr_scores_dict.get(doc_id, 0)
        combined_score = tf_idf_score + page_rank_score
        res.append((combined_score, str(doc_id), self.title_dict.get(doc_id)))

    # Sort by the combined score
    res_sorted = sorted(res, key=lambda x: x[0], reverse=True)

    # Extract only the title and document ID from the sorted results
    res_final = [(doc_id, doc_title) for _, doc_id, doc_title in res_sorted]

    return res_final


def optimize_weights(query, query_tokens):
    query_type = classify_query(query)
    text_weight, title_weight = 0.6, 0.4

    if query_type == 'factual' or len(query_tokens) <= 3:
        text_weight, title_weight = 0.1, 0.9
    elif query_type == 'detailed_explanation':
        text_weight, title_weight = 0.7, 0.3

    return text_weight, title_weight

import pickle


def get_pv():
    client = storage.Client()
    bucket_name = 'ir-dict'
    blob_name = 'pageviews-202108-user.pkl'
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    with blob.open("rb") as f:
        data = pickle.load(f)
    return data
