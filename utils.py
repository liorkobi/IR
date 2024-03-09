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
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

NUM_BUCKETS = 124


def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


def token2bucket_id(token):
    return int(_hash(token), 16) % NUM_BUCKETS


stemmer = PorterStemmer()

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


def convert_pv_scores_to_dict(pr_scores_df):
    return pd.Series(pr_scores_df.views.values, index=pr_scores_df.doc_id).to_dict()


def normalize_scores(scores_dict):
    """Normalize scores to a 0-1 range."""
    if not scores_dict:
        return {}
    min_score = min(scores_dict.values())
    max_score = max(scores_dict.values())
    if max_score == min_score:
        return {k: 1 for k in scores_dict}  # Normalize to 1 if all scores are the same
    return {k: (v - min_score) / (max_score - min_score) for k, v in scores_dict.items()}
# def expand_query_with_synonyms(query):
#     expanded_query = set()
#
#     # Tokenize the query
#     query_tokens = query.split()
#
#     # Iterate over each token in the query
#     for token in query_tokens:
#         # Get synonyms from WordNet
#         synonyms = set()
#         for syn in wordnet.synsets(token):
#             for lemma in syn.lemmas():
#                 synonyms.add(lemma.name().lower())
#
#         # Add the original token and its synonyms to the expanded query
#         expanded_query.add(token.lower())
#         expanded_query.update(synonyms)
#     return expanded_query



    # def calc_idf(self, list_of_tokens):
    #     """
    #     This function calculates the idf values according to the precomputed BM25 idf scores for each term in the query.
    #
    #     Parameters:
    #     -----------
    #     list_of_tokens: list of token representing the query. For example: ['look', 'blue', 'sky']
    #
    #     Returns:
    #     -----------
    #     idf: dictionary of idf scores. As follows:
    #                                                     key: term
    #                                                     value: bm25 idf score
    #     """
    #     idf = {}
    #     for term in list_of_tokens:
    #         if term in self.idf_scores:
    #             idf[term] = self.idf_scores[term]
    #         else:
    #             # Optionally handle terms not present in the precomputed scores,
    #             # for example, by assigning them a default score.
    #             pass
    #
    #     return idf
