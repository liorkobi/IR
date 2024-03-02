import math
import re

import gcsfs
import requests

import pandas as pd
import nltk
# nltk.download('stopwords')

from inverted_index_gcp import InvertedIndex
from nltk.corpus import stopwords

import hashlib

from google.cloud import storage
import csv
import gzip
import os
import shutil

def read_pr(base_dir, name, bucket_name):
    page_rank_scores = {}
    # Create a GCS client
    client = storage.Client()
    # Get the bucket object
    bucket = client.bucket(bucket_name)
    # Construct the blob path
    blob_path = f'{base_dir}/{name}'
    # Get the blob object
    blob = bucket.blob(blob_path)
    # Download the blob to a temporary file
    with blob.open('rb') as blob_file:
        # Open the gzipped file
        with gzip.open(blob_file, mode='rt', encoding='utf-8') as gz:
            # Read the CSV file
            reader = csv.reader(gz)
            for row in reader:
                doc_id, pr_score = row
                page_rank_scores[doc_id] = float(pr_score)
    return page_rank_scores

def convert_pr_scores_to_dict(pr_scores_df):
    return pd.Series(pr_scores_df.pr.values,index=pr_scores_df.doc_id).to_dict()



def get_wikipedia_page_title(doc_id):
    # Construct the URL for the Wikipedia API
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=info&pageids={doc_id}&inprop=url&format=json"

    # Send a GET request to the Wikipedia API
    response = requests.get(url)

    # Check if the response was successful
    if response.status_code == 200:
        data = response.json()
        # Extract the page title from the response
        page = next(iter(data['query']['pages'].values()))
        try:
            title=page['title']
        except:
            pass
            title ="Unknown Title"
        return title
    else:
        return "Unknown Title"



class Backend:
    def __init__(self):
        colnames_pr = ['doc_id', 'pr'] # pagerank
        colnames_id_len = ['doc_id', 'len'] # dict_id_len

        # pagerank score
        self.pr_scores = pd.read_csv('gs://ir-proj/pr/part-00000-1ff1ba87-95eb-4744-acae-eef3dd0fa58f-c000.csv.gz', names=colnames_pr, compression='gzip')
        self.pr_scores_dict = convert_pr_scores_to_dict(self.pr_scores)

       # dict of title and id document
        self.title_dict = InvertedIndex.read_index("title_id", "titles", "ir-proj").title_dict

        # total term in corpus
        self.total_term = InvertedIndex.read_index("total_term", "index", "ir-proj").term_total

        # Id's of documents and there len
        dict_id_len = pd.read_csv('gs://ir-proj/doc_id_len/part-00000-1740a912-5c7a-428a-859e-4b8ab436c316-c000.csv.gz', names=colnames_id_len, compression='gzip')
        self.dict_id_len = dict_id_len.set_index('doc_id')['len'].to_dict()

        # Assuming the read_index method returns an InvertedIndex instance
        self.index_text = InvertedIndex.read_index("text", "index", "ir-proj")

        # Ensure that InvertedIndex has an attribute that gives the total number of documents
        self.total_docs = self.index_text.total_document_count()
        print(self.index_text.term_total.items())
        self.idf_scores = {term.lower(): math.log(self.total_docs / self.index_text.get_doc_frequency(term)) for term in set(self.index_text.df.keys())}

    def calculate_tf_idf(self, query):
        tf_idf_scores = {}
        # Tokenize and filter out stopwords
        tokens = [token.group() for token in re.finditer(r'\w+', query.lower()) if token.group() not in all_stopwords]

        # Calculate DF and IDF for each term
        # doc_freq = {term: self.index_title.get_doc_frequency(term) for term in tokens}
        # idf = {term: math.log(self.total_docs / df) if df > 0 else 0 for term, df in doc_freq.items()}

        # Calculate TF and aggregate TF-IDF for each document
        for term in tokens:
            postings = self.index_text.get_posting_list(term, "text", "ir-proj")
            for doc_id, freq in postings:
                tf = freq
                tf_idf = tf * self.idf_scores[term]
                tf_idf_scores[doc_id] = tf_idf_scores.get(doc_id, 0) + tf_idf

        # Combine TF-IDF scores with PageRank and fetch Wikipedia titles
        res = []
        for doc_id, tf_idf_score in sorted(tf_idf_scores.items(), key=lambda item: item[1], reverse=True)[:100]:
            page_rank_score = self.pr_scores_dict.get(doc_id, 0)
            combined_score = tf_idf_score + page_rank_score
            res.append((combined_score, doc_id, self.title_dict.title_dict.get(doc_id)))

        # Sort by the combined score
        res_sorted = sorted(res, key=lambda x: x[0], reverse=True)

        # Extract only the title and document ID from the sorted results
        res_final = [(doc_title, doc_id) for _, doc_id, doc_title in res_sorted]

        # Now, res_final contains tuples of (title, document ID), sorted by combined_score
        print(res_final)
        return res_final


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
  return int(_hash(token),16) % NUM_BUCKETS

# PLACE YOUR CODE HERE


