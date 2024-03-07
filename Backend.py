import math
import re

import gcsfs
import numpy as np
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
from nltk.corpus import wordnet


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
    return pd.Series(pr_scores_df.pr.values, index=pr_scores_df.doc_id).to_dict()


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
            title = page['title']
        except:
            pass
            title = "Unknown Title"
        return title
    else:
        return "Unknown Title"


class Backend:
    def __init__(self):
        colnames_pr = ['doc_id', 'pr']  # pagerank
        colnames_id_len = ['doc_id', 'len']  # dict_id_len

        # pagerank score
        self.pr_scores = pd.read_csv('gs://ir-proj/pr/part-00000-1ff1ba87-95eb-4744-acae-eef3dd0fa58f-c000.csv.gz',
                                     names=colnames_pr, compression='gzip')
        self.pr_scores_dict = convert_pr_scores_to_dict(self.pr_scores)

        # dict of title and id document
        self.title_dict = InvertedIndex.read_index("title_id", "titles", "ir-proj").title_dict

        # total term in corpus
        self.total_term = InvertedIndex.read_index("total_term", "index", "ir-proj").term_total

        # Assuming the read_index method returns an InvertedIndex instance
        self.index_text = InvertedIndex.read_index("text", "index", "ir-proj")

        # Id's of documents and there len
        dict_id_len = pd.read_csv('gs://ir-proj/doc_id_len/part-00000-1740a912-5c7a-428a-859e-4b8ab436c316-c000.csv.gz',
                                  names=colnames_id_len, compression='gzip')
        self.index_text.DL = dict_id_len.set_index('doc_id')['len'].to_dict()
        self.N = len(self.index_text.DL)
        self.avg_DL = sum(self.index_text.DL.values()) / self.N
        # Ensure that InvertedIndex has an attribute that gives the total number of documents
        self.total_docs = self.index_text.total_document_count()
        print(self.index_text.term_total.items())
        self.idf_scores = {term.lower(): math.log(self.total_docs / self.index_text.get_doc_frequency(term)) for term in
                           set(self.index_text.df.keys())}


    def expand_query_with_synonyms(self, query):
        expanded_query = set()

        # Tokenize the query
        query_tokens = query.split()

        # Iterate over each token in the query
        for token in query_tokens:
            # Get synonyms from WordNet
            synonyms = set()
            for syn in wordnet.synsets(token):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name().lower())

            # Add the original token and its synonyms to the expanded query
            expanded_query.add(token.lower())
            expanded_query.update(synonyms)


        return expanded_query

    def get_candidate_documents_and_scores(self, query_to_search):
        candidates = {}
        for term in np.unique(query_to_search):
            if term in self.index_text.df.keys():
                list_of_doc = self.index_text.get_posting_list(term, "text", "ir-proj")
                normalized_tfidf = [(doc_id, (freq / self.index_text.DL[doc_id]) * math.log(len(self.index_text.DL) / self.index_text.df[term], 10))
                                   for
                                   doc_id, freq in list_of_doc]

                for doc_id, tfidf in normalized_tfidf:
                    candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

        return candidates

    def calc_idf(self, list_of_tokens):
        """
        This function calculates the idf values according to the precomputed BM25 idf scores for each term in the query.

        Parameters:
        -----------
        list_of_tokens: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                        key: term
                                                        value: bm25 idf score
        """
        idf = {}
        for term in list_of_tokens:
            if term in self.idf_scores:
                idf[term] = self.idf_scores[term]
            else:
                # Optionally handle terms not present in the precomputed scores,
                # for example, by assigning them a default score.
                pass

        return idf


    def calculate_tf_idf(self, query):
        tf_idf_scores = {}
        # Tokenize and filter out stopwords
        tokens = [token.group() for token in re.finditer(r'\w+', query.lower()) if token.group() not in all_stopwords]

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
            res.append((combined_score, str(doc_id), self.title_dict.get(doc_id)))

        # Sort by the combined score
        res_sorted = sorted(res, key=lambda x: x[0], reverse=True)

        # Extract only the title and document ID from the sorted results
        res_final = [(doc_id,doc_title ) for _,doc_id, doc_title in res_sorted]

        # Now, res_final contains tuples of (title, document ID), sorted by combined_score
        return res_final


    def calculate_bm25_scores(self, query):
        # self.k1 = 1.0
        # self.b = 0.9
        k2 = 10.0
        k1 = 1.5
        b = 0.75
        res = []
        # create an empty Counter object to store document scores
        candidates = Counter()

        # tokenize and remove stopwords from the query
        # query = [token.group() for token in re.finditer(r'\w+', query.lower()) if token.group() not in all_stopwords]

        # loop through each term in the query
        for term in query:
            # check if the term exists in the corpus
            if term in self.index_text.df:
                # read the posting list of the term
                posting_list = self.index_text.get_posting_list(term, "text", "ir-proj")

                # calculate idf of the term
                df = self.index_text.df[term]
                idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

                # loop through each (doc_id, freq) pair in the posting list
                for doc_id, freq in posting_list:
                    # check if the doc_id exists in the corpus
                    if doc_id in self.index_text.DL.keys():
                        len_doc = self.index_text.DL[doc_id]

                        # calculate bm25 score of the term for the document
                        numerator = idf * freq * (k1 + 1)
                        denominator = (freq + k1 * (1 - b + b * len_doc / self.avg_DL))
                        bm25_score = numerator / denominator
                        bm25_score = bm25_score * ((k2 + 1) * freq / (k2 + freq))

                        # add the bm25 score to the document's score in the candidates Counter
                        candidates[doc_id] += bm25_score
        return candidates

    def search_by_title(self, query_tokens):
        # Initialize a dictionary to store scores for documents
        title_scores = {}

        # Tokenize the query, similar to how it's done in text search
        # query_tokens = [token.group() for token in re.finditer(r'\w+', query.lower()) if
        #                 token.group() not in all_stopwords]

        # Loop through each token in the query
        for term in query_tokens:

            # Assuming you can get a posting list for titles similar to text
            if term in self.lower_title_dict.values():
                posting_list = self.index_title.get_posting_list(term, "title", "ir-dict")

                # Loop through each document in the posting list
                for doc_id, freq in posting_list:
                    # Increment score for document; for simplicity, each match increases score by 1
                    # Consider a more sophisticated scoring method as needed
                    title_scores[doc_id] = title_scores.get(doc_id, 0) + freq

        # Convert scores to a list of tuples and sort by score
        sorted_title_scores = sorted(title_scores.items(), key=lambda x: x[1], reverse=True)

        # For simplicity, just return the sorted scores; adjust as needed
        return sorted_title_scores

    def search_and_merge(self, query):
        TEXT_WEIGHT = 0.1
        TITLE_WEIGHT = 0.9
        # query_tokens = [token.group() for token in re.finditer(r'\w+', query.lower()) if
        #                 token.group() not in all_stopwords]
        #
        query_tokens = self.expand_query_with_synonyms(query)
        # if len(query_tokens) > 1:
        #     stemmed_tokens = [ps.stem(token) for token in query_tokens if token not in all_stopwords]
        #     text_results = self.calculate_bm25_scores(stemmed_tokens)
        # else:
        #     text_results = self.calculate_bm25_scores(query_tokens)

        # Step 1: Retrieve documents for both text and title
        text_results = self.calculate_bm25_scores(query_tokens)  # Assuming this or similar function is used for text
        title_results = self.search_by_title(query_tokens)  # You would need to implement something similar for title search

        # Step 2: Merge results with a chosen strategy (e.g., weighted scores)
        merged_results = {}
        for doc_id, score in text_results.items():
            merged_results[doc_id] = merged_results.get(doc_id, 0) + score * TEXT_WEIGHT

        for doc_id, score in title_results:
            merged_results[doc_id] = merged_results.get(doc_id, 0) + score * TITLE_WEIGHT

        for doc_id in merged_results.keys():

            page_rank_score = self.pr_scores_dict.get(doc_id, 0)
            if page_rank_score > 1:
                page_rank_score = int(math.log10(page_rank_score))
            merged_results[doc_id] += page_rank_score

        # Step 3: Sort and return final results
        final_results = sorted(merged_results.items(), key=lambda x: x[1], reverse=True)[:100]

        res = [(str(doc_id), self.title_dict.get(doc_id))
               for doc_id, _ in final_results]

        return res

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

# From HW1
# def most_viewed(pages):
#   """Rank pages from most viewed to least viewed using the above `wid2pv`
#      counter.
#   Parameters:
#   -----------
#     pages: An iterable list of pages as returned from `page_iter` where each
#            item is an article with (id, title, body)
#   Returns:
#   --------
#   A list of tuples
#     Sorted list of articles from most viewed to least viewed article with
#     article title and page views. For example:
#     [('Langnes, Troms': 16), ('Langenes': 10), ('Langenes, Finnmark': 4), ...]
#   """
#   # YOUR CODE HERE
#   result = []
#   for p in pages:
#         id = p[0]
#         if str(id) in wid2pv:
#             result.append((str(p[1]), wid2pv[id]))  # title, page views
#         else:
#           result.append((str(p[1]), 0))
#   # Sort the result in descending order based on page views
#   result.sort(key=lambda x: x[1], reverse=True)
#   return result
