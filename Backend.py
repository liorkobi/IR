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

colnames_pr = ['doc_id', 'pr']  # pagerank
colnames_pv = ['doc_id', 'views']  # pageviews
colnames_id_len = ['doc_id', 'len']  # dict_id_len
import re


def preprocess_query(query):
    # Extract phrases in quotation marks
    phrases_in_quotes = re.findall(r'"([^"]+)"', query)
    # Remove the extracted phrases from the query to handle the rest of the tokens as usual
    query_without_phrases = re.sub(r'"[^"]+"', '', query)

    # Continue with stemming and stopwords removal for the remaining parts of the query
    query_tokens_stem = [stemmer.stem(token.group()) for token in re.finditer(r'\w+', query_without_phrases.lower()) if
                         token.group() not in all_stopwords]
    query_tokens = [token.group() for token in re.finditer(r'\w+', query_without_phrases.lower()) if
                    token.group() not in all_stopwords]

    # Add extracted phrases back into tokens list without stemming but removing stopwords could be considered
    for phrase in phrases_in_quotes:
        # Optionally remove stopwords from the phrase or treat as is for exact match
        query_tokens.append(phrase)
        # Depending on how you handle stemming for phrases, you might want to add them as is or apply some form of normalization

    return query_tokens_stem, query_tokens


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
    # Create a GCS client
    client = storage.Client()
    # Get the bucket and blob from the GCS path
    bucket_name, blob_name = gcs_path.split("//")[1].split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Download the content of the blob
    content = blob.download_as_text()
    # Load the JSON content into a dictionary
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


class Backend:
    def __init__(self):

        # pagerank score
        self.pr_scores = pd.read_csv('gs://ir-proj/pr/part-00000-1ff1ba87-95eb-4744-acae-eef3dd0fa58f-c000.csv.gz',
                                     names=colnames_pr, compression='gzip')
        self.pr_scores_dict = convert_pr_scores_to_dict(self.pr_scores)
        # pageviews score
        self.pv_scores = open_json("gs://ir-dict/wid2pv_result.json")
        # dict of title and id document
        self.title_dict = InvertedIndex.read_index("title_id", "titles", "ir-proj").title_dict
        self.index_title = InvertedIndex.read_index("title", "index", "ir-dict")

        # self.index_text = InvertedIndex.read_index("text", "index", "ir-proj")
        # with wtemming
        self.index_text = InvertedIndex.read_index("postings_gcp", "index", "with_stemming")
        # no stemming
        # self.index_text = InvertedIndex.read_index("text", "index", "new_index_text")

        # # Id's of documents and there len

        # no stemming
        # dict_id_len = pd.read_csv('gs://ir-proj/DL/part-00000-0cc0ccdf-560d-4de0-a3db-b8a0e69db3f8-c000.csv.gz',
        #                           names=colnames_id_len, compression='gzip')
        # with stemming
        dict_id_len = pd.read_csv('gs://ir-proj/doc_id_len/part-00000-1740a912-5c7a-428a-859e-4b8ab436c316-c000.csv.gz',
                                  names=colnames_id_len, compression='gzip')
        self.index_text.DL = dict_id_len.set_index('doc_id')['len'].to_dict()

        # corpus size
        self.N = self.index_text.total_document_count()
        # average document length in corpus
        self.avg_DL = sum(self.index_text.DL.values()) / self.N

        self.idf_scores = {
            term.lower(): math.log((self.N - self.index_text.get_doc_frequency(term) + 0.5) / (
                    self.index_text.get_doc_frequency(term) + 0.5) + 1)
            for term in set(self.index_text.df.keys())
        }

        # Pre-calculate constant part of idf formula BM25
        self.constant_part = 0.5 / (self.N + 0.5)

        lowercase_title_dict = {key: value.lower() for key, value in self.title_dict.items()}
        self.lower_title_dict = lowercase_title_dict
        # # total term in corpus
        # self.total_term = InvertedIndex.read_index("total_term", "index", "ir-proj").term_total



    # def expand_query_with_synonyms(self, query):
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
    #
    #
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
        res_final = [(doc_id, doc_title) for _, doc_id, doc_title in res_sorted]

        # Now, res_final contains tuples of (title, document ID), sorted by combined_score
        return res_final


    def calculate_bm25_scores(self, query):
        k2 = 10.0
        k1 = 1.5
        b = 0.75
        candidates = Counter()
        for term in query:
            # check if the term exists in the corpus
            if term in self.index_text.df:
                # read the posting list of the term

                posting_list = self.index_text.get_posting_list(term, "text", "with_stemming")
                # posting_list = self.index_text.get_posting_list(term, "text", "new_index_text")

                # calculate idf of the term
                df = self.index_text.df[term]
                idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

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
        title_scores = {}
        for term in query_tokens:
            if term in self.index_title.df:
                posting_list = self.index_title.get_posting_list(term, "title", "ir-dict")
                # Loop through each document in the posting list
                for doc_id, freq in posting_list:
                    # Increment score for document
                    title_scores[doc_id] = title_scores.get(doc_id, 0) + freq

        # Convert scores to a list of tuples and sort by score
        sorted_title_scores = sorted(title_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_title_scores


    def search_and_merge(self, query):
            # Existing preprocessing
        query_tokens_stem = [stemmer.stem(token.group()) for token in re.finditer(r'\w+', query.lower()) if
                             token.group() not in all_stopwords]
        query_tokens = [token.group() for token in re.finditer(r'\w+', query.lower()) if
                        token.group() not in all_stopwords]

        # query_tokens_stem, query_tokens = preprocess_query(query)

            # Classify the query
        query_type = classify_query(query)

        # Adjust weights based on query type
        if query_type == 'factual':
            TEXT_WEIGHT = 0.1
            TITLE_WEIGHT = 0.9
        elif query_type == 'detailed_explanation':
            TEXT_WEIGHT = 0.7
            TITLE_WEIGHT = 0.3
        else:  # Broad topic
            TEXT_WEIGHT = 0.6
            TITLE_WEIGHT = 0.4

        # TEXT_WEIGHT = 0.6
        # TITLE_WEIGHT = 0.4
        # query_tokens_stem= [stemmer.stem(token.group()) for token in re.finditer(r'\w+', query.lower()) if
        #                 token.group() not in all_stopwords]
        #
        # query_tokens = [token.group() for token in re.finditer(r'\w+', query.lower()) if
        #                 token.group() not in all_stopwords]
        #
        if len(query_tokens)<=3:
            # if query_tokens[0]  in self.lower_title_dict.values():
            TEXT_WEIGHT = 0.1
            TITLE_WEIGHT = 0.9

        # Step 1: Retrieve documents for both text and title
        text_results = self.calculate_bm25_scores(query_tokens_stem)
        title_results = self.search_by_title(query_tokens)

        # Step 2: Merge results with a chosen strategy (e.g., weighted scores)
        merged_results = {}
        for doc_id, score in text_results.items():
            merged_results[doc_id] = merged_results.get(doc_id, 0) + score * TEXT_WEIGHT

        for doc_id, score in title_results:
            merged_results[doc_id] = merged_results.get(doc_id, 0) + score * TITLE_WEIGHT


            page_rank_score = self.pr_scores_dict.get(doc_id, 0)
            # page_view_score = self.pv_scores.get(doc_id, 0)
            if page_rank_score > 1:
                page_rank_score = int(math.log10(page_rank_score))
            # if page_view_score > 1:
            #     page_view_score = int(math.log10(page_view_score))
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

