import math

import numpy as np

from inverted_index_gcp import InvertedIndex


class BM25:
    def __init__(self, index, path, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.path = path
        self.N = len(self.index.DL)
        self.AVGDL = sum(index.DL.values()) / self.N

    def calc_idf(self, list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: bm25 idf score
        """
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def get_candidate_documents_BM25(self,query,path,index):
        """
        Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
        and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
        Then it will populate the dictionary 'candidates.'
        For calculation of IDF, use log with base 10.
        tf will be normalized based on the length of the document.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                         Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.

        words,pls: iterator for working with posting.

        Returns:
        -----------
        dictionary of candidates. In the following format:
                                                                   key: pair (doc_id,term)
                                                                   value: tfidf score.
        """
        candidates = []
        avg=0
        for term in np.unique(query):
            if term in index.df:
                pl=InvertedIndex.read_posting_list(self.index,term,path)
                candidates+=pl
                avg+=index.term_total[term]/len(pl)

            else:
                continue
        res=defaultdict(int)
        for doc, tf in candidates:
            res[doc]+=int(tf)
        sstresh=avg*10
        return [doc_id for doc_id,tf in res.items() if tf>sstresh]

    def bm25_search(self, query,path,N=100):
        """
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query.
        This function return a dictionary of scores as the following:
                                                                    key: query_id
                                                                    value: a ranked list of pairs (doc_id, score) in the length of N.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        # YOUR CODE HERE
        scores = {}
        self.idf = self.calc_idf(query)
        term_tf={}
        for term in query:
            if term in self.index.df:
                # dict: key - term value - dict posting list of term (key - doc_id , value - tf)
                term_tf[term] = dict(InvertedIndex.read_posting_list(self.index,term,path))
        docs = self.get_candidate_documents_BM25(query,path,self.index)
        docList = []
        for doc in docs:
            docList.append((doc,self._score(query, doc,term_tf)))
         #sort docScores by scores
        return sorted(docList,key=lambda x: x[1],reverse=True)[:N]


    def _score(self, query, doc_id,term_tf):
        """
        This function calculate the bm25 score for given query and document.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """

        score = 0.0
        if doc_id in self.index.DL:
            doc_len = self.index.DL[doc_id]
            for term in query:
                if term in term_tf and doc_id in term_tf[term]:
                    tfij = term_tf[term][doc_id]
                    numerator = self.idf[term] * tfij * (self.k1 + 1)
                    B = 1 - self.b + self.b * doc_len / self.AVGDL
                    denominator = tfij + self.k1 * B
                    score += (numerator / denominator)
        return score