from utils import *

stemmer = PorterStemmer()


class Backend:
    def __init__(self):
        """
        Initializes the Backend object by loading necessary data for search operations.
        This includes reading and processing PageRank and vector space model scores,
        creating dictionaries for document titles and lengths, calculating average document length,
        and pre-computing IDF scores for terms.
        """

        self.tf_idf_scores = None
        self.pr_scores = pd.read_csv('gs://ir-proj/pr/part-00000-1ff1ba87-95eb-4744-acae-eef3dd0fa58f-c000.csv.gz',names=colnames_pr, compression='gzip')
        pr_scores_dict_tmp = convert_pr_scores_to_dict(self.pr_scores)
        self.pr_scores_dict = {
        doc_id: int(math.log10(score)) if score > 1 else score
        for doc_id, score in pr_scores_dict_tmp.items()
        }
        self.pv_scores = get_pv()
        self.pv_scores_dict = {
        doc_id: int(math.log10(score)) if score > 1 else score
        for doc_id, score in self.pv_scores.items()
        }

        # dict of title and id document
        self.title_dict = InvertedIndex.read_index("title_id", "titles", "ir-proj").title_dict

        self.index_title = InvertedIndex.read_index("title", "index", "ir-proj")
        # self.index_title = InvertedIndex.read_index("title", "index", "with_stemming")


        # with wtemming
        self.index_text = InvertedIndex.read_index("postings_gcp", "index", "with_stemming")
        # no stemming
        # self.index_text = InvertedIndex.read_index("text", "index", "new_index_text")

        # Id's of documents and there len
        # no stemming
        # dict_id_len = pd.read_csv('gs://ir-proj/DL/part-00000-0cc0ccdf-560d-4de0-a3db-b8a0e69db3f8-c000.csv.gz',
        #                           names=colnames_id_len, compression='gzip')
        # with stemming
        dict_id_len = pd.read_csv('gs://ir-proj/doc_id_len/part-00000-1740a912-5c7a-428a-859e-4b8ab436c316-c000.csv.gz',
                                  names=colnames_id_len, compression='gzip')
        self.index_text.DL = dict_id_len.set_index('doc_id')['len'].to_dict()

        # corpus size-as calc in gcp
        self.N = 6348910
        self.avg_DL = sum(self.index_text.DL.values()) / self.N
        self.idf_scores = {
            term.lower(): math.log((self.N - self.index_text.get_doc_frequency(term) + 0.5) / (
                    self.index_text.get_doc_frequency(term) + 0.5) + 1)
            for term in set(self.index_text.df.keys())
        }

    def calculate_bm25_scores(self, query):
        """
        Calculates BM25 scores for a given query against all documents in the index.

        Parameters:
        - query: A list of query terms (after processing like stemming).

        Returns:
        - A Counter object with document IDs as keys and their BM25 scores as values.
        """

        k2 = 10.0
        k1 = 1
        b = 0.75
        candidates = Counter()

        for term in query:
            if term not in self.index_text.df: continue  # Skip terms not in the index directly

            posting_list = self.index_text.get_posting_list(term, "text", "with_stemming")
            idf = self.idf_scores.get(term, 0)  # Assuming idf_scores is pre-computed outside this method

            for doc_id, freq in posting_list:
                if doc_id not in self.index_text.DL: continue  # Skip if doc_id not in DL

                len_doc = self.index_text.DL[doc_id]
                numerator = idf * freq * (k1 + 1)
                denominator = freq + k1 * (1 - b + b * len_doc / self.avg_DL)
                bm25_score = numerator / denominator
                bm25_score *= (k2 + 1) * freq / (k2 + freq)
                candidates[doc_id] += bm25_score

        return candidates
    def search_by_title(self, query_tokens):
        """
        Searches documents by titles using the given query tokens.

        Args:
            query_tokens (list of str): Tokens extracted from the user query, used to match document titles.

        Returns:
            dict: A dictionary with document IDs as keys and their associated scores based on title matches.
        """

        title_scores = {}
        for term in query_tokens:
            if term in self.index_title.df:
                posting_list = self.index_title.get_posting_list(term, "title", "ir-proj")
                for doc_id, freq in posting_list:
                    title_scores[doc_id] = title_scores.get(doc_id, 0) + freq
        return title_scores


    def search_and_merge(self, query):
        """
        Processes the query, performs a search by both text and title, merges the results based on weights,
        and applies PageRank score adjustments before returning the top results.

        Args:
            query (str): The user's search query.

        Returns:
            list of tuples: Each tuple contains the document ID and title for the top matching documents.
        """

        query_tokens_stem = [stemmer.stem(token.group()) for token in re.finditer(r'\w+', query.lower()) if
                             token.group() not in all_stopwords]
        query_tokens = [token.group() for token in re.finditer(r'\w+', query.lower()) if
                        token.group() not in all_stopwords]

        # Classify the query
        TEXT_WEIGHT, TITLE_WEIGHT = optimize_weights(query, query_tokens)

        # Step 1: Retrieve documents for both text and title
        text_results = self.calculate_bm25_scores(query_tokens_stem)
        title_results = self.search_by_title(query_tokens)

         # Step 2: Merge results with a chosen strategy (e.g., weighted scores)

        merged_results = {}
        for doc_id, score in {**text_results, **title_results}.items():
            text_score = text_results.get(doc_id, 0) * TEXT_WEIGHT
            title_score = title_results.get(doc_id, 0) * TITLE_WEIGHT
            merged_results[doc_id] = text_score + title_score

         #Step 3:  Apply page rank score adjustments Sort and return the final results
        final_results = sorted(merged_results.items(),
                               key=lambda x: x[1] + self.pv_scores_dict.get(x[0], 0)*0.9 + self.pr_scores_dict.get(x[0], 0),
                               reverse=True)[:100]
        res = [(str(doc_id), self.title_dict.get(doc_id, 'Unknown Title')) for doc_id, _ in final_results]

        return res

