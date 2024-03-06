from flask import Flask, request, jsonify
from pyngrok import ngrok

from Backend import Backend

import time  # Import the time module

class MyFlaskApp(Flask):
    def __init__(self, *args, **kwargs):
        super(MyFlaskApp, self).__init__(*args, **kwargs)
        self.backend = Backend()  # Initialize the backend attribute correctly

    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/")
def home():
    return "HOME PAGE- LIOR AND NOAM QUEENS"


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []

    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    start_time = time.time()  # Start timing

    # BEGIN SOLUTION
    res = app.backend.calculate_tf_idf_title(query)

    # END SOLUTION

    end_time = time.time()  # End timing
    query_time = end_time - start_time  # Calculate query processing time

    print(f"Query processing time: {query_time} seconds.")  # Print processing time to console or log
    # print(jsonify(res))
    # return jsonify(res)
    # print(jsonify(res))

    return jsonify(res)



@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    res = [{"wiki_id": wiki_id, "title": title} for wiki_id, title in app.backend.calculate_tf_idf_title(query)]
    return res

    # return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

def test():
    import json

    with open('queries_train.json', 'rt') as f:
        queries = json.load(f)

    def average_precision(true_list, predicted_list, k=40):
        true_set = frozenset(true_list)
        predicted_list = predicted_list[:k]
        precisions = []
        for i, doc_id in enumerate(predicted_list):
            if doc_id in true_set:
                prec = (len(precisions) + 1) / (i + 1)
                precisions.append(prec)
        if len(precisions) == 0:
            return 0.0
        return round(sum(precisions) / len(precisions), 3)

    def precision_at_k(true_list, predicted_list, k):
        true_set = frozenset(true_list)
        predicted_list = predicted_list[:k]
        if len(predicted_list) == 0:
            return 0.0
        return round(len([1 for doc_id in predicted_list if doc_id in true_set]) / len(predicted_list), 3)

    def recall_at_k(true_list, predicted_list, k):
        true_set = frozenset(true_list)
        predicted_list = predicted_list[:k]
        if len(true_set) < 1:
            return 1.0
        return round(len([1 for doc_id in predicted_list if doc_id in true_set]) / len(true_set), 3)

    def f1_at_k(true_list, predicted_list, k):
        p = precision_at_k(true_list, predicted_list, k)
        r = recall_at_k(true_list, predicted_list, k)
        if p == 0.0 or r == 0.0:
            return 0.0
        return round(2.0 / (1.0 / p + 1.0 / r), 3)

    def results_quality(true_list, predicted_list):
        p5 = precision_at_k(true_list, predicted_list, 5)
        f1_30 = f1_at_k(true_list, predicted_list, 30)
        if p5 == 0.0 or f1_30 == 0.0:
            return 0.0
        return round(2.0 / (1.0 / p5 + 1.0 / f1_30), 3)

    assert precision_at_k(range(10), [1, 2, 3], 2) == 1.0
    assert recall_at_k(range(10), [10, 5, 3], 2) == 0.1
    assert precision_at_k(range(10), [], 2) == 0.0
    assert precision_at_k([], [1, 2, 3], 5) == 0.0
    assert recall_at_k([], [10, 5, 3], 2) == 1.0
    assert recall_at_k(range(10), [], 2) == 0.0
    assert f1_at_k([], [1, 2, 3], 5) == 0.0
    assert f1_at_k(range(10), [], 2) == 0.0
    assert f1_at_k(range(10), [0, 1, 2], 2) == 0.333
    assert f1_at_k(range(50), range(5), 30) == 0.182
    assert f1_at_k(range(50), range(10), 30) == 0.333
    assert f1_at_k(range(50), range(30), 30) == 0.75
    assert results_quality(range(50), range(5)) == 0.308
    assert results_quality(range(50), range(10)) == 0.5
    assert results_quality(range(50), range(30)) == 0.857
    assert results_quality(range(50), [-1] * 5 + list(range(5, 30))) == 0.0

    import requests
    from time import time
    # url = 'http://35.232.59.3:8080'
    # http://192.168.68.115:8080/.ngrok.io
    # place the domain you got from ngrok or GCP IP below.
    url = 'https://6176-35-232-193-179.ngrok-free.app'

    qs_res = []
    for q, true_wids in queries.items():
        duration, ap = None, None
        t_start = time()
        try:
            res = requests.get(url + '/search', {'query': q}, timeout=35)
            duration = time() - t_start
            if res.status_code == 200:
                pred_wids, _ = zip(*res.json())
                rq = results_quality(true_wids, pred_wids)
                qs_res.append((q, duration, rq))

        except Exception as e:
            print(f"Request failed for query '{q}' with exception: {e}")
            rq = None
    return qs_res



if __name__ == '__main__':
    ngrok.set_auth_token("2d0PCekwyANHSOeQAN6oIvHGSKW_3MLQVx7ZSvZfdFgsSPoFC")
    public_url = ngrok.connect(5000).public_url
    print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:5000\"")
    # Update any base URLs to use the public ngrok URL
    app.config["BASE_URL"] = public_url
    app.run(port=5000)

    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    # app.run(host='0.0.0.0', port=5000, debug=True)
    # print(test())
