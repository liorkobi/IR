import requests
import pickle

from inverted_index_gcp import InvertedIndex


class titles:
    def __init__(self):
        self.dict={}

    def get_wikipedia_page_title(self,doc_id):
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



if __name__ == '__main__':
    Titles=titles()
    index_title = InvertedIndex.read_index("title", "index", "ir-proj")

    for term in index_title.df.keys():
        postings = index_title.get_posting_list(term, "text", "ir-proj")
        for doc_id, _ in postings:
            if doc_id not in Titles.dict:
                Titles.dict[doc_id]=Titles.get_wikipedia_page_title(doc_id)
                print(doc_id,Titles.dict[doc_id])

    # Assuming Titles.dict contains the dictionary of document IDs and titles
    with open('titles_dict.pkl', 'wb') as f:
        pickle.dump(Titles.dict, f)

