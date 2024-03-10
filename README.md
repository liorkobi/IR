# Information Retrieval System with Flask Backend

## Project Overview
This project presents an advanced search engine built on top of a Flask web server, designed to efficiently retrieve and rank Wikipedia articles based on user queries. It utilizes several information retrieval techniques and algorithms, including BM25, Binary search, and PageRank, to ensure high-quality search results. The system is capable of handling different types of search queries by analyzing the content, titles, and anchor texts of articles.

## System Components

### Backend
The core of the system, responsible for query processing, document retrieval, and ranking. It incorporates:

- Precomputed scores for TF-IDF, PageRank, and page views.
- Inverted indexes for quick lookup of documents.
- Algorithms for scoring and ranking documents based on query relevance.

### Flask Web Server
Serves as the interface between the user and the backend system. It offers several endpoints to interact with the search engine, including:

- `/search`: Main search endpoint, returns top 100 results based on query.
- `/search_body`: Searches within the body of articles using TF-IDF and cosine similarity.
- `/search_title`: Returns all results with query words in the title, ordered by the number of distinct query words.
- `/search_anchor`: Searches based on anchor text, with similar ordering criteria as `/search_title`.
- `/get_pagerank`: Retrieves PageRank values for given Wikipedia article IDs.
- `/get_pageview`: Fetches page view statistics for articles.

## Deployment
This application uses `ngrok` for tunneling local server traffic to a public URL, making it accessible over the internet without deploying it to an external server.

## Usage
To use the system, navigate to the respective endpoints via a web browser or API tool (like Postman or `curl`), appending query parameters as needed (e.g., `?query=hello+world`). For endpoints requiring POST requests, JSON payloads should be sent with the required data.

## Contributions
This project is a collaborative effort by Lior and Noam, aiming to explore and implement cutting-edge techniques in information retrieval within an accessible web-based application.

## Setup and Execution
To run this application:

1. Ensure all dependencies are installed, including Flask, `ngrok`, and necessary libraries for information retrieval and data processing.
2. Set your `ngrok` authentication token.
3. Run the Flask app locally, which will automatically initiate an `ngrok` tunnel and print the public URL to the console.

This setup allows for immediate testing and interaction with the search engine through the generated `ngrok` URL.

