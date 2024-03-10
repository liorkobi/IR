# Information Retrieval System with Flask Backend

## Project Overview
This project presents an advanced search engine built on top of a Flask web server, designed to efficiently retrieve and rank Wikipedia articles based on user queries. It utilizes several information retrieval techniques and algorithms, including BM25, Binary search, and PageRank, to ensure high-quality search results. The system is capable of handling different types of search queries by analyzing the content, titles, and anchor texts of articles.

## System Components

### Backend
The core of the system, responsible for query processing, document retrieval, and ranking. It incorporates:

- Precomputed scores for TF-IDF, PageRank.
- Inverted indexes for quick lookup of documents.
- Algorithms for scoring and ranking documents based on query relevance.

### Flask Web Server
Serves as the interface between the user and the backend system. It offers several endpoints to interact with the search engine, including:

- `/search`: Main search endpoint, returns top 100 results based on query.
- 
## Deployment
This application is accessible via the following URL: [http://34.170.46.156:8080](http://34.170.46.156:8080). This public IP allows direct interaction with the web server and the backend search engine from anywhere on the internet.

## Usage
To use the system, navigate to the respective endpoints  via a web browser or API tool (like Postman or `curl`), appending query parameters as needed (e.g., `?query=hello+world`). For endpoints requiring POST requests, JSON payloads should be sent with the required data.

## Contributions
This project is a collaborative effort by Lior and Noam, aiming to explore and implement cutting-edge techniques in information retrieval within an accessible web-based application.
