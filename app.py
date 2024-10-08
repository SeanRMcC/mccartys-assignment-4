from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)


# TODO: Fetch dataset, initialize vectorizer and LSA here

newsgroups = fetch_20newsgroups(subset="all")
docs = newsgroups.data

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(docs)

reduced_rank = 500
svd = TruncatedSVD(n_components=reduced_rank)
reduced_tfidf = svd.fit_transform(tfidf_matrix)


def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # TODO: Implement search engine here
    # return documents, similarities, indices 

    query_vec = vectorizer.transform([query])
    query_vec_red = svd.transform(query_vec)

    similarities = cosine_similarity(query_vec_red, reduced_tfidf).flatten()

    sim_index_pairs = [[similarities[i], i] for i in range(len(similarities))]

    sim_index_pairs.sort(reverse=True)

    top5 = sim_index_pairs[:5]

    top_docs = [docs[i] for _, i in top5]
    top_similarities = [sim for sim, _ in top5]
    top_indices = [i for _, i in top5]

    return top_docs, top_similarities, top_indices

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(debug=True)
