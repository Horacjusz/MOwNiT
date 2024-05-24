from flask import Flask, render_template, request, redirect
import os
import numpy as np
from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix, load_npz
from sklearn.decomposition import TruncatedSVD
from collections import Counter
import time

app = Flask(__name__)

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
DOCS_DIR = os.path.join("docs")
DATA_DIR = os.path.join("data")
TERM_MATRIX_FILE = os.path.join(DATA_DIR, "term_matrix.npz")
KEYWORD_SET_FILE = os.path.join(DATA_DIR, "keyword_set.npz")
WORD_INDEXES_FILE = os.path.join(DATA_DIR, "word_indexes.npy")
FILE_URLS_FILE = os.path.join(DATA_DIR, "file_urls.npy")

# Function to load a document from file
def load_document(filename):
    words = []
    url = ""
    if filename.endswith(".txt"):
        filepath = os.path.join(SCRIPT_PATH, DOCS_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            url = next(file)[:-1]
            text = file.read().lower()
            words = word_tokenize(text)
    return url, words

# Function to load keyword set from file
def load_keyword_set():
    return np.load(os.path.join(SCRIPT_PATH, KEYWORD_SET_FILE))['arr_0']

# Function to load word indexes from file
def load_word_indexes():
    return np.load(os.path.join(SCRIPT_PATH, WORD_INDEXES_FILE), allow_pickle=True).item()

# Function to load file URLs from file
def load_file_urls():
    return np.load(os.path.join(SCRIPT_PATH, FILE_URLS_FILE), allow_pickle=True).item()

# Function to load term matrix from file
def load_term_matrix():
    return load_npz(os.path.join(SCRIPT_PATH, TERM_MATRIX_FILE))

# Function to create a vector for a document
def vector(words, keyword_set, word_indexes):
    row_indices = []
    col_indices = []
    data = []
    word_counter = Counter(words)
    for word, count in word_counter.items():
        if word in word_indexes:
            row_indices.append(0)
            col_indices.append(word_indexes[word])
            data.append(count)

    # Create the sparse matrix
    vector_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(1, len(keyword_set)), dtype=np.float64)

    # Calculate the length of the vector
    vector_length = np.linalg.norm(vector_matrix.toarray())

    # Normalize the vector
    normalized_vector = vector_matrix / vector_length

    return normalized_vector

# Function to find similar documents
def find_similar_documents(query, k, keyword_set, word_indexes, term_matrix, file_urls, file_urls_list_keys, with_svd=False):
    query_words = word_tokenize(query.lower())
    query_vector = vector(query_words, keyword_set, word_indexes)
    
    values = None
    
    if with_svd:
        
        svd = TruncatedSVD(n_components=150)
        svd.fit(term_matrix)
        
        us_matrix = svd.transform(term_matrix)
        v_t_matrix = np.array(svd.components_)
        
        values = ((query_vector @ us_matrix) @ v_t_matrix).flatten().astype(np.float64)
        
    else:
        values = term_matrix.dot(query_vector.transpose()).toarray().flatten().astype(np.float64)

    # Find top k indices
    top_k_indices = np.argsort(values)[::-1][:k]
    
    similar_documents = []
    
    for idx in top_k_indices:
        similarity = round(values[idx], 2)
        if similarity > 0:  # If similarity is non-zero
            document_filename = file_urls_list_keys[idx]
            document_url = file_urls[document_filename]
            similar_documents.append((document_url, similarity))
        
    if not similar_documents:
        similar_documents = [("No matches found", 0)]  # Return a message of no match
        
    return similar_documents

keyword_set = load_keyword_set()
word_indexes = load_word_indexes()
file_urls = load_file_urls()
term_matrix = load_term_matrix()

print("Shape of term_matrix:", term_matrix.shape)

file_urls_list_keys = list(file_urls.keys())

@app.route("/", methods=["GET", "POST"])
def search():
    global keyword_set, word_indexes, file_urls, term_matrix, file_urls_list_keys
    if request.method == "POST":
        query = request.form["query"]
        k = request.form.get("k", 30)  # Set default value to 5
        if k == "":
            k = 30  # Set default value if k is empty
        else:
            k = int(k)
        k = min(k, len(file_urls))
        
        with_svd = False
        # with_svd = True
        
        if with_svd :
            term_matrix = term_matrix.transpose()
        
        start = time.time()
        similar_documents = find_similar_documents(query, k, keyword_set, word_indexes, term_matrix, file_urls, file_urls_list_keys, with_svd)
        end = time.time()
        execution_time = end - start
        
        return render_template("results.html", query=query, k=k, similar_documents=similar_documents, execution_time=execution_time)
    
    return render_template("index.html")

@app.route("/<path:url>")
def redirect_to_url(url):
    return redirect(url)

# Run the Flask app only if this file is executed as the main module
if __name__ == "__main__":
    app.run(debug=True)