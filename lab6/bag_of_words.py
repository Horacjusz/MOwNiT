import os
import numpy as np
from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix, vstack as csr_vstack, save_npz, diags
from collections import Counter
import time
import shutil
from useful_functions import time_sentence
from sklearn.preprocessing import normalize

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
DOCS_DIR = os.path.join(SCRIPT_PATH, "docs")
DATA_DIR = os.path.join(SCRIPT_PATH, "data")
TERM_MATRIX_FILE = os.path.join(DATA_DIR, "term_matrix.npz")
KEYWORD_SET_FILE = os.path.join(DATA_DIR, "keyword_set.npz")
WORD_INDEXES_FILE = os.path.join(DATA_DIR, "word_indexes.npy")
FILE_URLS_FILE = os.path.join(DATA_DIR, "file_urls.npy")

FILES = os.listdir(DOCS_DIR)
FILE_NUM = len(FILES)
WORDS_NUM = 0
FIRSTSAVE = True

# Function to load a document from file
def load_document(filename):
    words = []
    url = ""
    if filename.endswith(".txt"):
        filepath = os.path.join(DOCS_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            url = next(file)[:-1]
            text = file.read().lower()
            words = word_tokenize(text)
    return url, words

# Function to generate the keyword set and file URLs
def generate_keyword_set():
    keyword_set = set()
    file_urls = {}
    for filename in FILES:
        url, words = load_document(filename)
        keyword_set.update(words)
        file_urls[filename] = url
    return np.array(list(keyword_set)), file_urls

# Function to generate word indexes
def generate_word_indexes(keyword_set):
    word_indexes = {word: i for i, word in enumerate(keyword_set)}
    return word_indexes

# Function to compute IDF (Inverse Document Frequency)
def compute_idf(term_matrix):
    df = np.sum(term_matrix != 0, axis=0)
    idf_values = np.log(FILE_NUM / (1 + df))
    return idf_values

# Function to generate term matrix
def generate_term_matrix(keyword_set, word_indexes, idf_multiplying=False):
    term_matrix = csr_matrix((0, len(keyword_set)), dtype=np.float64)  # Empty csr_matrix
    for i, filename in enumerate(FILES):
        _, words = load_document(filename)
        doc_vector = vector(words, keyword_set, word_indexes)
        term_matrix = csr_vstack([term_matrix, doc_vector], format='csr')
        
    print("Matrix created")
        
    if idf_multiplying:
        term_matrix = term_matrix.transpose()
        idf_values = compute_idf(term_matrix)
        idf_diag = diags([idf_values], [0], format='csr', shape=(len(idf_values), len(idf_values)))
        term_matrix = term_matrix.multiply(idf_diag)
        term_matrix = term_matrix.transpose()
        
        print("IDFs calculated")
        
        term_matrix = normalize(term_matrix, norm='l2', axis=1)

        print("Rows normalized")
        
    return term_matrix

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

# Function to save matrix to file
def save_matrix(matrix, filename):
    global FIRSTSAVE
    if FIRSTSAVE:
        FIRSTSAVE = False
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        else:
            shutil.rmtree(DATA_DIR)
            os.makedirs(DATA_DIR)
    
    if filename == TERM_MATRIX_FILE :
        save_npz(filename, matrix)
    else :
        np.savez_compressed(filename, matrix)

# Function to save word indexes to file
def save_word_indexes(word_indexes, filename):
    np.save(filename, word_indexes)

# Function to save file URLs to file
def save_file_urls(file_urls, filename):
    np.save(filename, file_urls)

if __name__ == "__main__":
    start_total = time.time()
    
    print("Generating keyword_set...")
    start = time.time()
    keyword_set, file_urls = generate_keyword_set()
    end = time.time()
    print("Finished generating keyword_set in" + time_sentence(end - start), "\n")
    
    print("Saving keyword_set...")
    start = time.time()
    save_matrix(keyword_set, KEYWORD_SET_FILE)
    end = time.time()
    print("Finished saving keyword_set in" + time_sentence(end - start), "\n")
    
    WORDS_NUM = len(keyword_set)
    
    print("Generating word_indexes...")
    start = time.time()
    word_indexes = generate_word_indexes(keyword_set)
    end = time.time()
    print("Finished generating word_indexes in" + time_sentence(end - start), "\n")
    
    print("Saving word_indexes...")
    start = time.time()
    save_word_indexes(word_indexes, WORD_INDEXES_FILE)
    end = time.time()
    print("Finished saving word_indexes in" + time_sentence(end - start), "\n")
    
    print("Saving file_urls...")
    start = time.time()
    save_file_urls(file_urls, FILE_URLS_FILE)
    end = time.time()
    print("Finished saving file_urls in" + time_sentence(end - start), "\n")
    
    print("Generating term_matrix...")
    start = time.time()
    term_matrix = generate_term_matrix(keyword_set, word_indexes, idf_multiplying=True)
    end = time.time()
    print("Finished generating term_matrix in" + time_sentence(end - start), "\n")
    
    print(FILE_NUM,WORDS_NUM)
    print(term_matrix.shape)
    
    print("Saving term_matrix...")
    start = time.time()
    save_matrix(term_matrix, TERM_MATRIX_FILE)
    end = time.time()
    print("Finished saving term_matrix in" + time_sentence(end - start),"\n")
    
    end_total = time.time()
    print("Total execution time:" + time_sentence(end_total - start_total))