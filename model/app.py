from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load required data: vocabulary, embedding model, and trained model
vocab = {}
with open('vocab.txt', 'r') as f:
    for line in f:
        word, idx = line.strip().split(':')
        vocab[word] = int(idx)

embedding_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Load the trained logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(np.load('combined_weighted_vectors.npy'), pd.read_csv('assignment3.csv')['Recommended IND'])  # Train with existing data

# Load the training data used for fitting the vectorizer (e.g., reviews and titles combined)
train_data = pd.read_csv('assignment3.csv')
all_text = train_data['Title'].fillna('') + " " + train_data['Review Text'].fillna('')

# Fit the TF-IDF vectorizer on the training data
tfidf_vectorizer = TfidfVectorizer(vocabulary=vocab.keys())
tfidf_vectorizer.fit(all_text)

# Load stopwords
with open('stopwords_en.txt', 'r') as file:
    stopwords = set(file.read().splitlines())

# Helper function: preprocess the text (for both title and review)
def preprocess_text(text):
    tokens = re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?", text)
    tokens = [token.lower() for token in tokens if len(token) >= 2 and token not in stopwords]
    return ' '.join(tokens)

# Helper function: generate weighted vector for a given text
def get_weighted_vector(text, tfidf_vectorizer, embedding_model):
    tokens = preprocess_text(text)
    tfidf_matrix = tfidf_vectorizer.transform([tokens])
    weighted_vectors = []
    index_to_word = {v: k for k, v in vocab.items()}

    for idx, score in zip(tfidf_matrix.indices, tfidf_matrix.data):
        word = index_to_word[idx]
        if word in embedding_model:
            weighted_vectors.append(embedding_model[word] * score)
    
    if weighted_vectors:
        return np.sum(weighted_vectors, axis=0)
    else:
        return np.zeros(embedding_model.vector_size)

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.get_json()

    title = data.get('title', '')
    review_text = data.get('review_text', '')

    # Generate vectors for title and review
    title_vector = get_weighted_vector(title, tfidf_vectorizer, embedding_model)
    review_vector = get_weighted_vector(review_text, tfidf_vectorizer, embedding_model)

    # Combine title and review vectors
    combined_vector = np.hstack((title_vector, review_vector)).reshape(1, -1)

    # Predict using the logistic regression model
    prediction = model.predict(combined_vector)
    recommendation = "Recommended" if prediction[0] == 1 else "Not Recommended"

    # Return the result as JSON
    return jsonify({
        'prediction': recommendation
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
