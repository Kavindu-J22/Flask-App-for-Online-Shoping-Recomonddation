#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 2&3
# #### Student Name: Tharkana Vishmika Indrahenaka Henaka Ralalage
# #### Student ID: s4065784
# 
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used: please include all the libraries you used in your assignment, e.g.,:
# * pandas
# * re
# * numpy
# 
# ## Introduction
# In this assignment, you will focus on text processing and feature extraction of clothing reviews to build and evaluate classification algorithms. Features representations via BoW and unweighted/TF-IDF weighted word embeddings in **Task 2** also explored the contributions of them on influenced the model. The goal of **Task 3** was to determine whether the addition of said features and the concatenation review text with title would improve classification accuracy an overall.

# ## Importing libraries 

# In[1]:


# Code to import libraries as you need in this assessment, e.g.,
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import KeyedVectors
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm 
from scipy.sparse import dok_matrix


# ## Task 2. Generating Feature Representations for Clothing Items Reviews

# ##### Load Data and Vocabulary
# Load processed.csv and vocab.txt to ensure you have access to the processed reviews and vocabulary.

# In[2]:


# Load processed data (assuming the column 'Review Text' contains the processed reviews)
df = pd.read_csv('processed.csv')

# Load the vocabulary with word indices
vocab = {}
with open('vocab.txt', 'r') as f:
    for line in f:
        word, idx = line.strip().split(':')
        vocab[word] = int(idx)

# Confirm that the vocabulary is loaded correctly
print(f"Vocabulary size: {len(vocab)}")


# ##### Handle NaN Values
# Identified NaN values and replace them with an empty string to maintain consistency.

# In[3]:


# Replace NaN values in 'Review Text' with an empty string
df['Processed_Review'].fillna('No review', inplace=True)

# Confirm that there are no more NaN values
print(df['Processed_Review'].isnull().sum())  # Should print 0


# ##### Generate Count Vector Representations (Bag-of-Words Model)
# Use CountVectorizer to generate count vectors based on vocabulary from Task 1.

# In[4]:


# Create CountVectorizer using the provided vocabulary
vectorizer = CountVectorizer(vocabulary=vocab.keys())

# Transform the review text into count vectors (only 'Review Text' column)
count_vectors = vectorizer.transform(df['Processed_Review'])


# ##### Generate Embedding-Based Feature Representations
# Using GoogleNews-vectors-negative300.bin as the pre-trained word embedding model.

# ##### **Note:** Before executing the .ipynb files, please ensure that the GoogleNews-vectors-negative300.bin file is downloaded, extracted from its zip archive, and placed in the same directory as the notebook. This is required to properly load the pretrained word vectors for the embedding-based feature representations.

# In[5]:


# Load a pre-trained word embedding model
embedding_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Define a function to generate the unweighted vector representation
def get_unweighted_vector(review):
    words = review.split()
    word_vectors = [embedding_model[word] for word in words if word in embedding_model]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(embedding_model.vector_size)

# Generate unweighted vectors for each review
unweighted_vectors = np.array([get_unweighted_vector(review) for review in tqdm(df['Processed_Review'], desc="Generating Unweighted Vectors")])


# ##### Generate Weighted Embedding Representations (TF-IDF Weighted)
# Use TfidfVectorizer to generate TF-IDF weights and incorporate them into the word vectors.

# In[6]:


# Create a TF-IDF vectorizer using the same vocabulary
tfidf_vectorizer = TfidfVectorizer(vocabulary=vocab.keys())
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Processed_Review'])

# Map vocabulary indices to their corresponding words
index_to_word = {v: k for k, v in vocab.items()}

# Define a function to generate the weighted vector representation
def get_weighted_vector(review_index):
    tfidf_scores = tfidf_matrix[review_index]
    weighted_vectors = []
    for idx, score in zip(tfidf_scores.indices, tfidf_scores.data):
        word = index_to_word[idx]
        if word in embedding_model:
            weighted_vectors.append(embedding_model[word] * score)
    if weighted_vectors:
        return np.sum(weighted_vectors, axis=0)
    else:
        return np.zeros(embedding_model.vector_size)

# Generate weighted vectors for each review
weighted_vectors = np.array([get_weighted_vector(i) for i in tqdm(range(len(df)), desc="Generating Weighted Vectors")])


# ### Saving outputs
# Save the count vector representation as per spectification.
# - count_vectors.txt

# In[7]:


# code to save output data...
# Save count_vectors to count_vectors.txt in the required format
with open('count_vectors.txt', 'w') as f:
    for i in tqdm(range(count_vectors.shape[0]), desc="Generating Count Vectors"):
        indices = count_vectors[i].indices
        counts = count_vectors[i].data
        # Create the sparse format as required: #webindex,word_idx:count
        row_representation = f"#{i}," + ",".join([f"{indices[j]}:{counts[j]}" for j in range(len(indices))])
        f.write(row_representation + "\n")


# Save unweighted vectors and weighted vectors

# In[8]:


# Save unweighted vectors to unweighted_vectors.txt
with open('unweighted_vectors.txt', 'w') as f:
    for i, vector in enumerate(unweighted_vectors):
        vector_str = ','.join(map(str, vector))
        f.write(f"#{i},{vector_str}\n")

# Save weighted vectors to weighted_vectors.txt
with open('weighted_vectors.txt', 'w') as f:
    for i, vector in enumerate(weighted_vectors):
        vector_str = ','.join(map(str, vector))
        f.write(f"#{i},{vector_str}\n")

# Save unweighted vectors to unweighted_vectors.npy
np.save('unweighted_vectors.npy', unweighted_vectors)

# Save weighted vectors to weighted_vectors.npy
np.save('weighted_vectors.npy', weighted_vectors)


# ## Task 3. Clothing Review Classification

# ###### Load Data and Feature Representations
# Need to load the processed data and the feature representations (count vectors, unweighted vectors, and weighted vectors) generated in Task 2.

# In[9]:


# Load processed dataset
df = pd.read_csv('processed.csv')

# Replace NaN values in 'Review Text' with an empty string
df['Processed_Review'].fillna('No review', inplace=True)

# Confirm that there are no more NaN values
print(df['Processed_Review'].isnull().sum())  # Should print 0



# In[10]:


# Load the full dataset which contains the 'Title' column
full_df = pd.read_csv('assignment3.csv')

print("Length of df:", len(df))
print("Length of full_df:", len(full_df))  


# In[11]:


# Reset the index of `df` to ensure proper alignment
df.reset_index(drop=True, inplace=True)

# Filter `full_df` based on the index of `df` to ensure they match
full_df_filtered = full_df.loc[df.index].reset_index(drop=True)

# Create the combined DataFrame using `df` and the filtered `full_df`
combined_df = pd.DataFrame({
    'Processed_Review': df['Processed_Review'],
    'Recommended IND': full_df_filtered['Recommended IND']
})

# Check the shape of combined_df
print("Shape of combined_df:", combined_df.shape)


# In[12]:


# Load count vector representation
count_vectors = []
with open('count_vectors.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        vector = {int(word.split(':')[0]): int(word.split(':')[1]) for word in parts[1:] if word}  # Ensure word is not empty
        count_vectors.append(vector)

# Convert count_vectors into a sparse matrix (assuming same dimensions as vocab size)
vocab_size = len(open('vocab.txt').readlines())  # Length of vocabulary
X_count = dok_matrix((len(count_vectors), vocab_size))

for i, vector in enumerate(count_vectors):
    for word_index, freq in vector.items():
        X_count[i, word_index] = freq

# Load unweighted and weighted vectors
unweighted_vectors = np.load('unweighted_vectors.npy')
weighted_vectors = np.load('weighted_vectors.npy')

# Load labels for classification task from 'Recommended IND'
y = combined_df['Recommended IND'].values


# In[13]:


print("Shape of X_count:", X_count.shape)
print("Shape of unweighted_vectors:", unweighted_vectors.shape)
print("Shape of weighted_vectors:", weighted_vectors.shape)
print("Length of y:", len(y))


# ##### Choose Machine Learning Models

# In[14]:


models = {
    'Logistic Regression': LogisticRegression(max_iter=1000)
}


# ##### Q1 - Language Model Comparisons
# For Q1, train and evaluate each model on all three feature representations: count vectors, unweighted vectors, and weighted vectors.

# In[15]:


# Function to perform 5-fold cross-validation and return average metrics
def evaluate_model(model, X, y):
    accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    precision = cross_val_score(model, X, y, cv=5, scoring='precision').mean()
    recall = cross_val_score(model, X, y, cv=5, scoring='recall').mean()
    f1 = cross_val_score(model, X, y, cv=5, scoring='f1').mean()
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

# Evaluate models on count vectors, unweighted vectors, and weighted vectors
results = {}
for model_name, model in models.items():
    print(f"Evaluating {model_name}...")
    
    # Count Vectors
    print("On Count Vectors:")
    results[(model_name, 'Count Vectors')] = evaluate_model(model, X_count, y)
    
    # Unweighted Embedding Vectors
    print("On Unweighted Vectors:")
    results[(model_name, 'Unweighted Vectors')] = evaluate_model(model, unweighted_vectors, y)
    
    # Weighted Embedding Vectors
    print("On Weighted Vectors:")
    results[(model_name, 'Weighted Vectors')] = evaluate_model(model, weighted_vectors, y)

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame.from_dict(results, orient='index')
print(results_df)


# The model that performed the best in evaluating the **Logistic Regression model** was the **Count Vectors (Bag of Words)**, which got an accuracy of 87.53%, precision of 90.35%, recall of 94.90% and f1-score of 92.57%. It shows the Count Vector representation is capturing the most amount of predictive information in classification of clothing reviews. The **Unweighted** and **TF-IDF Weighted Embedding Vectors** had slightly lower accuracy (85.87% and 86.19%, respectively), but still with high recall, which poses a wide net for positive cases. To put it all together, the performance of extracted features in terms of performance and generalization capabilities are precisely evaluated and we can conclude that the Count Vector representation by itself has been able to provide excellent overall balance compared to other feature representation models.

# ##### Q2 - Effect of Additional Features (Title & Review Text)
# Generate feature vectors for 'Title'

# In[16]:


# Load the full dataset which contains the 'Title' column
full_df = pd.read_csv('assignment3.csv')

# Check for null values in the DataFrame
null_values = full_df.isnull().sum()

# Display the columns with their corresponding count of null values
print(null_values)


# **Generate Count Vectors for Title**

# In[17]:


# Extract the 'Title' column and fill NaNs with empty strings
titles = full_df['Title'].fillna('')

# Use the same vocabulary from Task 1 to generate count vectors for `Title`
title_vectorizer = CountVectorizer(vocabulary=vocab.keys())

# Transform the title text into count vectors
title_count_vectors = title_vectorizer.transform(titles)

# Save the count vectors to a .txt file
with open('title_count_vectors.txt', 'w') as f:
    for i in range(title_count_vectors.shape[0]):
        indices = title_count_vectors[i].indices
        counts = title_count_vectors[i].data
        row_representation = f"#{i}," + " ".join([f"{indices[j]}:{counts[j]}" for j in range(len(indices))])
        f.write(row_representation + "\n")

# Save count vectors to .npy file
np.save('title_count_vectors.npy', title_count_vectors)


# **Generate Unweighted Embedding Vectors for Title**

# In[18]:


# Define function to generate unweighted vector for each title
def get_unweighted_title_vector(title):
    words = title.split()
    word_vectors = [embedding_model[word] for word in words if word in embedding_model]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(embedding_model.vector_size)

# Generate unweighted vectors for `Title`
title_unweighted_vectors = np.array([get_unweighted_title_vector(title) for title in titles])

# Save unweighted vectors to a .txt file
with open('title_unweighted_vectors.txt', 'w') as f:
    for i, vector in enumerate(title_unweighted_vectors):
        vector_str = ','.join(map(str, vector))
        f.write(f"#{i},{vector_str}\n")

# Save unweighted vectors to .npy file
np.save('title_unweighted_vectors.npy', title_unweighted_vectors)


# **Generate Weighted Embedding Vectors for Title Using TF-IDF**

# In[19]:


# Generate TF-IDF weights for `Title`
title_tfidf_vectorizer = TfidfVectorizer(vocabulary=vocab.keys())
title_tfidf_matrix = title_tfidf_vectorizer.fit_transform(titles)

# Define function to generate weighted vector for each title
def get_weighted_title_vector(title_index):
    tfidf_scores = title_tfidf_matrix[title_index]
    weighted_vectors = []
    for idx, score in zip(tfidf_scores.indices, tfidf_scores.data):
        word = index_to_word[idx]
        if word in embedding_model:
            weighted_vectors.append(embedding_model[word] * score)
    if weighted_vectors:
        return np.sum(weighted_vectors, axis=0)
    else:
        return np.zeros(embedding_model.vector_size)

# Generate weighted vectors for `Title`
title_weighted_vectors = np.array([get_weighted_title_vector(i) for i in range(len(titles))])

# Save weighted vectors to a .txt file
with open('title_weighted_vectors.txt', 'w') as f:
    for i, vector in enumerate(title_weighted_vectors):
        vector_str = ','.join(map(str, vector))
        f.write(f"#{i},{vector_str}\n")

# Save weighted vectors to .npy file
np.save('title_weighted_vectors.npy', title_weighted_vectors)


# **Combine and Evaluate Feature Representations**

# In[20]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Load existing `Processed_Review` vectors (already generated)
unweighted_vectors = np.load('unweighted_vectors.npy')
weighted_vectors = np.load('weighted_vectors.npy')

# Combine `Title` and `Processed_Review` vectors
combined_unweighted_vectors = np.hstack((title_unweighted_vectors, unweighted_vectors))
combined_weighted_vectors = np.hstack((title_weighted_vectors, weighted_vectors))

# Define model for evaluation
model = LogisticRegression(max_iter=1000)

# Function to evaluate the model
def evaluate_model(X, y):
    accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    return accuracy

# Evaluate using Title only
print("Accuracy using Title Count Vectors:", evaluate_model(title_count_vectors, full_df['Recommended IND']))
print("Accuracy using Title Unweighted Vectors:", evaluate_model(title_unweighted_vectors, full_df['Recommended IND']))
print("Accuracy using Title Weighted Vectors:", evaluate_model(title_weighted_vectors, full_df['Recommended IND']))

# Evaluate combined vectors
print("Accuracy using Combined Unweighted Vectors:", evaluate_model(combined_unweighted_vectors, full_df['Recommended IND']))
print("Accuracy using Combined Weighted Vectors:", evaluate_model(combined_weighted_vectors, full_df['Recommended IND']))


# As can be seen from the results, adding more data (the title) definitely helps the model learn better. Using just the title vectors, the model managed **87.45%** accuracy for Count Vectors, **86.92%** accuracy for Unweighted Embeddings and an accuracy of **86.97%** with TF-IDF Weighted Embeddings. The overall shifted from 79.5% to around **88.96%** for Unweighted Embeddings and TF-IDF Weighted Embeddings. And this means that adding both the title and description in a review helps improve how well our model does, implying that more data generally leads to better predictions. When features are combined it gives a better context to the reviews making it easier for classifier to classify them.

# ## Summary
# In this assignment, clothing Reviews- Preprocessing and Feature SelectionIn this assignment, you will use information about customers to predict whether the owner of a clothing review is a woman or man. The algorithms were then fed these representations to learn how review sentiment can be classified; the Bag-of-Words approach best dealt with it. Additional experiments to evaluate the efficacy of incorporating more features were also performed, and results obtained in this case demonstrate that the model accuracy increases by including both title and review text together with BoW features for classification.Text Classification:Further analysis of the usefulness offered by additional information was carried out here.
