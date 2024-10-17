#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing
# #### Student Name: Tharkana Vishmika Indrahenaka
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
# In this assignment, the review text was preprocessed by tokenizing, converting to lowercase, and removing short words, stopwords, rare words, and the top 20 most frequent words, resulting in cleaned data saved in processed.csv and a vocabulary stored in vocab.txt.

# ## Importing libraries 

# In[1]:


# Code to import libraries as you need in this assessment, e.g.,
import pandas as pd
import re
from collections import Counter


# The assignment employed the "Pandas" library to load and manipulate the dataset (assignment3.csv). The "re" module was utilized for tokenizing the review text through regular expressions, ensuring an accurate breakdown of textual data into tokens. Furthermore, "collections.Counter" was leveraged to compute word frequencies across the dataset, enabling the identification and subsequent removal of infrequent words and the top 20 most frequently occurring terms. The combined use of these libraries provided an efficient and systematic approach to the text preprocessing workflow.

# ### 1.1 Examining and loading data
# - Examine the data and explain your findings
# - Load the data into proper data structures and get it ready for processing.

# In this step, This would involve loading the data into a DataFrame, studying the first few rows to get an initial feel for the structure of the data, and being especially attentive to the 'Review Text' column since that is what this whole cleaning and preprocessing task has been for. This will also be useful to take note of what cleaning and transformation may be required on the data.

# In[2]:


# Load stopwords
with open('stopwords_en.txt', 'r') as file:
    stopwords = set(file.read().splitlines())

# Load the dataset
df = pd.read_csv('assignment3.csv')

# Display data for inspection
df.head()


# In[3]:


reviews = df['Review Text'].tolist()
print("Sample Set of Reviews : ", reviews[:3])


# In[4]:


print(f"Total number of reviews: {len(reviews)}")


# ### 1.2 Pre-processing data
# Perform the required text pre-processing steps.

# ...... Sections and code blocks on basic text pre-processing
# 
# 
# <span style="color: red"> You might have complex notebook structure in this section, please feel free to create your own notebook structure. </span>

# The code checks for **missing** or **null value**s in each column of the DataFrame, helping to identify any potential issues in the data that need to be addressed before further analysis.

# In[5]:


# Check for null values in the DataFrame
null_values = df.isnull().sum()

# Display the columns with their corresponding count of null values
print(null_values)


# The preprocess_text function cleans and tokenizes the review text by splitting it into words based on a **regex pattern**, **converting tokens to lowercase**, **removing words shorter than two character**s, and **filtering out stopwords**. This prepares the text for meaningful analysis by ensuring it is standardized and free from irrelevant words.

# In[6]:


# Function for preprocessing the review text
def preprocess_text(text):
    # Tokenize using regex
    tokens = re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?", text)
    # Convert to lowercase
    tokens = [token.lower() for token in tokens]
    # Filter out short words
    tokens = [token for token in tokens if len(token) >= 2]
    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords]
    return tokens


# The preprocessing function is applied to each review, creating a new column in the DataFrame with cleaned tokens, making the data ready for further analysis such as word frequency counting.

# In[7]:


# Apply preprocessing
df['Processed_Review'] = df['Review Text'].apply(preprocess_text)
print("Preprocessing completed.")


# ### Remove Rare and Frequent Words

# All tokens from the Processed_Review column are combined into a single list, and their frequencies are counted using collections.Counter. This step helps in determining which words are rare and which are frequently occurring.

# In[8]:


# Count term frequencies across all documents
all_tokens = [token for tokens in df['Processed_Review'] for token in tokens]
word_counts = Counter(all_tokens)


# The Processed_Review column is updated to exclude tokens that occur only once across the entire dataset. **Removing such rare words** reduces noise and helps in focusing on more meaningful tokens that appear frequently.

# In[9]:


# Remove words that appear only once
df['Processed_Review'] = df['Processed_Review'].apply(
    lambda tokens: [token for token in tokens if word_counts[token] > 1]
)


# After filtering out rare words, **word frequencie**s are recomputed to find the top 20 most frequent tokens. This helps in identifying commonly occurring words that may not contribute significantly to the context and should be excluded.

# In[10]:


# Recompute word frequencies to find the top 20 most frequent words
all_tokens = [token for tokens in df['Processed_Review'] for token in tokens]
word_counts = Counter(all_tokens)


# The top 20 most frequent words are picked based on document frequency and then removed from each review in the Processed_Review column. This step strengthens the focus towards more contextually relevant words by removing the very common ones.

# In[11]:


# Identify and remove the top 20 most frequent words
top_20_words = [word for word, _ in word_counts.most_common(20)]
df['Processed_Review'] = df['Processed_Review'].apply(
    lambda tokens: [token for token in tokens if token not in top_20_words]
)


# The cleaned tokens are converted back to space-separated strings within the Processed_Review column. This transformation prepares the processed text for output and further analysis in a readable and usable format.

# In[12]:


# Convert lists of tokens to space-separated strings
df['Processed_Review'] = df['Processed_Review'].apply(lambda tokens: ' '.join(tokens))

print("Converted processed reviews to strings.")


# ## Saving required outputs
# Save the requested information as per specification.
# - vocab.txt

# This saves the preprocessed reviews to a new CSV file, entitled processed.csv, containing only the Processed_Review column. The index = False ensures that the saved file doesn't save the DataFrame index.

# In[13]:


# Save the processed data to 'processed.csv'
df[['Processed_Review']].to_csv('processed.csv', index=False)
print("Processed data saved to 'processed.csv'.")


# Then it picks up all the unique tokens from the Processed_Review column and creates a sorted vocabulary. The sorted vocabulary will save as a dictionary in which each word will be mapped to a unique index starting from 0. The vocabulary is written to a text file called vocab.txt, where each line consists of a word in the vocabulary, along with its index in the format word:index. It would serve as a key to understand what the token means in the reviews that have gone through preprocessing.

# In[14]:


# Create a vocabulary of the cleaned reviews
vocab = sorted(set(all_tokens))  # Sort vocabulary alphabetically
vocab_dict = {word: index for index, word in enumerate(vocab)}

# Save to 'vocab.txt' in the correct format
with open('vocab.txt', 'w') as f:
    for word, index in vocab_dict.items():
        f.write(f"{word}:{index}\n")



# ## Summary
# The preprocessed dataset cleaned and transformed the review text in this task. First, it tokenized the text by using a regular expression and changed all words to lowercase. Then, it removed words that were less than two characters and stopwords to keep only meaningful contents. Besides, rare words-words that appeared just once across the reviews-and the top 20 most frequent words have been filtered out in order to reduce the noise and increase the quality of the data. Save cleaned reviews into a new file called processed.csv which will be further used in feature generation and modeling tasks. A sorted vocabulary of the rest of the tokens was further created and saved to vocab.txt, providing a structured way of referring to the conversion of textual data into vectorized formats for further analysis.
