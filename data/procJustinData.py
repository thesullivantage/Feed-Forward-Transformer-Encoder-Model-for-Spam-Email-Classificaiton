import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import re
import random
import unicodedata
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization
import pickle
import tensorflow as tf

'''
    Pre-processing script for transformer model.
    In using other datasets, some configuration may be necessary.
'''

def vectorize_data(dat, vectorizer):
    '''
    function to vectorize numpy array (vectorizer already fitted)
    
    MARK DEV: isinstance testing of vectorizer and numpy array
    
    '''
    return vectorizer(dat)

path_to_csv ='spam_or_not_spam.csv' # LOAD YOUR OWN DATA HERE
# set file_encoding to the file encoding (utf8, latin1, etc.)
# Probe the encoding of your data on the command line.
file_encoding = 'ISO-8859-1'        
with open(path_to_csv, encoding=file_encoding, errors = 'replace') as my_csv:
    df = pd.read_csv(my_csv)

# enforce string
df['email'] = df['email'].astype(str)

# convert to lowercase
df['email'] = df['email'].str.lower()

# remove excess whitespace
df['email'] = df['email'].apply(lambda x: ' '.join(x.split()))

# extra text standardization steps - we will exclude these for now
# # Remove punctuation
# df['email'] = df['email'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
# # Remove URLs
# df['email'] = df['email'].apply(lambda x: re.sub(r'http\S+', '', x))
# # Remove email addresses
# df['email'] = df['email'].apply(lambda x: re.sub(r'\S*@\S*\s?', '', x))
# Remove numerical data (optional, based on your needs)
# df['email'] = df['email'].apply(lambda x: re.sub(r'\d+', '', x))


# Calculate the token lengths for each email as 'token_length' column
df['token_length'] = df['email'].str.split().map(len)

# Calculate the 95th percentile of the sequence length
percentile = 95
percentile_data = int(round(np.percentile(df['token_length'], percentile), 0))

# Filter the DataFrame to only include emails with a token length at or below the 95th percentile; so we shouldn't be classifying any emails longer than this
# Actually, this is handled inside of TextVectorizer
# filtered_df = df[df['token_length'] <= percentile_data]
# filtered_df = filtered_df.drop(columns=['token_length'])

tokens = set()
eng_maxlen = 0

for eng in df['email'].to_numpy():
    eng_tok = eng.split()
    eng_maxlen = max(eng_maxlen, len(eng_tok))
    ### update set if unique token not found
    tokens.update(eng_tok)

print(f"Total Number of Unique tokens in dataset: {len(tokens)}")
print(f"The 95th percentile of token length is: {percentile_data}")
print(f"Number of emails: {len(df)}")

if not os.path.exists('justin_data_diagnostics'):
    os.makedirs('justin_data_diagnostics')
    
with open(f'justin_data_diagnostics/data_metrics_{percentile}.txt', 'w+') as f:
    (f"Total Number of Unique tokens in dataset: {len(tokens)}")
    f.write(f"95th percentile of token length: {percentile_data}")
    f.write(f"Number of emails: {len(df)}")

# Plot histogram of segment length in tokens
plt.figure(figsize=(10, 6))
plt.hist(df['token_length'], bins=500, alpha=0.7, color='blue', edgecolor='black')
plt.title(f'Histogram of Token Lengths in Emails\nTruncated to {percentile}th Percentile')
plt.xlabel('Token Length')
plt.ylabel('Number of Emails')
plt.axvline(percentile_data, color='red', linestyle='dashed', linewidth=2, label=f'{percentile}th Percentile:\n{percentile_data}')
plt.xlim(0,2000)
plt.legend()
plt.savefig(f'justin_data_diagnostics/token_len_{percentile}.png')

# maximum numb. vocab words (tokens) we'd like to use; can experiment with different lengths.
vocab_size_en = 10000
# set up text vectorizer
vectorizer = TextVectorization(
    max_tokens=vocab_size_en,
    standardize=None, # we've already done this, above
    split="whitespace",
    output_mode="int",
    output_sequence_length=percentile_data, # truncate OR pad to 95th percentile of sequence length, in tokens
)

text_dat_total = df['email']
# fit and saving vectorizer data
print('Fitting Text Vectorizer...')
vectorizer.adapt(text_dat_total)

vocabulary = vectorizer.get_vocabulary()

# save vectorizer data for fitting of other data, later, if desired
print('Saving Vectorizer Data...')
if not os.path.exists('vectorizer_data'):
    os.makedirs('vectorizer_data')
with open("vectorizer_data/vectorizer.pickle", "wb") as fp:
    data = {
        "vec_config":  vectorizer.get_config(),
        "vec_weights": vectorizer.get_weights(),
        "vec_vocabulary": vocabulary,
        "sequence_length": percentile_data
    }
    pickle.dump(data, fp)


vectorized_texts = vectorizer(text_dat_total)

# shuffle df for train-test-validation set splitting.
# random_state=# set for reproducibility; we also reset & drop the previous df's indexing, here.
randomized_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# split-off labels
random_all_text_data = randomized_df['email'].to_numpy()
labels = randomized_df.pop('label').to_numpy()

# split into training (0.8), testing (0.1), and validation (0.1) sets 
X_train, X_temp, y_train, y_temp = train_test_split(random_all_text_data, labels, test_size=0.2, random_state=42)
X_test, X_valid, y_test, y_valid = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
del(X_temp, y_temp)    

# apply (already fit) vectorizer to features
X_train_vec = vectorizer(X_train)
X_test_vec = vectorizer(X_test)
X_valid_vec = vectorizer(X_valid)

# save vectorized & unvectorized data
print('Saving Vectorizer Data...')
if not os.path.exists('vectorizer_data'):
    os.makedirs('vectorizer_data')
with open("vectorizer_data/normalized_split_data.pickle", "wb") as fp:
    data = {
        'X_train': X_train,
        'y_train': y_train, 
        'X_test': X_test,
        'X_valid': X_valid, 
        'y_test': y_test, 
        'y_valid': y_valid,
        'X_train_vec': X_train_vec,
        'X_test_vec': X_test_vec,
        'X_valid_vec': X_valid_vec
    }
    pickle.dump(data, fp)
print('Datasets saved.')

