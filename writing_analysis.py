import os
import matplotlib.pyplot as plt
import nltk
from nltk.probability import FreqDist
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.decomposition import LatentDirichletAllocation 
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize

corpusdir = "Scripts/"
TWW_corpus = PlaintextCorpusReader(corpusdir, '^.*\.txt')

for infile in sorted(TWW_corpus.fileids()):
    print(infile) # The fileids of each file.

# stopwords_en = stopwords.words("english")
# custom_stopword_list = ["--", '."', "...", "'", "-", "'", "’", "–"]
# stopwords_en.extend(custom_stopword_list)
# stopwords_en.extend(string.punctuation)

# TWW_corpus_words_no_stop = []
# for w in TWW_corpus.words():
#   if ((w.lower() not in stopwords_en) & (w.isupper() == False)):
#     TWW_corpus_words_no_stop.append(w)

#freq1 = FreqDist(TWW_corpus_words_no_stop)
#freq1.plot(10)
#print(freq1.most_common(n=10))

# # set max features and whether we want stopwords or not
# cvect_tww = CountVectorizer(stop_words='english', max_features = 1000)
# X_tww = cvect_tww.fit_transform(TWW_corpus.raw().split()) 
# vocab_tww = cvect_tww.get_feature_names()

# NUM_TOPICS = 10
# lda = LatentDirichletAllocation(n_components=NUM_TOPICS) 

# lda.fit(X_tww) 

# # look at the top tokens for each topic

# TOP_N = 10  # change this to see the top N words per topic

# topic_norm = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]

# for idx, topic in enumerate(topic_norm):
#     print("Topic id: {}".format(idx))
#     #print(topic)
#     top_tokens = np.argsort(topic)[::-1] 
#     for i in range(TOP_N):
#       print('{}: {}'.format(vocab_tww[top_tokens[i]], topic[top_tokens[i]]))
#     print()

vocab = set(TWW_corpus.words())
# build our idx_to_token dictionary
idx_to_tokens = {}
tokens_to_idx = {}

i=0

for token in vocab:
  tokens_to_idx[token] = i 
  idx_to_tokens[i] = token 
  i+=1

num_ep = 148
num_tokens = len(vocab) 
counts_matrix = np.zeros((num_ep, num_tokens))

for i in range(len(TWW_corpus.fileids())):
  doc = TWW_corpus.words(TWW_corpus.fileids()[i])
  for token in doc:
    token_idx = tokens_to_idx[token] 

    counts_matrix[i, token_idx] += 1 

doc_frequency = [0] * len(vocab)

for i in range(len(TWW_corpus.fileids())):
    ep_vocab = set(TWW_corpus.words(TWW_corpus.fileids()[i]))
    for token in ep_vocab:
        doc_frequency[tokens_to_idx[token]] += 1


# once we have all of our counts, we divide by the number of documents in our corpus
doc_frequency = [df / num_ep for df in doc_frequency]
#idf
inverse_doc_frequency = [np.log(1/df) for df in doc_frequency]

#tfidf
tww_tfidf = np.array(counts_matrix * inverse_doc_frequency)

top_tokens = np.argsort(tww_tfidf[0])[::-1]
top_tokens = top_tokens[0:10]
print(top_tokens)
for t in top_tokens:
  print(idx_to_tokens[t])