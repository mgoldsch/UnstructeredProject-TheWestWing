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


#nltk.download('stopwords')

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

def tfidf_vectorizer(corpus):
  cvect = CountVectorizer(stop_words='english', max_features = 1000) 
  count_matrix = cvect.fit_transform(corpus) 
  features = cvect.get_feature_names() 

  count_matrix = pd.DataFrame(count_matrix.todense()) 

  df_vect = count_matrix.astype(bool).sum(axis=0) 
  df_vect = np.log(len(corpus) / df_vect) 

  return features, np.array(count_matrix * df_vect) 


features, X = tfidf_vectorizer(TWW_corpus.raw().split())
print(X.shape)

# build our idx_to_token dictionary
idx_to_tokens = {}
tokens_to_idx = {}

for i in range(len(features)):
  token = features[i] 
  tokens_to_idx[token] = i 
  idx_to_tokens[i] = token 

top_tokens = np.argsort(X[0])[::-1]
top_tokens = top_tokens[0:10]
print(top_tokens)
for t in top_tokens:
  print(idx_to_tokens[t])