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

corpusdir = "Scripts/"
TWW_corpus = PlaintextCorpusReader(corpusdir, '^.*\.txt')

for infile in sorted(TWW_corpus.fileids()):
    print(infile) # The fileids of each file.

stopwords_en = stopwords.words("english")
custom_stopword_list = ["--", '."', "...", "’", "–"]
stopwords_en.extend(custom_stopword_list)
stopwords_en.extend(string.punctuation)

vocab_pre = set(TWW_corpus.words())
vocab = []
for w in vocab_pre:   
    if ((w.lower() not in stopwords_en) & (w.isupper() == False)):    
        vocab.append(w)
vocab = set(vocab)

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
  doc_pre = TWW_corpus.words(TWW_corpus.fileids()[i])
  doc = []
  for w in doc_pre:
      if ((w.lower() not in stopwords_en) & (w.isupper() == False)):    
        doc.append(w)

  for token in doc:
    token_idx = tokens_to_idx[token] 

    counts_matrix[i, token_idx] += 1 

doc_frequency = [0] * len(vocab)

for i in range(len(TWW_corpus.fileids())):
    ep_vocab_pre = set(TWW_corpus.words(TWW_corpus.fileids()[i]))
    ep_vocab = []
    for w in ep_vocab_pre:
      if ((w.lower() not in stopwords_en) & (w.isupper() == False)):    
        ep_vocab.append(w)
    ep_vocab = set(ep_vocab)

    for token in ep_vocab:
        doc_frequency[tokens_to_idx[token]] += 1


# once we have all of our counts, we divide by the number of documents in our corpus
doc_frequency = [df / num_ep for df in doc_frequency]
#idf
inverse_doc_frequency = [np.log(1/df) for df in doc_frequency]

#tfidf
tww_tfidf = np.array(counts_matrix * inverse_doc_frequency)

fileids = TWW_corpus.fileids()
fileid_to_id = {}
i = 0
for id in fileids:
    fileid_to_id[id] = i
    i += 1

tfidf_eps_print = ["ep_id_1_script.txt", "ep_id_44_script.txt", "ep_id_65_script.txt", "ep_id_79_script.txt", 
"ep_id_107_script.txt", "ep_id_126_script.txt", "ep_id_136_script.txt"]

for id in tfidf_eps_print:
    top_tokens = np.argsort(tww_tfidf[fileid_to_id[id]])[::-1]
    top_tokens = top_tokens[0:10]
    #print(top_tokens)
    print("\n")
    print(id)
    for t in top_tokens:
        print(idx_to_tokens[t])
