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

episode_metadata_df = pd.read_csv("episode_metadata.csv")

corpusdir = "Scripts/"
TWW_corpus = PlaintextCorpusReader(corpusdir, '^.*\.txt')

for infile in sorted(TWW_corpus.fileids()):
    print(infile) # The fileids of each file.

stopwords_en = stopwords.words("english")
custom_stopword_list = ["--", '."', "...", "’", "–"]
stopwords_en.extend(custom_stopword_list)
stopwords_en.extend(string.punctuation)

TWW_corpus_words_no_stop = []
for w in TWW_corpus.words():
  if ((w.lower() not in stopwords_en) & (w.isupper() == False)):
    TWW_corpus_words_no_stop.append(w)

seasons14 = range(1, 5, 1)
seasons57 = range(5, 8, 1)

seasons14_words = []
seasons14_words_no_stop = []

for season in seasons14:
    ep_ids = episode_metadata_df[episode_metadata_df["epnum"].str.match(str(season) + '...' ) == True]["id"]

    season_ep_file_ids = []
    for ep in ep_ids:
        fileid = 'ep_id_' + str(ep) + '_script.txt'
        season_ep_file_ids.append(fileid)
    
    for file_id in season_ep_file_ids:
        seasons14_words.extend(TWW_corpus.words(file_id))

for w in seasons14_words:
    if ((w.lower() not in stopwords_en) & (w.isupper() == False)):
        seasons14_words_no_stop.append(w.lower())

seasons57_words = []
seasons57_words_no_stop = []

for season in seasons57:
    ep_ids = episode_metadata_df[episode_metadata_df["epnum"].str.match(str(season) + '...' ) == True]["id"]

    season_ep_file_ids = []
    for ep in ep_ids:
        fileid = 'ep_id_' + str(ep) + '_script.txt'
        season_ep_file_ids.append(fileid)
    
    for file_id in season_ep_file_ids:
        seasons57_words.extend(TWW_corpus.words(file_id))

for w in seasons57_words:
    if ((w.lower() not in stopwords_en) & (w.isupper() == False)):
        seasons57_words_no_stop.append(w.lower())

#seasons 1-4
# set max features and whether we want stopwords or not
cvect_tww_14 = CountVectorizer()
X_tww_14 = cvect_tww_14.fit_transform(seasons14_words_no_stop) 
vocab_tww_14 = cvect_tww_14.get_feature_names()

NUM_TOPICS = 5
lda = LatentDirichletAllocation(n_components=NUM_TOPICS) 

lda.fit(X_tww_14) 

# look at the top tokens for each topic

TOP_N = 10  # change this to see the top N words per topic

topic_norm = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]

print("Seasons 1-4")
for idx, topic in enumerate(topic_norm):
    print("Topic id: {}".format(idx))
    #print(topic)
    top_tokens = np.argsort(topic)[::-1] 
    for i in range(TOP_N):
      print('{}: {}'.format(vocab_tww_14[top_tokens[i]], topic[top_tokens[i]]))
    print()

#seasons 5-7
# set max features and whether we want stopwords or not
cvect_tww_57 = CountVectorizer()
X_tww_57 = cvect_tww_57.fit_transform(seasons57_words_no_stop) 
vocab_tww_57 = cvect_tww_57.get_feature_names()

NUM_TOPICS = 5
lda = LatentDirichletAllocation(n_components=NUM_TOPICS) 

lda.fit(X_tww_57) 

# look at the top tokens for each topic

TOP_N = 10  # change this to see the top N words per topic

topic_norm = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]

print("Seasons 5-7")
for idx, topic in enumerate(topic_norm):
    print("Topic id: {}".format(idx))
    #print(topic)
    top_tokens = np.argsort(topic)[::-1] 
    for i in range(TOP_N):
      print('{}: {}'.format(vocab_tww_57[top_tokens[i]], topic[top_tokens[i]]))
    print()