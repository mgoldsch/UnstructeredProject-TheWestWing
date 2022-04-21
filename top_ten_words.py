import os
import matplotlib.pyplot as plt
import nltk
from nltk.probability import FreqDist
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus import stopwords
import string
import pandas as pd
import numpy as np

seasons = range(1, 8, 1) #seasons 1-7
episode_metadata_df = pd.read_csv("episode_metadata.csv")

corpusdir = "Scripts/"
TWW_corpus = PlaintextCorpusReader(corpusdir, '^.*\.txt')

stopwords_en = stopwords.words("english")
custom_stopword_list = ["--", '."', "...", "’", "–"]
stopwords_en.extend(custom_stopword_list)
stopwords_en.extend(string.punctuation)

for season in seasons:
    ep_ids = episode_metadata_df[episode_metadata_df["epnum"].str.match(str(season) + '...' ) == True]["id"]

    season_ep_file_ids = []
    for ep in ep_ids:
        fileid = 'ep_id_' + str(ep) + '_script.txt'
        season_ep_file_ids.append(fileid)

    words_season = []
    for file_id in season_ep_file_ids:
        words_season.extend(TWW_corpus.words(file_id))
    
    words_season_no_stop = []
    for w in words_season:
        if ((w.lower() not in stopwords_en) & (w.isupper() == False)):
            words_season_no_stop.append(w.lower())
    
    freq = FreqDist(words_season_no_stop)
    plt.subplots_adjust(bottom=0.30)
    freq.plot(10, title = "Season " + str(season) + " Top Ten Words")