import requests
from bs4 import BeautifulSoup
import pandas as pd

episode_metadata_df = pd.read_csv("episode_metadata.csv") #read episode metadata csv into dataframe

ids_to_scrape = episode_metadata_df[episode_metadata_df["Transcript Available?"] == "Y"]["id"].to_list() #get list of episode ids that have transcripts
#ids_to_scrape_test = [1,2]

for ep_id in ids_to_scrape:
    url = "http://www.westwingtranscripts.com/search.php?flag=getTranscript&id=" + str(ep_id)

    ep_page = requests.get(url)

    soup = BeautifulSoup(ep_page.content, 'html.parser')
    script = soup.pre

    script = str(script)#convert to string
    script = script[5:]#remove <pre> at start
    size = len(script)#get length
    script = script[:size - 6]#remove </pre> at end

    with open('Scripts/ep_id_' + str(ep_id) + '_script.txt', 'w') as f:
        f.write(str(script)) #write to file