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

#scrapping second source for missing scripts
episodes_to_scrape = episode_metadata_df[episode_metadata_df["Transcript Available?"] == "N"]
episodes_to_scrape = episodes_to_scrape[episodes_to_scrape["id"] != 67] #drop documentary special

for index, ep in episodes_to_scrape.iterrows():
    #get the epnum  and isolate the episode number,if the number is like 02 or 03, remove the 0
    ep_num = ep["epnum"]
    ep_num = ep_num[2:4]
    if ep_num[0] == "0":
        ep_num = ep_num[1]

    url = "https://westwingwiki.com/2014/04/season-" + str(ep["season"]) + "-episode-" + ep_num #url for each episode

    ep_page = requests.get(url)

    soup = BeautifulSoup(ep_page.content, 'html.parser')

    ep_content = soup.find_all('div', class_ = 'entry-content clr') #section of page with script

    ep_paragraphs = ep_content[0].find_all('p') #script section made of 4 paragraphs

    ep_script = ep_paragraphs[-1].get_text() #get last paragraph where the script is and get the text

    ep_id = ep["id"]

    with open('Scripts/ep_id_' + str(ep_id) + '_script.txt', 'w') as f:
        f.write(str(ep_script)) #write to file