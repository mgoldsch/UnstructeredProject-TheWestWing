#referenced https://isabella-b.com/blog/scraping-episode-imdb-ratings-tutorial/ for scraping approach
import requests
from bs4 import BeautifulSoup
import pandas as pd

seasons = range(1, 8, 1)#create list of season 1-7

tww_episodes = []

for s in seasons:
    url = 'https://www.imdb.com/title/tt0200276/episodes?season=' + str(s)
    s_page = requests.get(url)
    soup = BeautifulSoup(s_page.content, 'html.parser')

    # Select all the episode containers from the season's page
    episode_containers = soup.find_all('div', class_='info')

    # For each episode in each season
    for episodes in episode_containers:
        # Get the info of each episode on the page
        season = s
        episode_number = episodes.meta['content']
        title = episodes.a['title']
        rating = episodes.find('span', class_='ipl-rating-star__rating').text
        total_votes = episodes.find('span', class_='ipl-rating-star__total-votes').text
        size = len(total_votes)
        total_votes = total_votes[1:size-1] #so the parenthesis surounding the number arent pulled in
        # Compiling the episode info
        episode_data = [season, episode_number, title, rating, total_votes]

        # Append the episode info to the complete dataset
        tww_episodes.append(episode_data)

tww_episodes_df = pd.DataFrame(tww_episodes, columns = ['season', 'episode_number', 'title', 'rating', 'total_votes'])
tww_episodes_df.to_csv("episode_imdb_rating_data.csv", index = False)#write to csv
