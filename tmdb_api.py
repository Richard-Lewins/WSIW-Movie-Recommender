# Used to get information about movies from TMDB API
import requests
import os 
from dotenv import load_dotenv

load_dotenv()

def get_movie_info(movie_title):
    api_key = os.getenv('TMDB_API_KEY')
    # Seperate into year and title (year is 6 last characters)
    movie_query = movie_title[:-6]
    print(movie_query, ':')
    # Remove year, removing brackets as well
    year_query = movie_title[-5:-1]
    url = f"https://api.themoviedb.org/3/search/movie?query={movie_query}&include_adult=true&year={year_query}"
    headers = {
        'Accept ': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    response = requests.get(url, headers=headers)
    data = response.json()
    if "results" in data and len(data['results']) > 0:
        movie = data['results'][0]
        res = {
            'title': movie['title'],
            'overview': movie['overview'],
            'release_date': movie['release_date'],
            'poster_path': movie['poster_path']
        }
        print(res, '\n\n')
        return res
    else:
        return None