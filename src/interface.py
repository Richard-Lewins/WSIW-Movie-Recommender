from src.model import get_movie_recommendations
from difflib import get_close_matches
import re

def strip_year(movie_title):
    """
    Removes the year from a movie title, if present (As it might ruin comparisons)
    """
    return re.sub(r'\s*\(\d{4}\)$', '', movie_title)

def find_closest_movies(movie_name, movie_titles, n=3, cutoff=0.4):
    """
    Finds the closest n matches to a movie name in a list of movie titles,
    """
    # Strip years from movie titles for comparison
    movie_titles_without_year = [strip_year(title).lower() for title in movie_titles]
    movie_name_lower = strip_year(movie_name).lower()
    matches = get_close_matches(movie_name_lower, movie_titles_without_year, n=n, cutoff=cutoff)
    return [movie_titles[movie_titles_without_year.index(match)] for match in matches]

# TODO: Change this to be web based interface
def run_interface(model_knn, user_movie_matrix, movie_titles):
    while True:
        user_input = input("Enter a movie title or 'exit' to quit: ")
        if user_input.lower() == 'exit':
            break
        
        # Find the closest movie titles
        matches = find_closest_movies(user_input, movie_titles)
        print(matches)
        
        #Ask "Did you mean?". If the user enters "yes", then use the first match as the movie title.
        if matches:
            print(f"Did you mean: {matches[0]}?")
            user_input = input("Enter 'yes' or 'no': ")
            if user_input.lower() == 'yes':
                user_input = matches[0]
            else:
                print("Movie not found in the database.")
                continue

        if user_input in movie_titles:
            movie_index = movie_titles.get_loc(user_input)
            distances, indices = model_knn.kneighbors(user_movie_matrix[:, movie_index].T, n_neighbors=6)
            similar_movies = [movie_titles[indices.flatten()[i]] for i in range(1, len(indices.flatten()))]
            print(f"Movies similar to '{user_input}':")
            for movie in similar_movies:
                print(movie)
        else:
            print("Movie not found in the database.")

