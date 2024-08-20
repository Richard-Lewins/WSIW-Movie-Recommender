# src/preprocess.py
import pandas as pd
import numpy as np
from difflib import get_close_matches
from scipy.sparse import csr_matrix
import re

def strip_year(movie_title):
    """
    Removes the year from a movie title, if present (As it might ruin comparisons)
    """
    return re.sub(r'\s*\(\d{4}\)$', '', movie_title)

# Basic "Autocomplete" feature
def find_closest_movies(movie_name, movie_titles, n=3, cutoff=0.4):
    """
    Finds the closest n matches to a movie name in a list of movie titles.

    Returns titles that contain the movie_name as a substring first,
    followed by the closest matches based on similarity.
    """
    # Strip years from movie titles for comparison
    movie_titles_without_year = [strip_year(title).lower() for title in movie_titles]
    movie_name_lower = strip_year(movie_name).lower()

    # Find titles that contain the movie_name as a substring
    substring_matches = [title for title, stripped_title in zip(movie_titles, movie_titles_without_year) if movie_name_lower in stripped_title]

    # Find the closest matches excluding the ones already found as substrings
    remaining_titles_without_year = [title for title in movie_titles_without_year if title not in [strip_year(m).lower() for m in substring_matches]]
    closest_matches = get_close_matches(movie_name_lower, remaining_titles_without_year, n=n, cutoff=cutoff)

    # Map closest matches back to original titles
    closest_matches_original = [movie_titles[movie_titles_without_year.index(match)] for match in closest_matches]

    # Combine substring matches and closest matches, ensuring no duplicates
    combined_matches = substring_matches + [match for match in closest_matches_original if match not in substring_matches]

    return combined_matches

def merge_data(movies, ratings):
    return pd.merge(ratings, movies, on='movieId')

def create_user_movie_matrix(data):
    # Subtract 3 from each rating to make the ratings 0-indexed
    data['rating'] = data['rating'] - 3
    user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')
    return csr_matrix(user_movie_matrix.fillna(0)), user_movie_matrix.columns, user_movie_matrix.index

def build_user_profile(user_ratings, movie_titles, user_movie_matrix):
    # User_ratings is in the form of a sparse matrix with movie titles as keys and ratings as values
    # If a movie title is not in the movie_titles, ignore it
    user_profile = np.zeros(len(movie_titles))
    for movie, rating in user_ratings.items():
        if movie in movie_titles:
            movie_index = movie_titles.get_loc(movie)
            user_profile[movie_index] = rating
    
    return csr_matrix(user_profile)

