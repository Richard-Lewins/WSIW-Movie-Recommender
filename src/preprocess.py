# src/preprocess.py
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

def merge_data(movies, ratings):
    return pd.merge(ratings, movies, on='movieId')

def create_user_movie_matrix(data):
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

