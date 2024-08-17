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
    user_profile = np.zeros(user_movie_matrix.shape[0])
    count_array = np.zeros(user_movie_matrix.shape[0])
    for movie, rating in user_ratings.items():
        if movie in movie_titles:
            movie_index = movie_titles.get_loc(movie)
            
            for i in range(user_movie_matrix.shape[0]):
                if user_movie_matrix[i, movie_index] != 0:
                    user_profile[i] += 5 - np.abs(user_movie_matrix[i, movie_index] - rating)
                    count_array[i] += 1

    for i in range(user_movie_matrix.shape[0]):
        if count_array[i] != 0:
            user_profile[i] /= count_array[i]
    
    
    return csr_matrix(user_profile)
