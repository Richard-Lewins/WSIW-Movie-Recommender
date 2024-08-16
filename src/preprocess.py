# src/preprocess.py
import pandas as pd
from scipy.sparse import csr_matrix

def merge_data(movies, ratings):
    return pd.merge(ratings, movies, on='movieId')

def create_user_movie_matrix(data):
    user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')
    return csr_matrix(user_movie_matrix.fillna(0)), user_movie_matrix.columns, user_movie_matrix.index
