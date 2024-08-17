import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# Model fit on movies
def train_model_knn_movies(user_movie_matrix, n_neighbors=10):
    model_knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    model_knn.fit(user_movie_matrix.T)  # Transpose to fit on movies
    return model_knn

# Model fit on users (finds similar users, and uses their ratings to recommend movies)
def train_model_knn_users(user_movie_matrix, n_neighbors=10):
    model_knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    model_knn.fit(user_movie_matrix)  # Fit on users
    return model_knn

def get_movie_recommendations_by_name(movie_fit_knn, user_movie_matrix, movie_name, movie_titles, n_recommendations=5):
    movie_index = movie_titles.get_loc(movie_name)
    if movie_index is None:
        return []
    distances, indices = movie_fit_knn.kneighbors(user_movie_matrix.T[movie_index].reshape(1, -1), n_neighbors=n_recommendations + 1)
    recommendations = [movie_titles[i] for i in indices.flatten() if i != movie_index]
    return recommendations

def get_movie_recommendations_by_profile(user_fit_knn, user_movie_matrix, user_profile, movie_titles, n_recommendations=5):
    # Firstly, find similar users
    # User profile is a csr_matrix with shape (1, n_movies), with ratings as values

    # Indices will be userID
    distances, indices = user_fit_knn.kneighbors(user_profile.reshape(1, -1), n_neighbors=10)

    for i, index in enumerate(indices.flatten()):
        print(f"User {i}: {index}")
    # Get the ratings of the similar users
    movie_ratings_similar_users = user_movie_matrix[indices.flatten()]

    # Compute weighted ratings
    weighted_list = (1 - distances.flatten() / distances.flatten().sum())[:, np.newaxis]
    mean_rating_list = weighted_list * movie_ratings_similar_users.toarray()
    mean_rating_list = mean_rating_list.sum(axis=0) / weighted_list.sum()

    # Filter out movies that the user has already rated
    unrated_movies_indices = np.where(user_profile.toarray().flatten() == 0)[0]
    unrated_movies_ratings = mean_rating_list[unrated_movies_indices]
    unrated_movie_titles = movie_titles[unrated_movies_indices]

    # Sort by rating
    sorted_indices = np.argsort(unrated_movies_ratings)[::-1]
    recommendations = unrated_movie_titles[sorted_indices][:n_recommendations]

    return recommendations

    

    
