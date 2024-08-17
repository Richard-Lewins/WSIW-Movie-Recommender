from sklearn.neighbors import NearestNeighbors

def train_model_knn(user_movie_matrix, n_neighbors=10):
    model_knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    model_knn.fit(user_movie_matrix.T)  # Transpose to fit on movies
    return model_knn

def get_movie_recommendations_by_name(model_knn, user_movie_matrix, movie_name, movie_titles, n_recommendations=5):
    movie_index = movie_titles.get_loc(movie_name)
    if movie_index is None:
        return []
    distances, indices = model_knn.kneighbors(user_movie_matrix.T[movie_index].reshape(1, -1), n_neighbors=n_recommendations + 1)
    recommendations = [movie_titles[i] for i in indices.flatten() if i != movie_index]
    return recommendations

def get_movie_recommendations_by_profile(model_knn, user_profile, movie_titles, n_recommendations=5):
    pass
