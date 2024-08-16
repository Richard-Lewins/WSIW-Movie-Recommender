# src/evaluation.py
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

def evaluate_model(model_knn, user_movie_matrix, test_data, columns, index):
    user_movie_matrix_dense = user_movie_matrix.todense()
    user_movie_df = pd.DataFrame(user_movie_matrix_dense, columns=columns, index=index)
    
    true_ratings = []
    predicted_ratings = []
    
    for _, row in test_data.iterrows():
        user_id = row['userId']
        movie_name = row['title']
        true_ratings.append(row['rating'])
        
        if movie_name in user_movie_df.columns:
            movie_index = user_movie_df.columns.get_loc(movie_name)
            distances, indices = model_knn.kneighbors(user_movie_df.T.iloc[:, movie_index].reshape(1, -1), n_neighbors=6)
            
            # Predicting the rating based on nearest neighbors
            ratings = [user_movie_df.iloc[user_id, indices.flatten()[i]] for i in range(1, len(indices.flatten()))]
            predicted_ratings.append(np.mean(ratings) if ratings else np.nan)
        else:
            predicted_ratings.append(np.nan)
    
    mse = mean_squared_error(true_ratings, predicted_ratings, squared=False)
    return mse
