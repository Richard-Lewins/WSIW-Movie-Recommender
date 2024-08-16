import pandas as pd

def load_data(movies_path, ratings_path):
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    return movies, ratings

