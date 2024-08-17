# main.py
from src.data_loader import load_data
from src.preprocess import merge_data, create_user_movie_matrix, build_user_profile
from src.model import train_model_knn, get_movie_recommendations_by_profile
from src.interface import run_interface
from sklearn.model_selection import train_test_split

def main():
    # Load data
    movies, ratings = load_data('data/movies.csv', 'data/ratings.csv')
    
    # Preprocess data
    data = merge_data(movies, ratings)
    user_movie_matrix, columns, index = create_user_movie_matrix(data)
    
    # Train model
    model_knn = train_model_knn(user_movie_matrix, n_neighbors=10)
    
    # TODO: Add Evaluation
    user_ratings = {'Toy Story (1995)': 5, 'Toy Story 2 (1999)': 5, 'Toy Story 3 (2010)': 5}
    user_profile = build_user_profile(user_ratings, columns, user_movie_matrix)
    print(user_profile)
    print(get_movie_recommendations_by_profile(model_knn, user_profile, columns))
    # Run interface
    run_interface(model_knn, user_movie_matrix, columns)

if __name__ == '__main__':
    main()
