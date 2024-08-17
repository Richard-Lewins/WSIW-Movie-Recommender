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
    
    # Run interface
    run_interface(model_knn, user_movie_matrix, columns)

if __name__ == '__main__':
    main()
