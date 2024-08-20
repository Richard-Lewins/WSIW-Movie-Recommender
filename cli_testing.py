# main.py
from src.data_loader import load_data
from src.preprocess import merge_data, create_user_movie_matrix, build_user_profile
from src.model import train_model_knn_movies, train_model_knn_users, get_movie_recommendations_by_profile
from src.interface import run_interface
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

def main():
    # Load data
    movies, ratings = load_data('data/movies.csv', 'data/ratings.csv')
    
    # Preprocess data
    data = merge_data(movies, ratings)
    user_movie_matrix, columns, index = create_user_movie_matrix(data)
    print(user_movie_matrix)
    
    # Train model
    movie_fit_knn = train_model_knn_movies(user_movie_matrix, n_neighbors=10)

    # TODO: Add Evaluation
    
    #Trying things
    user_fit_knn = train_model_knn_users(user_movie_matrix, n_neighbors=10)
    # this user profile will be generic kids movie lover
    ratings = {'Toy Story (1995)': -4}
    user_profile = build_user_profile(ratings, columns, user_movie_matrix)
    print(user_profile)
    print(get_movie_recommendations_by_profile(user_fit_knn, user_movie_matrix, user_profile, columns))

    # Run interface
    run_interface(movie_fit_knn, user_movie_matrix, columns)

if __name__ == '__main__':
    main()
