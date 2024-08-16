# main.py
from src.data_loader import load_data
from src.preprocess import merge_data, create_user_movie_matrix
from src.model import train_model_knn
from src.evaluation import evaluate_model
from src.interface import run_interface
from sklearn.model_selection import train_test_split

def main():
    # Load data
    movies, ratings = load_data('data/movies.csv', 'data/ratings.csv')
    
    # Preprocess data
    data = merge_data(movies, ratings)
    user_movie_matrix, columns, index = create_user_movie_matrix(data)
    
    # Split data into training and testing sets
    # train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Train model
    model_knn = train_model_knn(user_movie_matrix, n_neighbors=10)
    
    # Evaluate model
    # mse = evaluate_model(model_knn, user_movie_matrix, test_data, columns, index)
    # print(f'RMSE: {mse}')
    
    # Run interface
    run_interface(model_knn, user_movie_matrix, columns)

if __name__ == '__main__':
    main()
