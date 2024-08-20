from src.model import get_movie_recommendations_by_movie
from src.preprocess import find_closest_movies



# TODO: Change this to be web based interface
def run_interface(model_knn, user_movie_matrix, movie_titles):
    while True:
        user_input = input("Enter a movie title or 'exit' to quit: ")
        if user_input.lower() == 'exit':
            break
        
        # Find the closest movie titles
        matches = find_closest_movies(user_input, movie_titles)
        print(matches)
        
        #Ask "Did you mean?". If the user enters "yes", then use the first match as the movie title.
        if matches:
            print(f"Did you mean: {matches[0]}?")
            user_input = input("Enter 'yes' or 'no': ")
            if user_input.lower() == 'yes':
                user_input = matches[0]
            else:
                print("Movie not found in the database.")
                continue

        if user_input in movie_titles:
            similar_movies = get_movie_recommendations_by_movie(model_knn, user_movie_matrix, user_input, movie_titles)
            print(f"Movies similar to '{user_input}':")
            for movie in similar_movies:
                print(movie)
        else:
            print("Movie not found in the database.")

