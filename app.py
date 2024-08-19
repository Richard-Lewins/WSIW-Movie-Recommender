from flask import Flask, render_template, request, jsonify
from src.data_loader import load_data
from src.preprocess import merge_data, create_user_movie_matrix
from src.model import train_model_knn_movies, get_movie_recommendations_by_movie
import requests

app = Flask(__name__)

# Load and preprocess data
movies, ratings = load_data('data/movies.csv', 'data/ratings.csv')
data = merge_data(movies, ratings)
user_movie_matrix, movie_titles, index = create_user_movie_matrix(data)
model_knn = train_model_knn_movies(user_movie_matrix, n_neighbors=10)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('q')
    # Implement search logic and return matching movie titles
    results = [{'movieId': movie_id, 'title': title} for movie_id, title in enumerate(movie_titles) if query.lower() in title.lower()]
    print(results)
    return jsonify(results)

@app.route('/recommendations/<int:movie_id>')
def recommendations(movie_id):
    # Get recommendations based on the selected movie_id
    recommendations = get_movie_recommendations_by_movie(model_knn, user_movie_matrix, movie_titles[movie_id], movie_titles)

    # Fetch additional details from TMDB API if needed, will do later
    movie_recommendations = [{'title': title, 'poster_path': None} for title in recommendations]
    print(movie_recommendations)
    return render_template('recommendations.html', movies=movie_recommendations)

if __name__ == '__main__':
    app.run(debug=True)
