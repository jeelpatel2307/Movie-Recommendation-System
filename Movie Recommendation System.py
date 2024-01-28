# app.py

from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Sample movie data
movies_data = pd.DataFrame({
    'movie_id': [1, 2, 3, 4],
    'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D'],
    'genre': ['Action', 'Comedy', 'Action', 'Drama']
})

# Create a CountVectorizer to convert genre into a matrix of token counts
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(movies_data['genre'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

def get_recommendations(movie_title):
    """
    Get movie recommendations based on user input.

    Args:
    - movie_title (str): Title of the movie for which recommendations are requested.

    Returns:
    - list: List of recommended movie titles.
    """
    movie_index = movies_data[movies_data['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[movie_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:3]  # Get top 2 similar movies (excluding itself)
    movie_indices = [i[0] for i in sim_scores]
    return movies_data['title'].iloc[movie_indices].tolist()

@app.route('/')
def index():
    return render_template('index.html', movie_titles=movies_data['title'].tolist())

@app.route('/recommend', methods=['POST'])
def recommend():
    selected_movie = request.form.get('movie')
    recommendations = get_recommendations(selected_movie)
    return render_template('index.html', movie_titles=movies_data['title'].tolist(), selected_movie=selected_movie, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)






<!-- templates/index.html -->

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Movie Recommendation System</title>
</head>
<body>

    <h1>Movie Recommendation System</h1>

    <form action="/recommend" method="post">
        <label for="movie">Select a Movie:</label>
        <select id="movie" name="movie">
            {% for movie in movie_titles %}
                <option value="{{ movie }}">{{ movie }}</option>
            {% endfor %}
        </select>
        <input type="submit" value="Get Recommendations">
    </form>

    {% if selected_movie %}
        <h2>Recommended Movies for {{ selected_movie }}:</h2>
        <ul>
            {% for recommendation in recommendations %}
                <li>{{ recommendation }}</li>
            {% endfor %}
        </ul>
    {% endif %}

</body>
</html>
