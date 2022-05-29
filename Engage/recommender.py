from flask import Flask, render_template
from flask import request
import os

import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

TEMPLATE_DIR = os.path.abspath(r'templates')
STATIC_DIR = os.path.abspath(r'static')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
@app.route('/')
def home():
    print(request.args)
    return render_template('index.html')

@app.route("/submit", methods=['POST','GET'])
def submit():
    print("started")
    if(request.method=='POST'):
        your_movie_name = str(request.form['mv_name'])
        your_movie_rating = int(request.form['R'])

    ratings = pd.read_csv("datasets/ratings_new.csv")
    movies = pd.read_csv("datasets/movies_new.csv")

    ratings = pd.merge(movies, ratings).drop(['genres', 'timestamp'], axis=1)

    print("done reading csv files")
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(movies["genres"])
    cosine_sim = cosine_similarity(count_matrix)
    cosine_sim_df = pd.DataFrame(cosine_sim)
    cosine_sim_df.head()

    print("content df made")
    # collaborative

    user_ratings = ratings.pivot_table(index=['userId'], columns=['Index'], values='rating')
    user_ratings = user_ratings.fillna(0)
    item_similarity_df = user_ratings.corr(method='pearson')
    #similarity_score = item_similarity_df[0] * (5 - 2.5)
    #print(similarity_score)

    print("collaborative df made)")

    user = [(your_movie_name, your_movie_rating)]

    similar_movies = pd.DataFrame()

    for movie, rating in user:
        similar_movies = similar_movies.append(get_similar_movies(movie, rating, item_similarity_df, cosine_sim_df, movies), ignore_index=True)

    sim_movies = pd.DataFrame()

    # print(similar_movies)
    sim_movies = similar_movies.sum().sort_values(ascending=False)
    # sim_movies
    # get_title_from_Index(sim_movies.head(10))
    sim_movies = sim_movies.reset_index()
    # print(sim_movies)
    sim_movies.head()

    print("adding req movies to list")

    my_dict = {"titles": []}

    i = 0
    for element in sim_movies['index']:
        my_movie = get_title_from_Index(element, movies)
        my_dict["titles"].append(my_movie)
        i = i + 1
        if i > 10:
            break

    df = pd.DataFrame(data=my_dict)
    df = df.fillna(' ')

    return(df.to_html(justify='center'))

def get_title_from_Index(Index, movies):
    return movies[movies.Index == Index]["title"].values[0]

def get_Index_from_title(title, movies):
    print(movies[movies.title == title]["Index"])
    return movies[movies.title == title]["Index"].values[0]

def get_similar_movies(movie_name, user_rating,item_similarity_df, cosine_sim_df, movies):
    print("getting similar movies")
    similarity_score1 = item_similarity_df[get_Index_from_title(movie_name, movies)] * (user_rating - 2.5)

    similarity_score2 = cosine_sim_df[get_Index_from_title(movie_name, movies)]

    final_similarity_score = (similarity_score1 + similarity_score2) / 2

    similar_score = final_similarity_score.sort_values(ascending=False)
    return similar_score

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)