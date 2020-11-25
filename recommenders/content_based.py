"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import numpy as np
import pandas as pd
import scipy as sp # <-- The sister of Numpy, used in our code for numerical efficientcy.
import matplotlib.pyplot as plt
import seaborn as sns

# Entity featurization and similarity computation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Libraries used during sorting procedures.
import operator # <-- Convienient item retrieval during iteration
import heapq # <-- Efficient sorting of large lists

# Importing data
movies_url = 'https://raw.githubusercontent.com/Gireen-Naidu/unsupervised-predict-streamlit-template/master/resources/data/movies.csv'
movies = pd.read_csv(movies_url,sep = ',',delimiter=',')
movies_adj = movies.loc[:50000]

#ratings_url = 'https://raw.githubusercontent.com/Gireen-Naidu/unsupervised-predict-streamlit-template/master/resources/data/ratings.csv'
#ratings = pd.read_csv(ratings_url)

movies.dropna(inplace=True)


# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.
def content_model(movie_list,top_n=10):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    # Initializing the empty list of recommended movies
    movies_adj['keyWords'] = movies_adj['genres'].str.replace('|', ' ')
    movies_adj['tags'] = movies_adj[['title', 'keyWords']].agg(' '.join, axis=1)

    movies_adj.dropna(inplace=True)

    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2),
                     min_df=1, stop_words='english')

    # Produce a feature matrix, where each row corresponds to a book,
    # with TF-IDF features as columns
    tf_authTags_matrix = tf.fit_transform(movies_adj['tags'])

    cosine_sim_authTags = cosine_similarity(tf_authTags_matrix,tf_authTags_matrix)

    # Convienient indexes to between map book titles and indexes of
    # the books dataframe
    titles = movies_adj['title']
    indices = pd.Series(movies_adj['title'])
    #indices = pd.Series(movies_adj.index, index=movies_adj['title'])
    #idx_1 = indices[indices == movie_list[0]]
    #return idx_1
    # Convert the string book title to a numeric index for our
    # similarity matrix
    #b_idx = indices[movie_title]
    idx_1 = indices[indices == movie_list[0]].index[0]
    idx_2 = indices[indices == movie_list[1]].index[0]
    idx_3 = indices[indices == movie_list[2]].index[0]
    rank_1 = cosine_sim_authTags[idx_1]
    rank_2 = cosine_sim_authTags[idx_2]
    rank_3 = cosine_sim_authTags[idx_3]
    score_series_1 = pd.Series(rank_1).sort_values(ascending = False)
    score_series_2 = pd.Series(rank_2).sort_values(ascending = False)
    score_series_3 = pd.Series(rank_3).sort_values(ascending = False)
    # Getting the indexes of the 10 most similar movies
    listings = score_series_1.append(score_series_1).append(score_series_3).sort_values(ascending = False)
    # Store movie names
    recommended_movies = []
    # Appending the names of movies
    top_50_indexes = list(listings.iloc[1:50].index)
   # Removing chosen movies
    top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
    for i in top_indexes[:top_n]:
        recommended_movies.append(list(movies['title'])[i])
    return recommended_movies
