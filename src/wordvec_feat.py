
"""
Beating the Benchmark 
Search Results Relevance @ Kaggle
__author__ : Abhishek

"""
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

def min_distances(query, text, k, word_vecs_file):
    query_words = set(query.lower().rstrip().split(' '))
    text_words = set(text.lower().rstrip().split(' '))

    query_vectors = []
    text_vectors = []
    file = open(word_vecs_file, "r")
    for line in file:
        data = line.rstrip().split(' ')
        if data[0] in query_words:
            query_vectors.append(data[1:])
        if data[0] in text_words:
            text_vectors.append(data[1:])

    X_query = np.array(query_vectors).astype(float)
    X = np.array(text_vectors).astype(float)

    min_dists = []
    if (X_query.shape[0]>0 and X.shape[0]>0):
        n = min(X.shape[0], k)
        nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree', metric=cosine_similarity).fit(X)
        distances, _ = nbrs.kneighbors(X_query)

        sorted_distances = np.sort(distances, axis=None)
        if (len(sorted_distances)>k):
            min_dists = sorted_distances[:k]
        else:
            min_dists = sorted_distances

    if (len(min_dists)<k):
        min_dists = np.append(min_dists, np.zeros(k-len(min_dists)))

    return min_dists

if __name__ == '__main__':

    # Load the training file
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')

    # Load the semantic vectors
    word_vecs_file = './data/glove.6B.300d.txt'
        
    print('Computing minimum distances from query to product_title ...')
    train_min_distances1 = list(train.apply(lambda x: min_distances(x['query'], x['product_title'], 3, word_vecs_file), axis=1))
    # train_min_distances2 = list(train.apply(lambda x: min_distances(x['query'], x['product_description'], 5, word_vecs_file), axis=1))    
    # test_min_distances1 = list(test.apply(lambda x: min_distances(x['query'], x['product_title'], 3, word_vecs_file), axis=1))
    # test_min_distances2 = list(test.apply(lambda x: min_distances(x['query'], x['product_description'], 5, word_vecs_file), axis=1))   
    
    # save file
    pickle.dump(train_min_distances1, open( "./data/train_min_distances1.p", "wb" ) )
