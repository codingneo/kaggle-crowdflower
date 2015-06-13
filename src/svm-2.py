
"""
Beating the Benchmark 
Search Results Relevance @ Kaggle
__author__ : Abhishek

"""
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from scipy.sparse import coo_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search

from utils import gen_features

# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)

def matchCnt(query, txt):
    num_matched = 0
    for word in query.split(' '):
        if word.lower() in txt.lower():
            num_matched += 1
            
    return num_matched/len(query.split(' '))

if __name__ == '__main__':

    # Load the training file
    train = pd.read_csv("./data/train.csv").fillna("")
    test  = pd.read_csv("./data/test.csv").fillna("")
    
    # we dont need ID columns
    idx = test.id.values.astype(int)
    X, X_test = gen_features(train, test)

    train = train.drop('id', axis=1)
    test = test.drop('id', axis=1)
    
    # create labels. drop useless columns
    y = train.median_relevance.values
    train = train.drop(['median_relevance', 'relevance_variance'], axis=1)

    train_match_ct1 = list(train.apply(lambda x: matchCnt(x['query'], x['product_title']),axis=1))
    train_match_ct2 = list(train.apply(lambda x: matchCnt(x['query'], x['product_description']),axis=1))
    test_match_ct1 = list(test.apply(lambda x: matchCnt(x['query'], x['product_title']),axis=1))
    test_match_ct2 = list(test.apply(lambda x: matchCnt(x['query'], x['product_description']),axis=1))

    ridx = range(train.shape[0])
    cidx = [0] * train.shape[0]
    X = hstack([X, csr_matrix((train_match_ct1, (ridx, cidx)), shape=(train.shape[0],1))])
    X = hstack([X, csr_matrix((train_match_ct2, (ridx, cidx)), shape=(train.shape[0],1))])
    # X = hstack([X, csr_matrix((train_query_cnt, (ridx, cidx)), shape=(train.shape[0],1))])
    # X = hstack([X, csr_matrix((train_title_cnt, (ridx, cidx)), shape=(train.shape[0],1))])
    # X = hstack([X, csr_matrix((train_description_cnt, (ridx, cidx)), shape=(train.shape[0],1))])
    X = csr_matrix(X)
    test_ridx = range(test.shape[0])
    test_cidx = [0] * test.shape[0]
    X_test = hstack([X_test, csr_matrix((test_match_ct1, (test_ridx, test_cidx)), shape=(test.shape[0],1))])
    X_test = hstack([X_test, csr_matrix((test_match_ct2, (test_ridx, test_cidx)), shape=(test.shape[0],1))])
    # X_test = hstack([X_test, csr_matrix((test_query_cnt, (test_ridx, test_cidx)), shape=(test.shape[0],1))])
    # X_test = hstack([X_test, csr_matrix((test_title_cnt, (test_ridx, test_cidx)), shape=(test.shape[0],1))])
    # X_test = hstack([X_test, csr_matrix((test_description_cnt, (test_ridx, test_cidx)), shape=(test.shape[0],1))])
    
    # Initialize SVD
    svd = TruncatedSVD()

    # Topic modeling with NMF
    nmf = NMF(random_state=1)
    
    # Initialize the standard scaler 
    scl = StandardScaler()
    
    # We will use SVM here..
    svm_model = SVC(kernel='linear')
    
    # Create the pipeline 
    #create sklearn pipeline, fit all, and predit test data
    best_model = pipeline.Pipeline([ 
        ('svd', TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)), 
        ('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), 
        ('svm', SVC(C=10.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None))])
        
    # Fit model with best parameters optimized for quadratic_weighted_kappa
    best_model.fit(X,y)
    preds = best_model.predict(X_test)
    
    # Create your first submission file
    submission = pd.DataFrame({"id": idx, "prediction": preds})
    submission.to_csv("./submission/svm-2.csv", index=False)
