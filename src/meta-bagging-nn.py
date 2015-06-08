
"""
Beating the Benchmark 
Search Results Relevance @ Kaggle
__author__ : Abhishek

"""
from __future__ import division

from math import ceil
import re
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from stemming.porter2 import stem
# from bs4 import BeautifulSoup
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search
import theano
from lasagne import layers
from lasagne.layers import DenseLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import InputLayer
from lasagne.nonlinearities import identity,sigmoid, tanh,rectify
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum,adagrad
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator

from utils import get_best_model

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)
class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()


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
    if (pd.isnull(txt)):
        return 0
    else:
        for word in query.split(' '):
            if word.lower() in txt.lower():
                num_matched += 1
            
        return num_matched/len(query.split(' '))

def countWords(txt):
    if (pd.isnull(txt)):
        return 0
    else:
        return len(txt.lower().split(' '))

def stemming(txt):
    s = (" ").join(stem(x) for x in txt.split(" "))
    # s = (" ").join(["q"+ z for z in txt.split(" ")])
    # s=re.sub("[^a-zA-Z0-9]"," ", s)
    # s= (" ").join([stemmer.stem(z) for z in s.split(" ")])

    # import string
    # exclude = set(string.punctuation)
    # s = ''.join(ch for ch in s if ch not in exclude)

    return s

if __name__ == '__main__':

    # Load the training file
    train = pd.read_csv('./data/train.csv').fillna("")
    test = pd.read_csv('./data/test.csv').fillna("")
    
    # we dont need ID columns
    idx = test.id.values.astype(int)
    train = train.drop('id', axis=1)
    test = test.drop('id', axis=1)
    
    # create labels. drop useless columns
    y = train.median_relevance.values
    train = train.drop(['median_relevance', 'relevance_variance'], axis=1)

    # stemming
    train['query'] = train.apply(lambda x: stemming(x['query']), axis=1)
    train['product_title'] = train.apply(lambda x: stemming(x['product_title']), axis=1)
    # train['product_description'] = train.apply(lambda x: stemming(x['product_description']), axis=1)

    test['query'] = test.apply(lambda x: stemming(x['query']), axis=1)
    test['product_title'] = test.apply(lambda x: stemming(x['product_title']), axis=1)
    # test['product_description'] = test.apply(lambda x: stemming(x['product_description']), axis=1)

    
    # do some lambda magic on text columns
    traindata = list(train.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
    testdata = list(test.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))

    train_match_ct1 = list(train.apply(lambda x: matchCnt(x['query'], x['product_title']),axis=1))
    train_match_ct2 = list(train.apply(lambda x: matchCnt(x['query'], x['product_description']),axis=1))
    test_match_ct1 = list(test.apply(lambda x: matchCnt(x['query'], x['product_title']),axis=1))
    test_match_ct2 = list(test.apply(lambda x: matchCnt(x['query'], x['product_description']),axis=1))

    # train_query_cnt = list(train.apply(lambda x: countWords(x['query']),axis=1))
    # train_title_cnt = list(train.apply(lambda x: countWords(x['product_title']),axis=1))
    # train_description_cnt = list(train.apply(lambda x: countWords(x['product_description']),axis=1))
    # test_query_cnt = list(test.apply(lambda x: countWords(x['query']),axis=1))
    # test_title_cnt = list(test.apply(lambda x: countWords(x['product_title']),axis=1))
    # test_description_cnt = list(test.apply(lambda x: countWords(x['product_description']),axis=1))

    
    # the infamous tfidf vectorizer (Do you remember this one?)
    tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 5), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
    
    # Fit TFIDF
    tfv.fit(traindata)
    X, X_test = tfv.transform(traindata), tfv.transform(testdata)

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
   
    # Initialize the standard scaler 
    scl = StandardScaler()
    
    # We will use SVM here..
    svm_model = SVC()
    
    # Create the pipeline 
    clf = pipeline.Pipeline([('svd', svd),
                             ('scl', scl),
                             ('svm', svm_model)])
    
    # Create a parameter grid to search for best parameters for everything in the pipeline
    param_grid = {'svd__n_components' : [200, 400],
                  'svm__C': [10, 12]}
    
    # Kappa Scorer 
    kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better = True)
    
    # Initialize Grid Search Model
    model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, scoring=kappa_scorer,
                                     verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)
                                     
    # Fit Grid Search Model
    model.fit(X, y)
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
    # Get best model
    best_model = model.best_estimator_
    
    # Fit model with best parameters optimized for quadratic_weighted_kappa
    best_model.fit(X,y)
    preds = best_model.predict(X_test)

    # submission = pd.DataFrame({"id": idx, "prediction": preds})
    # submission.to_csv("./submission/svc.csv", index=False)

    # We fit neural network
    nn= NeuralNet(
        layers=[  # three layers: one hidden layer
            ('input', layers.InputLayer),
            ('hidden1', layers.DenseLayer),
            ('dropout1', layers.DropoutLayer),
            ('hidden2', layers.DenseLayer),
            ('dropout2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
        ],
    
        # layer parameters:
        input_shape=(None,X.shape[1]),
        hidden1_num_units=512,  # number of units in hidden layer
        dropout1_p=0.5,
        hidden2_num_units=256,  # number of units in hidden layer
        hidden2_nonlinearity=rectify,
        dropout2_p=0.4,

        output_nonlinearity=softmax,  # output layer uses identity function
        output_num_units=5,  # target values

        # optimization method:
        update=adagrad,

        update_learning_rate=theano.shared(np.float32(0.1)),


        on_epoch_finished=[
            AdjustVariable('update_learning_rate', start=0.1, stop=0.0001),

            EarlyStopping(patience=10),
        ],
        use_label_encoder=False,

        batch_iterator_train=BatchIterator(batch_size=100),
        regression=False,  # flag to indicate we're dealing with regression problem
        max_epochs=100,  # we want to train this many epochs
        verbose=1,
        eval_size=0.1

    )


    # Create the pipeline 
    # Initialize SVD
    best_svd_nn = TruncatedSVD(n_components=200)
    scl_nn = StandardScaler()
    meta_clf1 = pipeline.Pipeline([('svd', best_svd_nn),
                                  ('scl', scl_nn),
                                  ('nn', nn)])

    # begging
    num_bagging = 20

    for bag_idx in range(num_bagging):
        print("Processing " + str(bag_idx) + " bagging models ...")
        random_indices = np.random.permutation(train.shape[0])

        X1 = X[random_indices[:ceil(train.shape[0]/2)]]
        y1 = y[random_indices[:ceil(train.shape[0]/2)]]
        X2 = X[random_indices[ceil(train.shape[0]/2):]]
        y2 = y[random_indices[ceil(train.shape[0]/2):]]

        X1, y1 = shuffle(X1, y1, random_state=7)
        meta_clf1.fit(X1, y1)
        pred_feat1 = meta_clf1.predict_proba(X2)

        new_data = hstack([X2, csr_matrix(pred_feat1)])
        # new_data = hstack([X2, csr_matrix(pred_feat1), csr_matrix(pred_feat2)])
        # new_data = hstack([X2, csr_matrix(pred_feat1), csr_matrix(pred_feat2), csr_matrix(pred_feat3)])
        
        best_model.fit(new_data, y2)
        # best_model_2 = get_best_model(new_data, y2)

        # Make prediction
        test_pred_feat1 = meta_clf1.predict_proba(X_test)
        # test_pred_feat2 = nbmodel.predict_proba(X_test)
        # test_pred_feat3 = meta_clf2.predict_proba(X_test)

        new_X_test = hstack([X_test, csr_matrix(test_pred_feat1)])
        # new_X_test = hstack([X_test, csr_matrix(test_pred_feat1), csr_matrix(test_pred_feat2)])
        # new_X_test = hstack([X_test, csr_matrix(test_pred_feat1), csr_matrix(test_pred_feat2), csr_matrix(test_pred_feat3)])

        preds += best_model.predict(new_X_test)

        print(preds/(bag_idx+1))    

    preds = np.round(preds/(num_bagging+1))
    preds = preds.astype(int)
    
    # Create your first submission file
    submission = pd.DataFrame({"id": idx, "prediction": preds})
    submission.to_csv("./submission/meta-bagging-nn.csv", index=False)
