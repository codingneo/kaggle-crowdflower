
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
from nltk.stem.porter import *
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from stemming.porter2 import stem
import re
from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search

from utils import get_best_model

sw=[]
s_data = []
s_labels = []
t_data = []
t_labels = []
#stopwords tweak - more overhead
stop_words = ['http','www','img','border','0','1','2','3','4','5','6','7','8','9']
stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)
for stw in stop_words:
    sw.append("q"+stw)
    sw.append("z"+stw)
stop_words = text.ENGLISH_STOP_WORDS.union(sw)


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

    #load data
    train = pd.read_csv("./data/train.csv").fillna("")
    test  = pd.read_csv("./data/test.csv").fillna("")

    idx = test.id.values.astype(int)
    
    #remove html, remove non text or numeric, make query and title unique features for counts using prefix (accounted for in stopwords tweak)
    stemmer = PorterStemmer()
    ## Stemming functionality
    class stemmerUtility(object):
        """Stemming functionality"""
        @staticmethod
        def stemPorter(review_text):
            porter = PorterStemmer()
            preprocessed_docs = []
            for doc in review_text:
                final_doc = []
                for word in doc:
                    final_doc.append(porter.stem(word))
                    #final_doc.append(wordnet.lemmatize(word)) #note that lemmatize() can also takes part of speech as an argument!
                preprocessed_docs.append(final_doc)
            return preprocessed_docs
    
    
    for i in range(len(train.id)):
        s=(" ").join(["q"+ z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(train.product_title[i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(train.product_description[i]).get_text(" ")
        s=re.sub("[^a-zA-Z0-9]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
        s_data.append(s)
        s_labels.append(str(train["median_relevance"][i]))
    for i in range(len(test.id)):
        s=(" ").join(["q"+ z for z in BeautifulSoup(test["query"][i]).get_text().split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(test.product_title[i]).get_text().split(" ")]) + " " + BeautifulSoup(test.product_description[i]).get_text()
        s=re.sub("[^a-zA-Z0-9]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
        t_data.append(s)
    
    tfv = TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')
    tfv.fit(s_data)
    X, X_test = tfv.transform(s_data), tfv.transform(t_data)

    #create sklearn pipeline, fit all, and predit test data
    best_model = pipeline.Pipeline([ 
        ('svd', TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)), 
        ('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), 
        ('svm', SVC(C=10.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None))])
    
    best_model.fit(X, s_labels)
    preds = best_model.predict(X_test)

    # submission = pd.DataFrame({"id": idx, "prediction": preds})
    # submission.to_csv("./submission/svc.csv", index=False)

    # We will use Random Forest here..
    rf = RandomForestClassifier(n_estimators=1000)

    # We will use Extremely Randomized Trees here..
    etc = ExtraTreesClassifier(n_estimators=500)    

    # We will use Random Forest here..
    # gbm = GradientBoostingClassifier(n_estimators=500)

    # Fit Naive Bayes Model
    nbmodel = MultinomialNB(alpha=.003)

    # Fit Logistical Regression
    lr = LogisticRegression(C=1.8)

    # Create the pipeline 
    # Initialize SVD
    best_svd_rf = TruncatedSVD(n_components=200)
    best_svd_lr = TruncatedSVD(n_components=600)
    best_svd_etc = TruncatedSVD(n_components=200)
    scl_rf = StandardScaler()
    scl_lr = StandardScaler()
    scl_etc = StandardScaler()
    meta_clf1 = pipeline.Pipeline([('svd', best_svd_rf),
                                  ('scl', scl_rf),
                                  ('rf', rf)])

    meta_clf2 = pipeline.Pipeline([('svd', best_svd_etc),
                                  ('scl', scl_etc),
                                  ('etc', etc)])

    # begging
    num_bagging = 10

    for bag_idx in range(num_bagging):
        print("Processing " + str(bag_idx) + " bagging models ...")
        random_indices = np.random.permutation(train.shape[0])

        X1 = X[random_indices[:ceil(train.shape[0]/2)]]
        y1 = list(s_labels[idx] for idx in random_indices[:ceil(train.shape[0]/2)])
        X2 = X[random_indices[ceil(train.shape[0]/2):]]
        y2 = list(s_labels[idx] for idx in random_indices[ceil(train.shape[0]/2):])

        meta_clf1.fit(X1, y1)
        pred_feat1 = meta_clf1.predict_proba(X2)
        nbmodel.fit(X1, y1)
        pred_feat2 = nbmodel.predict_proba(X2)
        meta_clf2.fit(X1, y1)
        pred_feat3 = meta_clf2.predict_proba(X2)


        # new_data = hstack([X2, csr_matrix(pred_feat1)])
        new_data = hstack([X2, csr_matrix(pred_feat1), csr_matrix(pred_feat2)])
        # new_data = hstack([X2, csr_matrix(pred_feat1), csr_matrix(pred_feat2), csr_matrix(pred_feat3)])
        
        best_model.fit(new_data, y2)
        # best_model_2 = get_best_model(new_data, y2)

        # Make prediction
        test_pred_feat1 = meta_clf1.predict_proba(X_test)
        test_pred_feat2 = nbmodel.predict_proba(X_test)
        test_pred_feat3 = meta_clf2.predict_proba(X_test)

        # new_X_test = hstack([X_test, csr_matrix(test_pred_feat1)])
        new_X_test = hstack([X_test, csr_matrix(test_pred_feat1), csr_matrix(test_pred_feat2)])
        # new_X_test = hstack([X_test, csr_matrix(test_pred_feat1), csr_matrix(test_pred_feat2), csr_matrix(test_pred_feat3)])

        meta_preds = best_model.predict(new_X_test)
        new_preds = []
        for i in range(len(preds)):
            x = (int(meta_preds[i]) + int(preds[i]))
            new_preds.append(int(x))

        preds = new_preds
        # print(preds/(bag_idx+1))    

    preds = list(int(x/(num_bagging+1)) for x in preds)
    
    # Create your first submission file
    submission = pd.DataFrame({"id": idx, "prediction": preds})
    submission.to_csv("./submission/meta-bagging-2.csv", index=False)
