import numpy as np
import pandas as pd
from nltk.stem.porter import *
import re
from bs4 import BeautifulSoup
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from stemming.porter2 import stem
from sklearn.svm import SVC
from sklearn.neighbors import DistanceMetric
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search

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

def gen_features_1(train, test):
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

    return X, X_test    

def gen_features_2(train, test):
    s_data = []
    s_labels = []
    t_data = []
    train_match_ct1 = []
    train_match_ct2 = []
    test_match_ct1 = []
    test_match_ct2 = []


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
        query = (" ").join([z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")])
        query = re.sub("[^a-zA-Z0-9]"," ", query)
        query = (" ").join([stemmer.stem(z) for z in query.split(" ")])
        title = (" ").join([z for z in BeautifulSoup(train.product_title[i]).get_text(" ").split(" ")])
        title = re.sub("[^a-zA-Z0-9]"," ", title)
        title = (" ").join([stemmer.stem(z) for z in title.split(" ")])
        description = BeautifulSoup(train.product_description[i]).get_text(" ")
        description = re.sub("[^a-zA-Z0-9]"," ", description)
        description = (" ").join([stemmer.stem(z) for z in description.split(" ")])

        match_cnt1 = matchCnt(query, title)
        match_cnt2 = matchCnt(query, description)
        train_match_ct1.append(match_cnt1)
        train_match_ct2.append(match_cnt2)

        s=(" ").join(["q"+ z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(train.product_title[i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(train.product_description[i]).get_text(" ")
        s=re.sub("[^a-zA-Z0-9]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
        s_data.append(s)
        s_labels.append(str(train["median_relevance"][i]))
    for i in range(len(test.id)):
        query = (" ").join([z for z in BeautifulSoup(test["query"][i]).get_text(" ").split(" ")])
        query = re.sub("[^a-zA-Z0-9]"," ", query)
        query = (" ").join([stemmer.stem(z) for z in query.split(" ")])
        title = (" ").join([z for z in BeautifulSoup(test.product_title[i]).get_text(" ").split(" ")])
        title = re.sub("[^a-zA-Z0-9]"," ", title)
        title = (" ").join([stemmer.stem(z) for z in title.split(" ")])
        description = BeautifulSoup(test.product_description[i]).get_text(" ")
        description = re.sub("[^a-zA-Z0-9]"," ", description)
        description = (" ").join([stemmer.stem(z) for z in description.split(" ")])

        match_cnt1 = matchCnt(query, title)
        match_cnt2 = matchCnt(query, description)
        test_match_ct1.append(match_cnt1)
        test_match_ct2.append(match_cnt2)

        s=(" ").join(["q"+ z for z in BeautifulSoup(test["query"][i]).get_text().split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(test.product_title[i]).get_text().split(" ")]) + " " + BeautifulSoup(test.product_description[i]).get_text()
        s=re.sub("[^a-zA-Z0-9]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
        t_data.append(s)
    
    # the infamous tfidf vectorizer (Do you remember this one?)
    tfv = TfidfVectorizer(
            min_df=5, max_df=500, max_features=None, 
            strip_accents='unicode', analyzer='word', 
            token_pattern=r'\w{1,}', ngram_range=(1, 2), 
            use_idf=True, smooth_idf=True, sublinear_tf=True, 
            stop_words = 'english')
    tfv.fit(s_data)
    X, X_test = tfv.transform(s_data), tfv.transform(t_data)

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

    return X, X_test

def gen_dist_features(train, test):
    # stemming
    train['query'] = train.apply(lambda x: stemming(x['query']), axis=1)
    train['product_title'] = train.apply(lambda x: stemming(x['product_title']), axis=1)
    # train['product_description'] = train.apply(lambda x: stemming(x['product_description']), axis=1)

    test['query'] = test.apply(lambda x: stemming(x['query']), axis=1)
    test['product_title'] = test.apply(lambda x: stemming(x['product_title']), axis=1)
    # test['product_description'] = test.apply(lambda x: stemming(x['product_description']), axis=1)

    
    # do some lambda magic on text columns
    traindata = list(train.apply(lambda x:'%s' % (x['query']),axis=1))
    traindata.extend(list(train.apply(lambda x:'%s' % (x['product_title']),axis=1)))
    testdata = list(test.apply(lambda x:'%s' % (x['query'],),axis=1))
    testdata.extend(list(test.apply(lambda x:'%s' % (x['product_title'],),axis=1)))

    # the infamous tfidf vectorizer (Do you remember this one?)
    tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 5), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
    
    # Fit TFIDF
    tfv.fit(traindata)
    X, X_test = tfv.transform(traindata), tfv.transform(testdata)

    # Initialize SVD
    svd = TruncatedSVD(n_components=200)

    # Initialize the standard scaler 
    scl = StandardScaler()
    
    preproc = pipeline.Pipeline([
        ('svd', svd),
        ('scl', scl)])
    preproc.fit(X)
    X =  preproc.transform(X).astype(np.float32)
    X_test = preproc.transform(X_test).astype(np.float32)

    dist1 = DistanceMetric.get_metric('euclidean')
    dist2 = DistanceMetric.get_metric('manhattan')
    dist3 = DistanceMetric.get_metric('chebyshev')

    train_feat1 = [dist1.pairwise([X[idx], X[idx+X.shape[0]/2]])[0,1] for idx in range(X.shape[0]/2)]    
    train_feat2 = [dist2.pairwise([X[idx], X[idx+X.shape[0]/2]])[0,1] for idx in range(X.shape[0]/2)]   
    train_feat3 = [dist3.pairwise([X[idx], X[idx+X.shape[0]/2]])[0,1] for idx in range(X.shape[0]/2)]

    test_feat1 = [dist1.pairwise([X_test[idx], X_test[idx+X_test.shape[0]/2]])[0,1] for idx in range(X_test.shape[0]/2)]
    test_feat2 = [dist2.pairwise([X_test[idx], X_test[idx+X_test.shape[0]/2]])[0,1] for idx in range(X_test.shape[0]/2)]
    test_feat3 = [dist3.pairwise([X_test[idx], X_test[idx+X_test.shape[0]/2]])[0,1] for idx in range(X_test.shape[0]/2)]

    train_feats = [train_feat1, train_feat2, train_feat3]
    test_feats = [test_feat1, test_feat2, test_feat3]
    return train_feats, test_feats



def get_best_model(X, y):

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
    param_grid = {'svd__n_components' : [200, 400, 600],
                  'svm__C': [8, 10, 12]}
    
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
    opt_model = model.best_estimator_
    
    # Fit model with best parameters optimized for quadratic_weighted_kappa
    opt_model.fit(X,y)
    
    return opt_model
