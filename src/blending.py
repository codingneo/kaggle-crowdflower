
"""
Beating the Benchmark 
Search Results Relevance @ Kaggle
__author__ : Abhishek

"""
from __future__ import division

from math import ceil
import pandas as pd
import numpy as np
from scipy.sparse import hstack, vstack
from scipy.sparse import csr_matrix
from stemming.porter2 import stem
# from bs4 import BeautifulSoup
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier, ExtraTreesClassifier
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

from utils import quadratic_weighted_kappa, gen_features_1, gen_features_2


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



if __name__ == '__main__':

    # Load the training file
    train = pd.read_csv('./data/train.csv').fillna("")
    test = pd.read_csv('./data/test.csv').fillna("")
    
    # we dont need ID columns
    idx = test.id.values.astype(int)
    X1, X_test_1 = gen_features_1(train, test)
    X2, X_test_2 = gen_features_2(train, test)

    X = csr_matrix(hstack([X1, X2]))
    X_test = csr_matrix(hstack([X_test_1, X_test_2]))

    train = train.drop('id', axis=1)
    test = test.drop('id', axis=1)
    
    # create labels. drop useless columns
    y=  train['median_relevance'].values.astype(np.int32)
    train = train.drop(['median_relevance', 'relevance_variance'], axis=1)

    # Initialize SVD
    svd = TruncatedSVD(
        n_components=200)
   
    # Initialize the standard scaler 
    scl = StandardScaler(
        copy=True, 
        with_mean=True, 
        with_std=True)
    
    # We will use SVM here..
    svm_clf = SVC(
        kernel='rbf', gamma=0.0,
        coef0=0.0, shrinking=True, 
        probability=True, tol=0.001, 
        cache_size=200, class_weight=None,
        verbose=False, max_iter=-1, 
        random_state=None)
    
    # Create the pipeline 
    svm = pipeline.Pipeline([('svd', svd),
                             ('scl', scl),
                             ('svm', svm_clf)])
    
    # submission = pd.DataFrame({"id": idx, "prediction": preds})
    # submission.to_csv("./submission/svc.csv", index=False)

   # We will use Random Forest here..
    gbm_clf = GradientBoostingClassifier(
        n_estimators=500,
        max_depth=11,
        learning_rate=0.1)

    # Create the pipeline 
    gbm = pipeline.Pipeline([('svd', svd),
                             ('scl', scl),
                             ('gbm', gbm_clf)])


    # Fit Naive Bayes Model
    nb = MultinomialNB(alpha=.003)

    # We will use Random Forest here..
    rf_clf = RandomForestClassifier(n_estimators=1000)
    rf = pipeline.Pipeline([('svd', svd),
                            ('scl', scl),
                            ('rf', rf_clf)])


    # We will use Extremely Randomized Trees here..
    et_clf = ExtraTreesClassifier(n_estimators=500)    
    etc = pipeline.Pipeline([('svd', svd),
                             ('scl', scl),
                             ('etc', et_clf)])

    # 
    lr_clf = LogisticRegression(C=1.8)
    # Create the pipeline 
    lr = pipeline.Pipeline([('svd', svd),
                             ('scl', scl),
                             ('lr', lr_clf)])
 
    # begging
    num_bagging = 10
    preds = np.zeros(test.shape[0])

    for bag_idx in range(num_bagging):
        print("Processing " + str(bag_idx) + " bagging models ...")
        random_indices = np.random.permutation(train.shape[0])

        X1 = X[random_indices[:ceil(train.shape[0]/2)]]
        y1 = y[random_indices[:ceil(train.shape[0]/2)]]
        X2 = X[random_indices[ceil(train.shape[0]/2):]]
        y2 = y[random_indices[ceil(train.shape[0]/2):]]

        # layer-1 prediction
        print("svm model ...")
        svm.fit(X1, y1)
        feat2 = svm.predict_proba(X2)
        test_feat2 = svm.predict_proba(X_test)
        svm.fit(X2, y2)
        feat1 = svm.predict_proba(X1)
        test_feat1 = svm.predict_proba(X_test)
        svm_feat = np.vstack((feat1, feat2))
        svm_test_feat = (test_feat1+test_feat2)/2.0

        # print("gbm model ...")
        # gbm.fit(X1, y1)
        # feat2 = gbm.predict_proba(X2)
        # test_feat2 = gbm.predict_proba(X_test)
        # gbm.fit(X2, y2)
        # feat1 = gbm.predict_proba(X1)
        # test_feat1 = gbm.predict_proba(X_test)
        # gbm_feat = np.vstack((feat1, feat2))
        # gbm_test_feat = (test_feat1+test_feat2)/2.0

        # print("rf model ...")
        rf.fit(X1, y1)
        feat2 = rf.predict_proba(X2)
        test_feat2 = rf.predict_proba(X_test)
        rf.fit(X2, y2)
        feat1 = rf.predict_proba(X1)
        test_feat1 = rf.predict_proba(X_test)
        rf_feat = np.vstack((feat1, feat2))
        rf_test_feat = (test_feat1+test_feat2)/2.0

        print("nb model ...")
        nb.fit(X1, y1)
        feat2 = nb.predict_proba(X2)
        test_feat2 = nb.predict_proba(X_test)
        nb.fit(X2, y2)
        feat1 = nb.predict_proba(X1)
        test_feat1 = nb.predict_proba(X_test)
        nb_feat = np.vstack((feat1, feat2))
        nb_test_feat = (test_feat1+test_feat2)/2.0

        print("etc model ...")
        etc.fit(X1, y1)
        feat2 = etc.predict_proba(X2)
        test_feat2 = etc.predict_proba(X_test)
        etc.fit(X2, y2)
        feat1 = etc.predict_proba(X1)
        test_feat1 = etc.predict_proba(X_test)
        etc_feat = np.vstack((feat1, feat2))
        etc_test_feat = (test_feat1+test_feat2)/2.0

        print("lr model ...")
        lr.fit(X1, y1)
        feat2 = lr.predict_proba(X2)
        test_feat2 = lr.predict_proba(X_test)
        lr.fit(X2, y2)
        feat1 = lr.predict_proba(X1)
        test_feat1 = lr.predict_proba(X_test)
        lr_feat = np.vstack((feat1, feat2))
        lr_test_feat = (test_feat1+test_feat2)/2.0

        new_data = hstack([
                    vstack([X1,X2]), 
                    csr_matrix(svm_feat), 
                    # csr_matrix(gbm_feat), 
                    csr_matrix(rf_feat),
                    csr_matrix(nb_feat),
                    csr_matrix(etc_feat),
                    csr_matrix(lr_feat)])

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
            input_shape=(None,200),
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
        meta_clf1.fit(new_data, np.concatenate((y1, y2)))

        new_test_data = hstack([
            X_test, 
            csr_matrix(svm_test_feat), 
            # csr_matrix(gbm_test_feat), 
            csr_matrix(rf_test_feat),
            csr_matrix(nb_test_feat),
            csr_matrix(etc_test_feat),
            csr_matrix(lr_test_feat)])

        preds += meta_clf1.predict(new_test_data)

        print(preds/(bag_idx+1))    

    preds = np.round(preds/(num_bagging+1))
    preds = preds.astype(int)
    
    # Create your first submission file
    submission = pd.DataFrame({"id": idx, "prediction": preds})
    submission.to_csv("./submission/blending.csv", index=False)
