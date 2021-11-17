
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.linear_model as lm
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import plot_confusion_matrix

def score_class_models(models, X_train, y_train, X_test, y_test):
    '''
    Tests a list of predictive models and returns accuracy and f1 scores for each model.

    Parameter
    ----------
    models:  list 
        A list of models to test.
    X_train:  arr
        X_train data.
    y_train:  arr
        y_train data.
    X_test:  arr
        X_test data.
    y_test:  arr
        y_test data.
        
    Returns
    ----------
    Accuracy and f1 scores for each model in the list of models passed in.
    '''
    acc_score_list = []
    f1_score_list = []

    for model in models:   
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc_score_list.append(model.score(X_test, y_test))
        f1_score_list.append(f1_score(y_test, y_pred, average='weighted'))
        
    for model, score, f1_scored in zip(models, acc_score_list, f1_score_list):
        print(f'{model} accuracy: {round(score * 100, 2)} %')
        print(f'{model} f1: {round(f1_scored * 100, 2)} %')     


def conf_matrix(model, X_train, y_train):
    '''
    Returns a confusion matrix for the model.

    Parameter
    ----------
    model: object
        A predictive model
    X_train: arr
        X_train data
    y_train: arr
        y_train data
        
    Returns
    ----------
    A confusion matrix for the model passed in.
    '''

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)


def stratified_k_fold(model, X_train, y_train, X_test, y_test, n_folds=5):
    '''
    Performs stratified K-fold cross validation and returns mean of scores and best score.

    Parameters
    ----------
    model: 
        Model to test.
    X_train:  arr
        X_train data.
    y_train:  arr
        y_train data.
    X_test:  arr
        X_test data.
    y_test:  arr
        y_test data.

    Returns
    ----------
    Mean of scores.  Score of the best model.
    '''

    skf = StratifiedKFold(n_splits=n_folds)
    scores = []
    models = []
    
    for train_index, test_index in skf.split(X_train, y_train):
        X_t = X_train[train_index]
        X_v = X_train[test_index]
        y_t = y_train[train_index]
        y_v = y_train[test_index]

        model.fit(X_t, y_t)
        models.append(model)
        scores.append(model.score(X_v, y_v))

    scores = np.array(scores)
    scores_mean = scores.mean()
    best_model = models[np.argmax(scores)]
    val_score = best_model.score(X_test, y_test)

    return f'scores mean: {scores_mean} \n val score: {val_score}'
