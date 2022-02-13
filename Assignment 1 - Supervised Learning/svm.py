import numpy as np


import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV, validation_curve
from sklearn.preprocessing import StandardScaler
from util import *
from sklearn.svm import SVC

def run_experiment(data, dataset):
    learner="svm"
    # Scale and split Data
    X, y = data
    maxabs_scaler = StandardScaler()

    X = maxabs_scaler.fit_transform(X, y)

    split = StratifiedShuffleSplit(n_splits=1 , test_size=0.2, random_state=42)

    for train_ind, test_ind in split.split(X, y):
        X_train, y_train = X[train_ind], y[train_ind]
        X_test, y_test = X[test_ind], y[test_ind]
        
    # Test with Base Model
    print("----- Test Base Model -----")
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    model_type="base"

    results = testClassifier(X, y, svm_model, dataset, learner, model_type)
    plotResults(results, "Base Model", dataset, learner)

    # Test as a linear model
    print("----- Best Linear Model -----")
    param_grid = {'C': [0.0001, 0.001, 0.1, 1, 10, 100, 1000]}

    # svm_model = SVC(kernel='linear')
    # svm_clf = GridSearchCV(svm_model, param_grid, n_jobs=-1)

    # svm_clf.fit(X_train, y_train)

    # best_clf = svm_clf.best_estimator_
    # print(svm_clf.best_params_)

    # output = ""

    # for param in param_grid.keys():
    #     output += param + ": " + str(svm_clf.best_params_[param]) + "\n"

    # output_path = "./images/" + dataset + "/" + learner + "/optimal_linear_params.txt"
    # output_file = open(output_path, "w")
    # output_file.write(output)
    # output_file.close()

    model_type="linear"

    best_clf = SVC(kernel='linear', C=0.1)

    results = testClassifier(X, y, best_clf, dataset, learner, model_type)
    plotResults(results, "Optimal Linear Model", dataset, learner)

    # Test as an rbf model
    print("----- Best RBF Model -----")
    # param_grid = {'C': [0.1, 1, 10, 100, 1000],
    #               'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

    # svm_model = SVC(kernel='rbf')
    # svm_clf = GridSearchCV(svm_model, param_grid, n_jobs=-1)

    # svm_clf.fit(X_train, y_train)

    # best_clf = svm_clf.best_estimator_
    # print(svm_clf.best_params_)

    # output = ""

    # for param in param_grid.keys():
    #     output += param + ": " + str(svm_clf.best_params_[param]) + "\n"

    # output_path = "./images/" + dataset + "/" + learner + "/optimal_rbf_params.txt"
    # output_file = open(output_path, "w")
    # output_file.write(output)
    # output_file.close()
    model_type="rbf"

    best_clf = SVC(kernel='rbf', C=1000, gamma=0.0001)

    results = testClassifier(X, y, best_clf, dataset, learner, model_type)
    plotResults(results, "Optimal RBF Model", dataset, learner)

    # Test as an rbf model
    print("----- Best Poly Model -----")
    # param_grid = {'C': [0.1, 1, 10, 100, 1000],
    #               'degree': [2,3,4],
    #               'coef0': [0.01, 0.05, 0.1, 0.5, 1, 2]}

    # svm_model = SVC(kernel='poly')
    # svm_clf = GridSearchCV(svm_model, param_grid, n_jobs=-1)

    # svm_clf.fit(X_train, y_train)

    # best_clf = svm_clf.best_estimator_
    # print(svm_clf.best_params_)

    # output = ""

    # for param in param_grid.keys():
    #     output += param + ": " + str(svm_clf.best_params_[param]) + "\n"

    # output_path = "./images/" + dataset + "/" + learner + "/optimal_poly_params.txt"
    # output_file = open(output_path, "w")
    # output_file.write(output)
    # output_file.close()    

    model_type="poly"

    best_clf = SVC(kernel='poly', C=1, degree=2, coef0=0.1)

    results = testClassifier(X, y, best_clf, dataset, learner, model_type)
    plotResults(results, "Optimal Poly Model", dataset, learner)


    
