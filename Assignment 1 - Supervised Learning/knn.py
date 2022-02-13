from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit, cross_val_score, validation_curve
import numpy as np
from sklearn.preprocessing import StandardScaler
from util import *
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def run_experiment(data, dataset):
    learner="knn"
    X, y = data
    maxabs_scaler = StandardScaler()

    X = maxabs_scaler.fit_transform(X, y)

    split = StratifiedShuffleSplit(n_splits=1 , test_size=0.2, random_state=42)

    for train_ind, test_ind in split.split(X, y):
        X_train, y_train = X[train_ind], y[train_ind]
        X_test, y_test = X[test_ind], y[test_ind]
        
    # Test with Base Model
    print("----- Test Base Model -----")
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)
    model_type="base"

    results = testClassifier(X, y, knn_model, dataset, learner, model_type)
    plotResults(results, "Base Model", dataset, learner)

    # Test with Optimal Model
    print("----- Test Optimal Model -----")
    param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 17, 19, 25, 50, 75, 100, 125, 150, 175, 200],
                  'weights': ['uniform', 'distance'],
                  'metric': ['euclidean', 'manhattan', 'chebyshev']}
    knn_model = KNeighborsClassifier(n_jobs=-1)
    knn_clf = GridSearchCV(knn_model, param_grid, n_jobs=-1)

    knn_clf.fit(X_train, y_train)

    best_clf = knn_clf.best_estimator_
    print(knn_clf.best_params_)

    output = ""

    for param in param_grid.keys():
        output += param + ": " + str(knn_clf.best_params_[param]) + "\n"

    output_path = "./images/" + dataset + "/" + learner + "/optimal_params.txt"
    output_file = open(output_path, "w")
    output_file.write(output)
    output_file.close()
    model_type="optimal"

    results = testClassifier(X, y, best_clf, dataset, learner, model_type)
    plotResults(results, "Optimal Model", dataset, learner)

    # Test as a function of K
    print("----- Function of K -----")
    K = [1, 3, 5, 7, 9, 11, 13, 17, 19, 25, 50, 75, 100, 125, 150, 175, 200]
    knn_model = KNeighborsClassifier(metric=knn_clf.best_params_['metric'], weights=knn_clf.best_params_['weights'], n_jobs=-1)
    acc_train_scores, acc_test_scores = validation_curve(knn_model, X_train, y_train, param_name='n_neighbors', param_range=K, scoring = 'accuracy', cv=10, n_jobs=-1)
    f1_train_scores, f1_test_scores = validation_curve(knn_model, X_train, y_train, param_name='n_neighbors', param_range=K, scoring = 'f1', cv=10, n_jobs=-1)
    results = np.mean(acc_train_scores, axis=1), np.mean(acc_test_scores, axis=1), np.mean(f1_train_scores, axis=1), np.mean(f1_test_scores, axis=1), K
    plotResults(results, "Function of K", dataset, learner, "K")