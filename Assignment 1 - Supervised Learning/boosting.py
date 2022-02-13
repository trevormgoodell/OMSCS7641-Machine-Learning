import numpy as np


import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from util import *
from sklearn.ensemble import AdaBoostClassifier

def run_experiment(data, dataset):
    learner = "boosting"
    # Scale and split Data
    X, y = data
    maxabs_scaler = StandardScaler()

    X = maxabs_scaler.fit_transform(X, y)

    ccp_alpha = 0.01
    criterion='gini'

    split = StratifiedShuffleSplit(n_splits=1 , test_size=0.2, random_state=42)

    for train_ind, test_ind in split.split(X, y):
        X_train, y_train = X[train_ind], y[train_ind]
        X_test, y_test = X[test_ind], y[test_ind]
        
    # Test with Base Model
    print("----- Test Base Model -----")
    boost_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion=criterion, ccp_alpha=ccp_alpha))
    boost_model.fit(X_train, y_train)
    model_type="base"

    results = testClassifier(X, y, boost_model, dataset, learner, model_type)
    plotResults(results, "Base Model", dataset, learner)

    print("----- Test Optimal Model -----")
    param_grid = {'n_estimators':[25, 50, 75, 100, 125, 150, 175, 200, 225, 250],
                  'learning_rate': [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]}
    boost_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion=criterion, ccp_alpha=ccp_alpha))
    boost_clf = GridSearchCV(boost_model, param_grid, n_jobs=-1)

    boost_clf.fit(X_train, y_train)

    best_clf = boost_clf.best_estimator_
    print(boost_clf.best_params_)
    output = ""

    for param in param_grid.keys():
        output += param + ": " + str(boost_clf.best_params_[param]) + "\n"

    output_path = "./images/" + dataset + "/" + learner + "/optimal_params.txt"
    output_file = open(output_path, "w")
    output_file.write(output)
    output_file.close()
    model_type="optimal"

    results = testClassifier(X, y, best_clf, dataset, learner, model_type)
    plotResults(results, "Optimal Model", dataset, learner)

    # Test as a function of Learning Rate
    print("----- Function of Learning Rate -----")
    learning_rates = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
    boost_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion=criterion, ccp_alpha=ccp_alpha), n_estimators=boost_clf.best_params_['n_estimators'])
    acc_train_scores, acc_test_scores = validation_curve(boost_model, X_train, y_train, param_name='learning_rate', param_range=learning_rates, scoring = 'accuracy', cv=10, n_jobs=-1)
    f1_train_scores, f1_test_scores = validation_curve(boost_model, X_train, y_train, param_name='learning_rate', param_range=learning_rates, scoring = 'f1', cv=10, n_jobs=-1)
    results = np.mean(acc_train_scores, axis=1), np.mean(acc_test_scores, axis=1), np.mean(f1_train_scores, axis=1), np.mean(f1_test_scores, axis=1), learning_rates
    plotResults(results, "Function of Learning Rate", dataset, learner, "Learning Rate")

    # Test as a function of N Estimators
    print("----- Function of N Estimators -----")
    N_estimators = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    boost_model_n = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion=criterion, ccp_alpha=ccp_alpha), learning_rate=boost_clf.best_params_['learning_rate'])
    acc_train_scores, acc_test_scores = validation_curve(boost_model_n, X_train, y_train, param_name='n_estimators', param_range=N_estimators, scoring = 'accuracy', cv=10, n_jobs=-1)
    f1_train_scores, f1_test_scores = validation_curve(boost_model_n, X_train, y_train, param_name='n_estimators', param_range=N_estimators, scoring = 'f1', cv=10, n_jobs=-1)
    results = np.mean(acc_train_scores, axis=1), np.mean(acc_test_scores, axis=1), np.mean(f1_train_scores, axis=1), np.mean(f1_test_scores, axis=1), N_estimators
    plotResults(results, "Function of N Estimators", dataset, learner, "N Estimators") 