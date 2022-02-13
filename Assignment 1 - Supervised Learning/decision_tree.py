import numpy as np


import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV, validation_curve
from sklearn.preprocessing import StandardScaler
from util import *
from sklearn.tree import DecisionTreeClassifier

def run_experiment(data, dataset):
    learner="dt"
    # Scale and split Data
    X, y = data
    maxabs_scaler = StandardScaler()

    X = maxabs_scaler.fit_transform(X, y)

    split = StratifiedShuffleSplit(n_splits=1 , test_size=0.2, random_state=42)

    for train_ind, test_ind in split.split(X, y):
        X_train, y_train = X[train_ind], y[train_ind]
        X_test, y_test = X[test_ind], y[test_ind]
    
    '''
    # Test with Base Model
    print("----- Test Base Model -----")
    dt_model = DecisionTreeClassifier(ccp_alpha=0.00001)
    dt_model.fit(X_train, y_train)
    model_type="base"


    results = testClassifier(X, y, dt_model, dataset, learner, model_type)
    plotResults(results, "Base Model", dataset, learner)

    # Test with Optimal Model
    print("----- Test Optimal Model -----")
    param_grid = {'criterion': ["gini", "entropy"],
                  'ccp_alpha': [0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025]}
    dt_model = DecisionTreeClassifier()
    dt_clf = GridSearchCV(dt_model, param_grid, n_jobs=-1)

    dt_clf.fit(X_train, y_train)

    best_clf = dt_clf.best_estimator_
    print(dt_clf.best_params_)
    model_type="optimal"
    output = ""

    for param in param_grid.keys():
        output += param + ": " + str(dt_clf.best_params_[param]) + "\n"

    output_path = "./images/" + dataset + "/" + learner + "/optimal_params.txt"
    output_file = open(output_path, "w")
    output_file.write(output)
    output_file.close()

    results = testClassifier(X, y, best_clf, dataset, learner, model_type)
    plotResults(results, "Optimal Model", dataset, learner)
    '''
    # Test as a function of CCP-Alpha
    print("----- Function of CCP-Alpha -----")
    ccp_alphas = [0.00001, 0.000025, 0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.0125, 0.015]
    dt_model = DecisionTreeClassifier(criterion='entropy' )
    acc_train_scores, acc_test_scores = validation_curve(dt_model, X_train, y_train, param_name='ccp_alpha', param_range=ccp_alphas, scoring = 'accuracy', cv=10)
    f1_train_scores, f1_test_scores = validation_curve(dt_model, X_train, y_train, param_name='ccp_alpha', param_range=ccp_alphas, scoring = 'f1', cv=10)
    results = np.mean(acc_train_scores, axis=1), np.mean(acc_test_scores, axis=1), np.mean(f1_train_scores, axis=1), np.mean(f1_test_scores, axis=1), ccp_alphas
    plotResults(results, "Function of CCP-Alpha", dataset, learner, "CCP-Alpha")


    
