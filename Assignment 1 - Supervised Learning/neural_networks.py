import numpy as np


import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from util import *
from sklearn.neural_network import MLPClassifier

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def run_experiment(data, dataset):
    learner="nn"
    # Scale and split Data
    X, y = data
    maxabs_scaler = StandardScaler()

    X_scaled = maxabs_scaler.fit_transform(X, y)

    split = StratifiedShuffleSplit(n_splits=1 , test_size=0.2, random_state=42)

    num_features = X.shape[1]

    for train_ind, test_ind in split.split(X_scaled, y):
        X_train, y_train = X_scaled[train_ind], y[train_ind]
        X_test, y_test = X_scaled[test_ind], y[test_ind]
        
    # Test with Base Model
    print("----- Test Base Model -----")
    nn_model = MLPClassifier(max_iter=750)
    nn_model.fit(X_train, y_train)
    model_type = "base"

    results = testClassifier(X, y, nn_model, dataset, learner, model_type)
    plotResults(results, "Base Model", dataset, learner)

    print("----- Test Optimal Model -----")
    param_grid = {'activation': ['logistic', 'tanh', 'relu'],
                  'solver': ['sgd', 'adam'],
                  'alpha': [0.0001, 0.001, 0.01],
                  'hidden_layer_sizes': [(num_features*2,num_features), (num_features*10,num_features*5), (num_features*50,num_features*25)]}

    nn_model = MLPClassifier(max_iter=750)
    nn_clf = GridSearchCV(nn_model, param_grid, n_jobs=-1)

    nn_clf.fit(X_train, y_train)

    best_clf = nn_clf.best_estimator_
    print(nn_clf.best_params_)
    output = ""

    for param in param_grid.keys():
        output += param + ": " + str(nn_clf.best_params_[param]) + "\n"

    output_path = "./images/" + dataset + "/" + learner + "/optimal_params.txt"
    output_file = open(output_path, "w")
    output_file.write(output)
    output_file.close()
    model_type="optimal"

    results = testClassifier(X, y, best_clf, dataset, learner, model_type)
    plotResults(results, "Optimal Model", dataset, learner)

    # Test as a function of Alpha
    print("----- Function of Alpha -----")
    alphas = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    boost_model = MLPClassifier(activation=nn_clf.best_params_['activation'], solver=nn_clf.best_params_['solver'], hidden_layer_sizes=nn_clf.best_params_['hidden_layer_sizes'], max_iter=750)
    acc_train_scores, acc_test_scores = validation_curve(boost_model, X_train, y_train, param_name='alpha', param_range=alphas, scoring = 'accuracy', cv=10, n_jobs=-1)
    f1_train_scores, f1_test_scores = validation_curve(boost_model, X_train, y_train, param_name='alpha', param_range=alphas, scoring = 'f1', cv=10, n_jobs=-1)
    results = np.mean(acc_train_scores, axis=1), np.mean(acc_test_scores, axis=1), np.mean(f1_train_scores, axis=1), np.mean(f1_test_scores, axis=1), alphas
    plotResults(results, "Function of Alpha", dataset, learner, "Alpha")


    
