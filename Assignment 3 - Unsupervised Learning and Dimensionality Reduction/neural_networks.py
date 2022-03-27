import numpy as np

import sklearn
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, validation_curve
from sklearn.preprocessing import MinMaxScaler
import util
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import util

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def run_experiment(data, dataset, subdir):
    learner="NN"
    # Scale and split Data
    X, y = data

    split = StratifiedShuffleSplit(n_splits=1 , test_size=0.2, random_state=42)

    num_features = X.shape[1]

    for train_ind, test_ind in split.split(X, y):
        X_train, y_train = X[train_ind], y[train_ind]
        X_test, y_test = X[test_ind], y[test_ind]
        
    # Test with Base Model
    print("----- Test Base Model -----")
    nn_model = MLPClassifier(early_stopping=True)
    nn_model.fit(X_train, y_train)
    model_type = "base"

    results = util.testClassifier(X, y, nn_model, dataset, learner, model_type, subdir=subdir)
    util.plotResults(results, "Base Model", dataset, learner, subdir=subdir)

    print("----- Test Optimal Model -----")
    param_grid = {'activation': ['tanh', 'relu'],
                  'solver': ['sgd', 'adam'],
                  'alpha': [0.0001, 0.001],
                  'hidden_layer_sizes': [(num_features*2,num_features), (num_features*10,num_features*5)]}

    nn_model = MLPClassifier(early_stopping=True)
    nn_clf = GridSearchCV(nn_model, param_grid, n_jobs=-1)

    nn_clf.fit(X_train, y_train)

    best_clf = nn_clf.best_estimator_
    print(nn_clf.best_params_)
    output = ""

    for param in param_grid.keys():
        output += param + ": " + str(nn_clf.best_params_[param]) + "\n"

    output_path = "./images/" + dataset + "/" + learner + "/" + subdir + "/optimal_params.txt"
    output_file = open(output_path, "w")
    output_file.write(output)
    output_file.close()
    model_type="optimal"

    results = util.testClassifier(X, y, best_clf, dataset, learner, model_type, subdir=subdir)
    util.plotResults(results, "Optimal Model", dataset, learner, subdir=subdir)

    # Test as a function of Alpha
    print("----- Function of Alpha -----")
    alphas = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    boost_model = MLPClassifier(early_stopping=True, activation=nn_clf.best_params_['activation'], solver=nn_clf.best_params_['solver'], hidden_layer_sizes=nn_clf.best_params_['hidden_layer_sizes'], max_iter=750)
    acc_train_scores, acc_test_scores = validation_curve(boost_model, X_train, y_train, param_name='alpha', param_range=alphas, scoring = 'accuracy', cv=10, n_jobs=-1)
    f1_train_scores, f1_test_scores = validation_curve(boost_model, X_train, y_train, param_name='alpha', param_range=alphas, scoring = 'f1', cv=10, n_jobs=-1)
    results = np.mean(acc_train_scores, axis=1), np.mean(acc_test_scores, axis=1), np.mean(f1_train_scores, axis=1), np.mean(f1_test_scores, axis=1), alphas
    util.plotResults(results, "Function of Alpha", dataset, learner, "Alpha", subdir=subdir)

def main(random_state=2022):
    data = util.load_wine_data_easy()

    X, y = data
    num_samples, num_features = X.shape[:2]

    std_scaler = MinMaxScaler()
    X = std_scaler.fit_transform(X, y)

    # PCA
    print("Run PCA")
    optimal_components = 8
    X_pca = PCA(n_components=optimal_components, random_state=random_state).fit_transform(X, y)
    # run_experiment((X_pca, y), 'wine', subdir='PCA/')

    # ICA
    print("Run ICA")
    optimal_components = 8
    X_ica = FastICA(n_components=optimal_components, random_state=random_state).fit_transform(X, y)
    # run_experiment((X_ica, y), 'wine', subdir='ICA/')

    # RP
    print("Run RP")
    optimal_components = 10
    X_rp = GaussianRandomProjection(n_components=optimal_components, random_state=random_state).fit_transform(X, y)
    # run_experiment((X_rp, y), 'wine', subdir='RP/')

    # NNMF
    print("Run NNMF")
    optimal_components = 8
    X_nnmf = NMF(n_components=optimal_components, random_state=random_state).fit_transform(X, y)
    # run_experiment((X_nnmf, y), 'wine', subdir='NNMF/')
    
    # K-Means
    print("Run KMeans")
    print("Run PCA")
    km = KMeans(n_clusters=2, random_state=random_state, max_iter=1000).fit_transform(X_pca)
    X_new = np.append(X_pca, km, axis=1)
    # run_experiment((X_new, y), 'wine', subdir='KMeans/PCA/')

    print("Run ICA")
    km = KMeans(n_clusters=2, random_state=random_state, max_iter=1000).fit_transform(X_ica)
    X_new = np.append(X_ica, km, axis=1)
    # run_experiment((X_new, y), 'wine', subdir='KMeans/ICA/')

    print("Run RP")
    km = KMeans(n_clusters=2, random_state=random_state, max_iter=1000).fit_transform(X_rp)
    X_new = np.append(X_rp, km, axis=1)
    # run_experiment((X_new, y), 'wine', subdir='KMeans/RP/')

    print("Run NNMF")
    km = KMeans(n_clusters=2, random_state=random_state, max_iter=1000).fit_transform(X_nnmf)
    X_new = np.append(X_nnmf, km, axis=1)
    # run_experiment((X_new, y), 'wine', subdir='KMeans/NNMF/')

    # EM
    print("Run EM")
    print("Run PCA")
    em = GaussianMixture(n_components=2, random_state=random_state, max_iter=1000).fit(X_pca).predict_proba(X_pca)
    X_new = np.append(X_pca, em, axis=1)
    run_experiment((X_new, y), 'wine', subdir='EM/PCA/')

    print("Run ICA")
    em = GaussianMixture(n_components=2, random_state=random_state, max_iter=1000).fit(X_pca).predict_proba(X_pca)
    X_new = np.append(X_ica, em, axis=1)
    run_experiment((X_new, y), 'wine', subdir='EM/ICA/')

    print("Run RP")
    em = GaussianMixture(n_components=2, random_state=random_state, max_iter=1000).fit(X_pca).predict_proba(X_pca)
    X_new = np.append(X_rp, em, axis=1)
    run_experiment((X_new, y), 'wine', subdir='EM/RP/')

    print("Run NNMF")
    em = GaussianMixture(n_components=2, random_state=random_state, max_iter=1000).fit(X_pca).predict_proba(X_pca)
    X_new = np.append(X_nnmf, em, axis=1)
    run_experiment((X_new, y), 'wine', subdir='EM/NNMF/')


if __name__ == "__main__":
    main()
