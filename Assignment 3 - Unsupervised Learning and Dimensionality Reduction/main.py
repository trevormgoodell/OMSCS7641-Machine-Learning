from cProfile import run
from random import random
from re import S
import numpy as np

from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, completeness_score, explained_variance_score
from sklearn.exceptions import ConvergenceWarning

import warnings

import util

import matplotlib.pyplot as plt

def run_experiment(data, output_dir, max_clusters = 25, random_state=2022):
    def plot_clusterings(scores, scoring_metric, silhouettes, completenesses, subdir, xlabel):
        if scoring_metric == "Inertia":
            max_value = max(scores.values())

            for key in scores.keys():
                scores[key] /= max_value

        fig, ax = plt.subplots(ncols = 1, nrows = 1)

        ax.plot(scores.keys(), scores.values(), 'r.-')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(scoring_metric)
        ax.set_title(scoring_metric)
        ax.grid(visible=True)
        ax.set_xlim(min(scores.keys()), max(scores.keys()))
        ax.set_ylim(min(scores.values()), max(scores.values()))
        plt.savefig(output_dir + subdir + scoring_metric)
        plt.close('all')

        fig, ax = plt.subplots(ncols = 1, nrows = 1)

        ax.plot(silhouettes.keys(), silhouettes.values(), 'r.-')
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Scores")
        ax.grid(visible=True)
        ax.set_xlim(min(silhouettes.keys()), max(silhouettes.keys()))
        ax.set_ylim(-1, 1)
        plt.savefig(output_dir + subdir + 'silhouette')
        plt.close('all')

        fig, ax = plt.subplots(ncols = 1, nrows = 1)

        ax.plot(completenesses.keys(), completenesses.values(), 'r.-')
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Completeness Score")
        ax.set_title("Completeness Scores")
        ax.grid(visible=True)
        ax.set_xlim(min(completenesses.keys()), max(completenesses.keys()))
        ax.set_ylim(0, 1)
        plt.savefig(output_dir + subdir + 'completeness')
        plt.close('all')
        pass

    def run_clusterings(data_X, data_y, subdir = ""):
        if True:
            inertias = {}
            silhouettes = {}
            completenesses = {}

            output = "K-Means"

            for n_clusters in range(2, max_clusters + 1):
                km = KMeans(n_clusters=n_clusters, random_state=random_state, max_iter=1000).fit(data_X)

                silhouettes[n_clusters] = silhouette_score(data_X, km.labels_)
                completenesses[n_clusters] = completeness_score(data_y, km.labels_)

                current_output = "Number of Clusters: {} Inertia: {} Silhouette Score: {} Completeness Score: {}".format(n_clusters, round(km.inertia_, 3), round(silhouettes[n_clusters], 3), round(completenesses[n_clusters], 3))
                # print(current_output)

                output += "\n" + current_output
                inertias[n_clusters] = km.inertia_

            with open(output_dir + subdir + "KMeans/report.txt", 'w') as report_file:
                report_file.writelines(output)

            plot_clusterings(inertias, "Inertia", silhouettes, completenesses, subdir + "KMeans/", "Number of Clusters")

        # Expectation Maximization
        if True:
            scores = {}
            silhouettes = {}
            completenesses = {}

            output = "Expectation Maximization"

            for n_components in range(2, max_clusters + 1):
                gm = GaussianMixture(n_components=n_components, random_state=random_state).fit(data_X)

                silhouettes[n_components] = silhouette_score(data_X, gm.predict(data_X))
                completenesses[n_components] = completeness_score(data_y, gm.predict(data_X))

                score = gm.score(data_X)

                current_output = "Number of Components: {} Score: {} Silhouette Score: {} Completeness Score: {}".format(n_components, round(score, 3), round(silhouettes[n_components], 3), round(completenesses[n_components], 3))
                # print(current_output)

                output += "\n" + current_output
                scores[n_components] = score

            with open(output_dir + subdir + "EM/report.txt", 'w') as report_file:
                report_file.writelines(output)

            plot_clusterings(scores, "Score", silhouettes, completenesses, subdir + "EM/", "Number of Components")

        pass
    
    def run_dimensionality_reduction(algorithm, algo_dir):
        evr = {}
        output = algo_dir.replace('/', '')

        for n_components in range(2, num_features):
            dra = algorithm(n_components=n_components, random_state=random_state)

            X_dra = dra.fit_transform(X, y)

            try:
                inv_X_dra = dra.inverse_transform(X_dra)
            except:
                inv_X_dra = np.dot(X_dra, np.linalg.pinv(dra.components_.T))

            evr[n_components] = explained_variance_score(X, inv_X_dra)

            current_output = current_output = "Number of Components: {} Explained Variance: {}".format(n_components, round(evr[n_components], 3))

            output += "\n" + current_output

        fig, ax = plt.subplots(ncols = 1, nrows = 1)

        ax.plot(evr.keys(), evr.values(), 'b.-', label="EVR")
        ax.set_xlabel("Number of Features")
        ax.set_title("Explained Variance")
        ax.grid(visible=True)
        ax.set_xlim(2, num_features - 1)
        ax.set_ylim(0, 1)
        plt.savefig(output_dir + algo_dir + "metrics")
        plt.close('all')

        dra = algorithm(n_components=2, random_state=random_state)
        X_dra = dra.fit_transform(X, y)

        fig, ax = plt.subplots(ncols = 1, nrows = 1)

        X_temp = X_dra[y == 0]
        ax.scatter(X_temp[:,0], X_temp[:,1], marker='.', linewidths=1, label='0')

        X_temp = X_dra[y == 1]
        ax.scatter(X_temp[:,0], X_temp[:,1], marker='.', linewidths=1, label='1')

        ax.set_xlabel("First Component")
        ax.set_ylabel("Second Component")
        ax.set_title("2d Visualization")
        ax.grid(visible=True)
        plt.savefig(output_dir + algo_dir + "2dvisual")
        plt.close('all')

        dra = algorithm(n_components=3, random_state=random_state)
        X_dra = dra.fit_transform(X, y)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        X_temp = X_dra[y == 0]
        ax.scatter(X_temp[:,0], X_temp[:,1], X_temp[:,2], marker='.', linewidths=1, label='0')

        X_temp = X_dra[y == 1]
        ax.scatter(X_temp[:,0], X_temp[:,1], X_temp[:,2], marker='.', linewidths=1, label='1')

        ax.set_xlabel("First Component")
        ax.set_ylabel("Second Component")
        ax.set_zlabel("Third Component")
        ax.set_title("3d Visualization")
        ax.grid(visible=True)
        plt.savefig(output_dir + algo_dir + "3dvisual")
        plt.close('all')

        with open(output_dir + algo_dir+ "/report.txt", 'w') as report_file:
                    report_file.writelines(output)

        min_dist = 1
        best_key = -1

        for key in evr.keys():
            dist = np.abs(evr[key] - 0.9)

            if dist < min_dist:
                min_dist = dist
                best_key = key

        optimal_components = best_key

        reduced_dim_X = algorithm(n_components=optimal_components, random_state=random_state).fit_transform(X, y)

        run_clusterings(reduced_dim_X, y, algo_dir)

        return optimal_components

    X, y = data

    num_samples, num_features = X.shape[:2]

    std_scaler = MinMaxScaler()

    X = std_scaler.fit_transform(X, y)
    # Run clustering
    # run_clusterings(X, y)
    
    # Run dimensionality reduction algorithms
    print("Running PCA")
    optimal_pca = run_dimensionality_reduction(PCA, "PCA/")

    print("Running ICA")
    optimal_ica = run_dimensionality_reduction(FastICA, "ICA/")

    print("Running RP")
    optimal_rp = run_dimensionality_reduction(GaussianRandomProjection, "RP/")

    print("Running NNMF")
    optimal_lda = run_dimensionality_reduction(NMF, "NNMF/")

def random_experiments(data, output_dir, max_clusters = 25, random_state=2022):
    pass

def main():
    data = util.load_wine_data_easy()
    run_experiment(data, './images/wine/')

    data = util.load_credit_data()
    run_experiment(data, './images/credit/')

    pass

if __name__ == "__main__":
    main()