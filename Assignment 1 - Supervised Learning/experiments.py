import boosting
import decision_tree
import knn
import svm
import neural_networks
import util
import numpy as np

dataset="wine"
data=util.load_wine_data_easy()

# print("Running Wine Decision Tree Experiment")
# decision_tree.run_experiment(data, dataset)

print("Running Wine Neural Network Experiment")
neural_networks.run_experiment(data, dataset)

# print("Running Wine Boosting Experiment")
# boosting.run_experiment(data, dataset)

# print("Running Wine Support Vector Machine Experiment")
# svm.run_experiment(data, dataset)

# print("Running Wine k-Nearest Neighbors Experiment")
# knn.run_experiment(data, dataset)


dataset="credit"
data=util.load_credit_data()

# print("Running Credit Decision Tree Experiment")
# decision_tree.run_experiment(data, dataset)

# print("Running Credit Neural Network Experiment")
# neural_networks.run_experiment(data, dataset)

# print("Running Credit Boosting Experiment")
# boosting.run_experiment(data, dataset)

# print("Running Credit Support Vector Machine Experiment")
# svm.run_experiment(data, dataset)

# print("Running Credit k-Nearest Neighbors Experiment")
# knn.run_experiment(data, dataset)
