import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report




def load_wine_data():
    with open('winequality-red.csv', 'r') as csvfile:
        winequality_red = csv.reader(csvfile)
        winequality_red_list = [wine for wine in winequality_red]
        headers = winequality_red_list[0]
        winequality_red_asarray = np.array([wine for wine in winequality_red_list[1:]], dtype=np.float32)

    with open('winequality-white.csv', 'r') as csvfile:
        winequality_white = csv.reader(csvfile)
        winequality_white_list = [wine for wine in winequality_white]
        headers = winequality_white_list[0]
        winequality_white_asarray = np.array([wine for wine in winequality_white_list[1:]], dtype=np.float32)

    X_red = winequality_red_asarray[:,:-1]
    y_red = winequality_red_asarray[:,-1].reshape(-1, 1)

    X_white = winequality_white_asarray[:,:-1]
    y_white = winequality_white_asarray[:,-1].reshape(-1,1)


    X = np.vstack((X_red, X_white))
    y = np.vstack((y_red, y_white))

    arr = np.hstack((X, y))
    np.random.shuffle(arr)

    X = arr[:,:-1]
    y = arr[:,-1]

    return X, y
    

def load_wine_data_easy():
    X, y = load_wine_data()
    hist = np.histogram(y, range=(0.,10.))
    y_easy = np.int32(y > 5)
    return X, y_easy

def load_credit_data():
    with open('SouthGermanCredit.asc', 'r') as input_file:
        credit_data = input_file.readlines()
        credit_data = credit_data[1:]
        credit_data = np.array([np.int32(line.strip().split()) for line in credit_data])

        
    np.random.shuffle(credit_data)

    X = credit_data[:,:-1]
    y = credit_data[:,-1]

    return X, y

def testClassifier(X, y, classifier, dataset, learner, model_type, n_splits=5):
    avg_train_acc_list = []
    avg_test_acc_list = []

    avg_train_f1_list = []
    avg_test_f1_list = []

    indices = []

    min_val = np.int32(X.shape[0] * (5/100))

    final_conf_mat = np.zeros((2,2))

    for num_samp in np.linspace(min_val, X.shape[0], 20, dtype=np.int32):
        X_train = X[:num_samp]
        y_train = y[:num_samp]

        skfolds = StratifiedKFold(n_splits=n_splits)

        accuracy_list_train = []
        accuracy_list_test = []

        f1_list_train = []
        f1_list_test = []
        
        for train_index, test_index in skfolds.split(X_train, y_train):
            # Split the training data into it's folds
            X_train_fold = X_train[train_index]
            y_train_fold = y_train[train_index]

            X_test_fold = X_train[test_index]
            y_test_fold = y_train[test_index]

            # Fit the classifier on the fold
            classifier.fit(X_train_fold, y_train_fold)

            # Get the training and testing predictions
            y_train_fold_pred = classifier.predict(X_train_fold)
            y_test_fold_pred = classifier.predict(X_test_fold)

            # Calculate the accuracy
            accuracy_list_train.append(np.sum(y_train_fold_pred == y_train_fold) / len(y_train_fold))
            accuracy_list_test.append(np.sum(y_test_fold_pred == y_test_fold) / len(y_test_fold))

            # Calculate the f1 score
            f1_list_train.append(f1_score(y_train_fold, y_train_fold_pred))
            f1_list_test.append(f1_score(y_test_fold, y_test_fold_pred))

            if num_samp == X.shape[0]:
                conf_mat = confusion_matrix(y_test_fold, y_test_fold_pred)
                final_conf_mat += conf_mat

        # Calculate average metric
        avg_train_acc = np.mean(accuracy_list_train)
        avg_test_acc = np.mean(accuracy_list_test)
        
        avg_train_f1 = np.mean(f1_list_train)
        avg_test_f1 = np.mean(f1_list_test)

        # Append average metric to their list for being returned
        avg_train_acc_list.append(avg_train_acc)
        avg_test_acc_list.append(avg_test_acc)

        avg_train_f1_list.append(avg_train_f1)
        avg_test_f1_list.append(avg_test_f1)

    final_conf_mat /= n_splits

    final_conf_mat = np.round(final_conf_mat, decimals=2)

    TP = final_conf_mat[1,1]
    TN = final_conf_mat[0,0]
    FP = final_conf_mat[0,1]
    FN = final_conf_mat[1,0]

    prec = TP / (TP + FP)
    recall = TP / (TP + FN)

    f1 = (2 * prec * recall) / (prec + recall)

    accuracy = (final_conf_mat[0,0] + final_conf_mat[1,1]) / np.sum(final_conf_mat)

    output = np.round(np.array([[prec   , recall, f1],
                                [0      , 0     , accuracy]]), decimals=2)

    conf_mat_title = "./images/" + dataset + "/" + learner + "/" + model_type + "_" + "confusion_matrix.txt"
    with open(conf_mat_title, "w") as output_file:
        output_file.write(str(final_conf_mat))

    class_rep_title = "./images/" + dataset + "/" + learner + "/" + model_type + "_" + "classification_report.txt"
    with open(class_rep_title, "w") as output_file:
        output_file.write(str(output))

    indices = np.linspace(5, 100, 20) / 100
    
    return avg_train_acc_list, avg_test_acc_list, avg_train_f1_list, avg_test_f1_list, indices

def plotResults(results, title, dataset, learner, xlabel="Number of Training Instances Used"):
    avg_train_acc_list, avg_test_acc_list, avg_train_f1_list, avg_test_f1_list, indices = results
    plt.clf()
    plt.plot(indices, avg_train_acc_list, "r:+", linewidth=1, label="Train Acc")
    plt.plot(indices, avg_test_acc_list, "r-+", linewidth=1, label="Test Acc")
    plt.plot(indices, avg_train_f1_list, "b:+", linewidth=1, label="Train F1")
    plt.plot(indices, avg_test_f1_list, "b-+", linewidth=1, label="Test F1")

    plt.title(title)
    plt.legend()
    plt.xlabel(xlabel)
    plt.margins(0.1,0.1)

    plt.ylim([0.5, 1.0])
    
    plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    fig_title = "./images/" + dataset + "/" + learner + "/" + title.replace(" ", "_")

    plt.savefig(fig_title)

def modelStatistics(model, X_test, y_test, dataset, learner, model_type):
    y_pred = model.predict(X_test)

    print("  --- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))

    conf_mat_title = "./images/" + dataset + "/" + learner + "/" + model_type + "_" + "confusion_matrix.txt"
    with open(conf_mat_title, "w") as output_file:
        output_file.write(str(confusion_matrix(y_test, y_pred)))

    print("  --- Classification Report ---")
    print(classification_report(y_test, y_pred))

    class_rep_title = "./images/" + dataset + "/" + learner + "/" + model_type + "_" + "classification_report.txt"
    with open(class_rep_title, "w") as output_file:
        output_file.write(classification_report(y_test, y_pred))
    
    pass


def report_sep(report_name, width=20):
        print("-"*width, " ", report_name, " ", "-"*width)

load_credit_data()