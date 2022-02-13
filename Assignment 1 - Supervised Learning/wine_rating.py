import csv
import numpy as np

def load_data():
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

    # Normalize Values
    for col in range(X.shape[1]):
        X[:, col] -= np.min(X[:,col])
        X[:, col] /= np.max(X[:,col])

    arr = np.hstack((X, y))
    np.random.shuffle(arr)

    X = arr[:,:-1]
    y = arr[:,-1]

    return X, y
    
X, y = load_data()