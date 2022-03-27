
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import time

data = util.load_wine_data_easy()

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

nn_model = MLPClassifier(hidden_layer_sizes=(550, 275), activation='tanh', solver='adam', alpha=0.0001, max_iter=100000)

#nn_model = MLPClassifier(activation='tanh', hidden_layermax_iter=750)
start_time = time.time()
nn_model.fit(X_train, y_train)
end_time = time.time()

print("Train Time: ", round(end_time - start_time, 3), "s", sep='')

print("Training Classification Report")
y_train_pred = nn_model.predict(X_train)
print(classification_report(y_train, y_train_pred))

print("Testing Classification Report")
y_test_pred = nn_model.predict(X_test)
print(classification_report(y_test, y_test_pred))
