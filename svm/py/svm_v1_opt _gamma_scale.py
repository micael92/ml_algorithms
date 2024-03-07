# %%
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  


# %%
# MNIST Datensatz aus keras laden
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# %%
# Das Shape des Datensatzes ausgeben
print("Ursprungs Shape des Trainingsdatensatzes:", x_train.shape)
print("Ursprungs Shape des Testdatensatzes:", x_test.shape)

# %%
# Den Datensatz für ein SVM vorbereiten

# Flatten der Bildmatrizen für die SVM (von 28x28 zu 784)
x_train_svm = x_train.reshape((x_train.shape[0], -1))
x_test_svm = x_test.reshape((x_test.shape[0], -1))
# Standardisierung der Feature-Werte
scaler = StandardScaler()
x_train_svm = scaler.fit_transform(x_train_svm)
x_test_svm = scaler.transform(x_test_svm)

# %%
# Das Shape des Datensatzes ausgeben
print("SVM Shape des Trainingsdatensatzes:", x_train_svm.shape)
print("SVM Shape des Testdatensatzes:", x_test_svm.shape)

# %%
# Parameteroptimierung mit RandomizedSearchCV

# Parameterverteilungen statt eines Gitters
param_distributions = {
    'C': [0.1],
    'gamma': ['scale'],
    #'gamma': [0.1, 0.01]
    'kernel': ['rbf']
}

# RandomizedSearchCV initialisieren
random_search = RandomizedSearchCV(svm.SVC(), param_distributions, n_iter=1, verbose=2, cv= None, random_state=42, n_jobs=-1)

# Auf den Trainingsdaten fitten
random_search.fit(x_train_svm, y_train)

# Beste Parameter-Kombination
print("Beste Parameter-Kombination:", random_search.best_params_)


# %%
import pandas as pd

# Save cv_results_df to a CSV file
cv_results_df.to_csv('svm_v1_opt_randcv_results.csv', index=False)
cv_results_df = pd.DataFrame(random_search.cv_results_)
print(cv_results_df)