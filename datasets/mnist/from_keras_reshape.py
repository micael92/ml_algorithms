from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. MNIST Datensatz aus keras laden
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Das Shape des Datensatzes ausgeben
print("CNN Shape des Trainingsdatensatzes:", x_train.shape)
print("CNN Shape des Testdatensatzes:", x_test.shape)

# 3. Den Datensatz für ein CNN vorbereiten
# Normalisieren der Pixelwerte von 0 bis 255 zu 0 bis 1
x_train_cnn = x_train.astype('float32') / 255.0
x_test_cnn = x_test.astype('float32') / 255.0
# Hinzufügen einer Dimension für den Farbkanal
x_train_cnn = x_train_cnn.reshape((x_train_cnn.shape[0], 28, 28, 1))
x_test_cnn = x_test_cnn.reshape((x_test_cnn.shape[0], 28, 28, 1))
# Umwandeln der Zielvariablen in kategorische (One-Hot-Encoding)
y_train_cnn = to_categorical(y_train, 10)
y_test_cnn = to_categorical(y_test, 10)

# 4. Den Datensatz für ein SVM vorbereiten
# Flatten der Bildmatrizen für die SVM (von 28x28 zu 784)
x_train_svm = x_train.reshape((x_train.shape[0], -1))
x_test_svm = x_test.reshape((x_test.shape[0], -1))
# Standardisierung der Feature-Werte
scaler = StandardScaler()
x_train_svm = scaler.fit_transform(x_train_svm)
x_test_svm = scaler.transform(x_test_svm)

# 5. Das Shape des Datensatzes ausgeben
print("SVM Shape des Trainingsdatensatzes:", x_train_svm.shape)
print("SVM Shape des Testdatensatzes:", x_test_svm.shape)

print("Vorbereitung abgeschlossen.")
