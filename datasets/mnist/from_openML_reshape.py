from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

# Laden des MNIST-Datensatzes von OpenML
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

# Umwandeln der Zielvariable in einen numerischen Typ
y = y.astype('int32')

# Ausgeben des ursprünglichen Shapes
print("Ursprünglicher Shape von X:", X.shape)
print("Ursprünglicher Shape von y:", y.shape)

# Teilen der Daten in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7, random_state=42)

# Ausgeben des Shapes nach dem Teilen
print("Shape von X_train nach dem Teilen:", X_train.shape)
print("Shape von X_test nach dem Teilen:", X_test.shape)

# Den Datensatz für ein CNN vorbereiten
X_train_cnn = X_train.values.reshape((-1, 28, 28, 1)).astype('float32') / 255
X_test_cnn = X_test.values.reshape((-1, 28, 28, 1)).astype('float32') / 255

# Ausgeben des Shapes nach der Vorbereitung für CNN
print("Shape von X_train_cnn:", X_train_cnn.shape)
print("Shape von X_test_cnn:", X_test_cnn.shape)

# One-Hot-Encoding der Zielvariablen
y_train_cnn = to_categorical(y_train, 10)
y_test_cnn = to_categorical(y_test, 10)

# Ausgeben des Shapes der Zielvariablen nach One-Hot-Encoding
print("Shape von y_train_cnn:", y_train_cnn.shape)
print("Shape von y_test_cnn:", y_test_cnn.shape)

# Den Datensatz für ein SVM vorbereiten
scaler = StandardScaler()
X_train_svm = scaler.fit_transform(X_train)
X_test_svm = scaler.transform(X_test)

# Ausgeben des Shapes nach der Vorbereitung für SVM
print("Shape von X_train_svm:", X_train_svm.shape)
print("Shape von X_test_svm:", X_test_svm.shape)
