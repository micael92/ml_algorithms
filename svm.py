from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, classification_report

import numpy as np
from sklearn import svm

# Load dataset
mnist = fetch_openml('mnist_784')

#print(mnist.data.shape)
#print(mnist.data)

print(type(mnist.target))
