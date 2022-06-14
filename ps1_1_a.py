import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y = True)

#Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 0]

#Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-88]
diabetes_X_test = diabetes_X[-88:]

#Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-88]
diabetes_y_test = diabetes_y[-88:]

diabetes_XTr = diabetes_X.transpose()
inverse = np.linalg.pinv(np.matmul(diabetes_XTr, diabetes_X))
Xy = np.matmul(diabetes_XTr, diabetes_y)
w = np.matmul(inverse, Xy)
print("w = ", w)
