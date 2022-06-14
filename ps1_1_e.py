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

x = np.array(diabetes_X_train).flatten()
y = np.array(diabetes_y_train).flatten()
a = 0
b = 0
iteration = 600
N = len(x)
learning_rate = 0.07

for i in range(iteration):
    h_x = (a * x) + b

    emprical_risk = (1 / N) * sum([val ** 2 for val in (y - h_x)])

    derivative_a = -(2 / N) * sum(x * (y - h_x))
    derivative_b = -(2 / N) * sum(y - h_x)

    a = a - learning_rate * derivative_a
    b = b - learning_rate * derivative_b

h_x = (a * diabetes_X_test) + b
plt.figure(4)
plt.scatter(diabetes_X_test, diabetes_y_test)
plt.plot(diabetes_X_test, h_x)

print("R^2 Score: %.2f" % r2_score(diabetes_y_test, h_x))

plt.show()