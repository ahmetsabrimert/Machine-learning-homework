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


regr = linear_model.LinearRegression()


regr.fit(diabetes_X_train, diabetes_y_train)


diabetes_y_pred = regr.predict(diabetes_X_test)


print("Coefficients: \n", regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

plt.scatter(diabetes_X_test, diabetes_y_test)
plt.plot(diabetes_X_test, diabetes_y_pred,linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()