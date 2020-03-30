import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,0:1].values
y = dataset.iloc[:,1:0:-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=1/3)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred= regressor.predict(x_train)

plt.scatter(x_train, y_train, color='orange')
plt.plot(x_train, regressor.predict(x_train), color='green')
plt.title('salary vs experience (Training Set)')
plt.xlabel('experience ')
plt.ylabel('salary')
plt.show()

plt.scatter(x_test, y_test, color='orange')
plt.plot(x_train, regressor.predict(x_train), color='green')
plt.title('salary vs experience (Test Set)')
plt.xlabel('experience ')
plt.ylabel('salary')
plt.show()


