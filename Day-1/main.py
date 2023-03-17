from sklearn.linear_model import LinearRegression
from pandas import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import numpy as np

data = read_csv('salary.csv')

print(data.head())
print(data.info())
print(data.describe())

plt.figure(figsize=(12, 6))
sns.pairplot(data, x_vars=['YearsExperience'], y_vars=['Salary'], height=7, kind='scatter')

plt.xlabel('Years')
plt.ylabel('Salary')
plt.title('Salary Prediction')
plt.show()

X = data[['YearsExperience']]
y = data[['Salary']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_addC = sm.add_constant(X_train)

model = sm.OLS(y_train, X_addC).fit()
model.summary()
print(model.rsquared, model.rsquared_adj)

X = np.array(X_train).reshape((-1, 1))
model = LinearRegression().fit(X_train, y_train)

fig = plt.figure()
ax = fig.add_subplot()

plt.scatter(X_train, y_train, color='g')
plt.scatter(X_test, y_test, color='r')

plt.plot(X_train, model.predict(X_train), color='r')
ax.text(1, 4, r'LR equation: $Y = a + bX$', fontsize=10)

print(model.coef_[0], model.intercept_)
# Predicting the Salary for the Test values
X_test_c = sm.add_constant(X_test)

y_predict = model.predict(X_test_c)

X_test['Prediction'] = y_predict
X_test['y_test'] = y_test

print(X_test)
