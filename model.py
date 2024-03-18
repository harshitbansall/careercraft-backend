import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

c_dataset = pd.read_csv('careercraft.csv')
c_dataset.head()
c_dataset.isnull().sum()
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(c_dataset['TARGET'], bins=30)
plt.show()
correlation_matrix = c_dataset.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)
plt.figure(figsize=(20, 5))

features = ['IK','SC']
target = c_dataset['TARGET']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = c_dataset[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('target')

X = pd.DataFrame(np.c_[c_dataset['IK'], c_dataset['SC']], columns = ['IK','SC'])
Y = c_dataset['TARGET']

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

from sklearn.metrics import r2_score

y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

tree_model = DecisionTreeRegressor()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


tree_model.fit(X_train, y_train)


y_train_predict = tree_model.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predict))
r2_train = r2_score(y_train, y_train_predict)

print("The model performance for the training set")
print("--------------------------------------")
print("RMSE is {}".format(rmse_train))
print("R2 score is {}".format(r2_train))
print("\n")

# Model evaluation for testing set
y_test_predict = tree_model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predict))
r2_test = r2_score(y_test, y_test_predict)

print("The model performance for the testing set")
print("--------------------------------------")
print("RMSE is {}".format(rmse_test))
print("R2 score is {}".format(r2_test))

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_test_predict)
rmse = np.sqrt(mse)


r2 = r2_score(y_test, y_test_predict)

print("Mean Squared Error (MSE): {:.2f}".format(mse))
print("Root Mean Squared Error (RMSE): {:.2f}".format(rmse))

import matplotlib.pyplot as plt

# Calculate residuals
residuals = y_test - y_test_predict

# Create a residual plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test_predict, residuals, c='b', marker='o', s=10)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (Actual - Predicted)")
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residual Plot")
plt.show()

# Assuming y_test_predict contains the predicted prices
predicted_prices = y_test_predict

# Define your price categories and their corresponding labels
price_categories = {
    "Management Consultance": (0, 5),
    "Contractor": (5, 10),  
    "Aerospace Engineers": (10, 15),
    "Lawyers": (15, 20),
    "cybersecurity": (20, 25),
    "Machine Learning": (25, 30),
    "Data Scientist": (30, 35),
    "BLockchain Developer": (35, 40),
    "Pharmacists": (40, 45),
    "Marketing Managers": (45, 50),
    "Financial Managers": (50, float("inf"))  # Example range for "High" price
}

# Create an empty list to store the labels
price_labels = []

# Classify the predicted prices based on the defined categories
for price in predicted_prices:
    for label, (min_range, max_range) in price_categories.items():
        if min_range <= price < max_range:
            price_labels.append(label)
            break
    else:
        price_labels.append("Unknown")

# Now 'price_labels' contains the category labels for each predicted price
print(price_labels)
