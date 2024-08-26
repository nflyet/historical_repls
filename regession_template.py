# Multiple Model Regression Template

# Import Libraries

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Import the Data - Multiple Regression
filename = input("Enter the name of the file (remember to add .csv on the end): ")

dataset = pd.read_csv(filename)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split the dataset into the training set and the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Run for SVR
y2 = y.reshape(len(y), 1)
X2_train, X2_test, y2_train, y2_test = train_test_split(X, y2, 
                                                        test_size=0.2, random_state=0)


# Multiple Linear Regression

## Training the Model on the Training Set
MRregressor = LinearRegression()
MRregressor.fit(X_train, y_train)

## Predicting the Test Set Results - Linear Regression
MRy_pred = MRregressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((MRy_pred.reshape(len(MRy_pred), 1), 
                      np.array(y_test).reshape(len(y_test), 1)),1))

## Evaluating Model Performance - Multiple Regression
MR_rscore = r2_score(y_test, MRy_pred, multioutput='uniform_average')


# Polynomial Regression

## Training the Model on the Dataset - Polynomial Regression
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X_train)
PRregressor = LinearRegression()
PRregressor.fit(X_poly, y_train)

## Predicting new results
PRy_pred = PRregressor.predict(poly_reg.transform(X_test))
np.set_printoptions(precision=2)
print(np.concatenate((PRy_pred.reshape(len(PRy_pred), 1), 
                      np.array(y_test).reshape(len(y_test), 1)), 1))

## Evaluating Performance - Polynomial Regression
PR_rscore = r2_score(y_test, PRy_pred, multioutput='uniform_average')


# Support Vector Regression 

## Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
feature_X = sc_X.fit_transform(X)
feature_y = sc_y.fit_transform(y)

## Training the SVR Model
SVRregressor = SVR(kernel='rbf')
SVRregressor.fit(X2_train, y2_train)

## Predicting a New Result
SVRy_pred = sc_y.inverse_transform(
    SVRregressor.predict(sc_X.transform(X2_test)).reshape(-1, 1))
np.set_printoptions(precision=2)
print(
    np.concatenate(
        (SVRy_pred.reshape(len(SVRy_pred), 1), np.array(y_test).reshape(
            len(y2_test), 1)), 1))

## Evaluating Model - SVR Regression
SVR_rscore = r2_score(y2_test, SVRy_pred, multioutput='uniform_average')


# Decision Tree Regression

## Training the Decision Tree Regression Model
DTRregressor = DecisionTreeRegressor(random_state=0)
DTRregressor.fit(X_train, y_train)

## Predicting a New Result
DTRy_pred = DTRregressor.predict(X_test)
np.set_printoptions(precision=2)
print(
    np.concatenate(
        (DTRy_pred.reshape(len(DTRy_pred), 1), 
         np.array(y_test).reshape(len(y_test), 1)), 1))

## Evaluating the Model Performance
DTR_rscore = r2_score(y_test, DTRy_pred, multioutput='uniform_average')


# Random Forrest Regression

## Training the Random Forest Regression Model
RFRregressor = RandomForestRegressor(n_estimators=10, random_state=0)
RFRregressor.fit(X_train, y_train)

## Predicting a New Result
RFRy_pred = RFRregressor.predict(X_test)
np.set_printoptions(precision=2)
print(
    np.concatenate(
        (RFRy_pred.reshape(len(RFRy_pred), 1), 
         np.array(y_test).reshape(len(y_test), 1)), 1))

## Evaluating the Model
RFR_rscore = r2_score(y_test, RFRy_pred, multioutput='uniform_average')

# Comparing Models"""
all_models = {
    'Multiple Linear Regression': MR_rscore,
    'Polynomial Regression': PR_rscore,
    'Support Vector Regression': SVR_rscore,
    'Decision Tree Regression': DTR_rscore,
    'Random Forest Regression': RFR_rscore
}

scores = [MR_rscore, PR_rscore, SVR_rscore, DTR_rscore, RFR_rscore]
max_score = max([float(score) for score in scores])
best = []
for key in all_models:
    if all_models[key] == max_score:
        best.append(key)
print("The model with the best fit is ", best)
