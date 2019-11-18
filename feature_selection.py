import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def lasso_r(data):

    # Perform the feature analysis on linear regression model

    X = data[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
                "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]]
    Y = data['quality']

    # Divide to training set 70% and test set 30%
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # Train Lasso regression with multiple alpha values
    lasso1 = Lasso(alpha=0.1, max_iter=1000000)
    lasso1.fit(X_train, Y_train)

    # The score of testing
    t_score = lasso1.score(X_test, Y_test)
    # Number of features used
    coeff_num = np.sum(lasso1.coef_!=0)

    print("Training score with Lasso regression (alpha=1): ",t_score)
    print("Number of coefficient used: ", coeff_num)

    lasso2 = Lasso(alpha=0.01, max_iter= 1000000)
    lasso2.fit(X_train, Y_train)

    # The score of testing
    t_score2 = lasso2.score(X_test, Y_test)
    # Number of features used
    coeff_num2 = np.sum(lasso2.coef_!=0)

    print("Training score with Lasso regression (alpha=0.01): ",t_score2)
    print("Number of coefficient used: ", coeff_num2)

    lasso3 = Lasso(alpha=0.00001, max_iter= 1000000)
    lasso3.fit(X_train, Y_train)

    # The score of testing
    t_score3 = lasso3.score(X_test, Y_test)
    # Number of features used
    coeff_num3 = np.sum(lasso3.coef_!=0)

    print("Training score with Lasso regression (alpha=0.00001): ",t_score3)
    print("Number of coefficient used: ", coeff_num3)
