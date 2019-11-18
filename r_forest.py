import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   #Data visualisation libraries
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import permutation_test_score
from sklearn import metrics

def rforest(data):

    # The data after feature selection; remove(1,3,9)
    #data = data.drop(columns="fixed acidity")
    #data = data.drop(columns="citric acid")
    #data = data.drop(columns="pH")

    # split to features and variables after feature selection
    #X = data[["volatile acidity","residual sugar","chlorides",
    #            "free sulfur dioxide","total sulfur dioxide","density","sulphates","alcohol"]]
    #Y = data['quality']

    # split to features and variables
    X = data[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
                "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]]
    Y = data['quality']

    # divide to training set 70% and test set 30%
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # Normal linear regression with train and test set
    m = RandomForestClassifier(n_estimators=50, criterion = 'entropy', random_state = 42)
    model1 = m.fit(X_train,Y_train)
    score1 = model1.score(X_test,Y_test)
    print("The train/test-set score: ",score1)

    # Create confusion matrix for the classifier
    pred = model1.predict(X_test)
    CM = metrics.confusion_matrix(Y_test, pred ,labels=[1,2,3])
    print(CM)

    # The feature selection (RFE)
    #select = RFE(model1, 7, step=1)
    #select = select.fit(X_train, Y_train)
    #print("The feature selection: ")
    #print(select.ranking_)


    # Cross-validation with K = 10-folds
    k = 10
    # Create the model
    model2 = RandomForestClassifier(n_estimators=50, criterion = 'entropy', random_state = 42)
    # get the score from the crossvalidation
    score2 = cross_val_score(model2, X, Y, cv=k)
    print("The Crossvalidation score: ",(sum(score2) / k))

    # The permutation analysis
    score, perm_scores, pvalue = permutation_test_score(model2, X, Y, scoring="accuracy", cv=k, n_permutations=100, n_jobs=1)
    print("The score of cross validation" ,score)
    print("The scores of permutation analysis", perm_scores)
    print("The pvalue of the analysis", pvalue)

    #Draw the permutation histogram
    plt.hist(perm_scores, 20, label='Permutation scores')
    # Get the Y-limit
    ylimit = plt.ylim()
    # Plot our own classifier score
    plt.plot(2 * [score], ylimit, '--g', linewidth=2)
    plt.ylim(ylimit)
    plt.legend()
    plt.xlabel('Score')
    plt.show()
