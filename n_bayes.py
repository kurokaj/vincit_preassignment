from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

def nbayes(data):
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

    # Naive bayes classification with train and test set
    m = GaussianNB()
    model1 = m.fit(X_train,Y_train)
    score1 = model1.score(X_test,Y_test)
    print("The train/test-set score: ",score1)

    # Cross-validation with K = 10-folds
    k = 10
    # Create the model
    model2 = GaussianNB()
    # get the score from the crossvalidation
    score2 = cross_val_score(model2, X, Y, cv=k)
    print("The Crossvalidation score: ",(sum(score2) / k))
