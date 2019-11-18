from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn import metrics

def SVM_train(data):

    # The data after feature selection; remove(1,3,7)
    #data = data.drop(columns="fixed acidity")
    #data = data.drop(columns="citric acid")
    #data = data.drop(columns="total sulfur dioxide")


    # split to features and variables after feature selection
    #X = data[["volatile acidity","residual sugar","chlorides",
    #            "free sulfur dioxide","density","pH","sulphates","alcohol"]]
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

    # Print the importance of the features by ExtraTreeClassifier
    # The data is same for every model so we need to run this once
    #model = ExtraTreesClassifier()
    #model.fit(X_train, Y_train)
    #print("The importance of the features: ")
    #print(model.feature_importances_)


    # Train the model
    model1 = svm.SVC(gamma='scale', kernel='linear')
    model1.fit(X_train,Y_train)

    # The feature selection (RFE)
    #select = RFE(model1, 9, step=1)
    #select = select.fit(X_train, Y_train)
    #print("The feature selection: ")
    #print(select.ranking_)

    score1 = model1.score(X_test,Y_test)
    print("The train/test-set score: ",score1)

    # Cross-validation with K = 10-folds
    k = 10
    # Create the model
    model2 = svm.SVC(gamma='scale', kernel='linear')
    # get the score from the crossvalidation
    score2 = cross_val_score(model2, X, Y, cv=k)
    print("The Crossvalidation score: ",(sum(score2) / k))
