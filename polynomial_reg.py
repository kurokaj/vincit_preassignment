from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures


def train_poly(data, degree):
    # Split to features and variables
    X = data[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
                "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]]
    Y = data['quality']

    # Divide to training set 70% and test set 30%
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    # First we transform the features into polynomial space and then generate linear reg. model
    poly_f = PolynomialFeatures(degree=degree)

    X_train_p = poly_f.fit_transform(X_train)

    model1 = LinearRegression()
    model1.fit(X_train_p,Y_train)

    # Calculate the score
    score1 = model1.score(poly_f.fit_transform(X_test),Y_test)
    print("The train/test-set score: ",score1)
