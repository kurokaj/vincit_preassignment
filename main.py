import data_process
import linear_reg
import polynomial_reg
import Logical_reg
import k_means
import r_forest
import feature_selection
import n_bayes
import SVM
import numpy as np
import matplotlib.pyplot as plt   #Data visualisation libraries
import seaborn as sns
import os



def main():
    # Find the file from the rigth folder
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    rel_path = "data/winequality-white.csv"
    abs_file_path = os.path.join(script_dir, rel_path)
    file_name = abs_file_path

    # Get the pandas dict-object
    data = data_process.load_data(file_name)

    # Visualize the table
    print(data)

    # Is there linear correlation?
    # -1 and 1 high linear correlation, 0 no linear correlation
    #cor = data.corr()
    #sns.heatmap(cor)

    # Train Lasso regression to identify unnecessary features for linear regression
    #feature_selection.lasso_r(data)

    #-----------------------------------------------------------------------------------------------------------------------------------------------------------
    # Beneath all functions to create different Regression and Classification models

    # Train linear regression with cross validation
    #linear_reg.train_linear(data)

    # Train polynomial regression
    #polynomial_reg.train_poly(data, 2)

    # Train logical Regression
    #Logical_reg.train_logical(data)

    #Train k-mean classifier
    #k_means.class_kmeans(data)

    #Train RandomForest Classifier
    #r_forest.rforest(data)

    #Train RandomForest Classifier
    #n_bayes.nbayes(data)

    #Train RandomForest Classifier
    #SVM.SVM_train(data)

    # Show plots
    plt.show()

if __name__ == "__main__":
    main()
