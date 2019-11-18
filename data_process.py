import numpy as np
import pandas as pd

def load_data(file_name):

    # Load the table from CSV file
    data = []

    colnames = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
                "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]

    data = pd.read_csv(file_name, names=colnames,  sep=';')

    # Let's determine the usability of the samples
    # 0 <= quality < 5 -> unusable (1)
    # 6 <= quality <= 7 -> cheaper set (2)
    # 8 < quality <= 10 -> expensive set (3)

    for i in range(0,len(data['quality'])):
        if data['quality'][i] <= 5 and data['quality'][i] >= 0:
            data.at[i, 'quality'] = 1
        elif data['quality'][i] <= 7 and data['quality'][i] > 5:
            data.at[i, 'quality'] = 2
        else:
            data.at[i, 'quality'] = 3

    data.head()
    data.info()
    data.describe()
    data.columns


    # return the created dictionary
    return data
