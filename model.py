# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('IRIS.csv')

dataset['sepal.length'].fillna(0, inplace=True)

dataset['sepal.width'].fillna(0, inplace=True)

dataset['petal.length'].fillna(0, inplace=True)

dataset['petal.width'].fillna(0, inplace=True)

X = dataset.iloc[:, :3]


#Converting words to integer values
def convert_to_int(word):
    word_dict = {'Setosa':1, 'Versicolor':2, 'Virginica':3, 0: 0}
    return word_dict[word]

y = dataset.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))