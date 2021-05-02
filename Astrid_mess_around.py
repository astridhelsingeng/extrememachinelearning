import pandas as pd

df = pd.read_csv("/Users/astridhelsingeng/Documents/Dropbox/Dokumenter Astrid/CBS/4 klasse høst/Pycharm/MachineLearning/extrememachinelearning/cleaned_data.csv", header=0)

print(df)

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

X = df.drop("target",axis=1).values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

mlp = MLPClassifier(hidden_layer_sizes=(10), activation='relu', solver='adam', max_iter=100)
mlp.fit(X_train,y_train)
predict = mlp.predict(X_test)

#Perfrom a gridSearch looking at hyperparameters and evaluationg what is best for our model.
hidden_layer = [12,(10,10),(12,12),(10,10,10),(50,50),(100,100)]

check_parameters = {
    'hidden_layer_sizes': hidden_layer,
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],}

gridsearchcv = GridSearchCV(mlp, check_parameters, n_jobs=-1, cv=3)
gridsearchcv.fit(X_train, y_train)

print('Best parameters found:\n', gridsearchcv.best_params_)
from sklearn.metrics import classification_report

print('Results on the test set:')
print(classification_report(y_test, predict))

"""from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predict))
plot_confusion_matrix(mlp, X_test,y_test)
plt.show()
"""