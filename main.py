import pandas as pd
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import linear_model

data = pd.read_csv('./student-mat.csv', sep=';')

data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

predict = 'G3'

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

model = linear_model.LinearRegression()

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

print(acc)

print('Coeficient: ', model.coef_)
print('Intercept: ', model.intercept_)