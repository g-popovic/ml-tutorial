import pandas as pd
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Read the data from the .csv file
data = pd.read_csv('student-mat.csv', sep=';')

# Select the features (properties) we want to use for our model
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

predict = 'G3'

# Create 2 numpy arrays:
# x - The dataset/array WITHOUT the label (label = the thing we're trying to predict)
# y - The dataset/array ONLY containing the label
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# Split the arrays into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

"""

best = 0

for __ in range(1000):
    # Split the arrays into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    # Instantiate out model
    model = linear_model.LinearRegression()

    # Train out model
    model.fit(x_train, y_train)

    # Test our models accuracy on the test sets
    acc = model.score(x_test, y_test)

    if acc > best:
        # Save our model as a pickle file
        best = acc
        with open('student_model.pickle', 'wb') as f:
            pickle.dump(model, f)
+
print(best)

"""

# Load the model from the pickle file
pickle_in = open('student_model.pickle', 'rb')
model = pickle.load(pickle_in)

predictions = model.predict(x_test)

for x in range(len(predictions)):
    print(round(predictions[x], 3), y_test[x])

# Plot the data with matplotlib
feature = 'G1'
style.use('ggplot')
pyplot.scatter(data[feature], data['G2'], data[predict])

pyplot.xlabel("Feature: " + feature)
pyplot.ylabel("Final Grade")

pyplot.show()