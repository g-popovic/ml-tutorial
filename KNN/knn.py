import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv('car.data')

le = preprocessing.LabelEncoder()

buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
car_class = le.fit_transform(list(data['class']))

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(car_class)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(7)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print(str(score * 100) + '%')

predicted = model.predict(x_test)
names = ['unacc', 'acc', 'good', 'vgood']

for i in range(len(predicted)):
    print('Predicted:', names[predicted[i]], '| Data:', x_test[i], '| Actual:', names[y_test[i]])
