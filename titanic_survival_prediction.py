import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv(r"C:\Users\ragin\OneDrive\Documents\train.csv")


data.head()

data.isnull().sum()

data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data.drop('Cabin', axis=1, inplace=True)

data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'S':0, 'C':1, 'Q':2})

data['FamilySize'] = data['SibSp'] + data['Parch']

sns.countplot(x='Survived', data=data)
plt.show()

sns.barplot(x='Sex', y='Survived', data=data)
plt.show()

X = data[['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize']]
y = data['Survived']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

import pandas as pd

data = pd.read_csv(r"C:\Users\ragin\OneDrive\Documents\train.csv")

print(data.head())