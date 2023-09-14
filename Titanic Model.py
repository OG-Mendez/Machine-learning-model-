import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
Col_names = ['ID', 'Survived', 'Class', 'Name', 'Sex', 'Age', 'Siblings', 'Parents', 'Ticket', 'Fare', 'Cabin', 'Embarked']
head_names = ['ID', 'Class', 'Name', 'Sex', 'Age', 'Siblings', 'Parents', 'Ticket', 'Fare', 'Cabin', 'Embarked']
data1 = pd.read_csv(r"C:\train.csv", header=None, names= Col_names)
data2 = pd.read_csv(r"C:\test.csv", names= head_names)
train_data = data1.drop([0])
test_data = data2.drop([0])
feature_names = [ 'Class', 'Sex', 'Age', 'Parents', 'Embarked']
combine = [train_data, test_data]
train_data['Age'] = train_data['Age'].fillna(41)
test_data['Age'] = test_data['Age'].fillna(41)
common_value = 'S'
train_data['Embarked'] = train_data['Embarked'].fillna(common_value)
test_data['Embarked'] = test_data['Embarked'].fillna(common_value)
le = LabelEncoder()
train_data['Sex'] = le.fit_transform(train_data['Sex'])
train_data['Embarked'] = le.fit_transform(train_data['Embarked'])
train_data['Class'] = le.fit_transform(train_data['Class'])
test_data['Sex'] = le.fit_transform(test_data['Sex'])
test_data['Embarked'] = le.fit_transform(test_data['Embarked'])
test_data['Class'] = le.fit_transform(test_data['Class'])
X_train = train_data[feature_names]
y_train = train_data.Survived
X_test = test_data[feature_names]
classifier = RandomForestClassifier(n_estimators=50)
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
acc_log = round(classifier.score(X_train, y_train) * 100, 2)
print("Accuracy of model: ", acc_log)
submission = pd.DataFrame({
    "PassengerId" : test_data['ID'], "Survived": pred
})

filename = 'Titanic Prediction RandomForrest.csv'

submission.to_csv(filename, index=False)