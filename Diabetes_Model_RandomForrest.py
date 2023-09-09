import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
col_names = ['pregnancy', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
df = pd.read_csv(r"C:\diabetes.csv", header= None, names=col_names)
data = df.drop([0])
feature_cols = ['pregnancy', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
X = data[feature_cols]
y = data.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
classifier = RandomForestClassifier(n_estimators=50)
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
print(accuracy_score(y_test, pred))
