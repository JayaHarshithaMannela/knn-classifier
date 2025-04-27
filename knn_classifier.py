import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

columns = [
    'fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym',
    'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class'
]
data = pd.read_csv('magic04.data', header=None, names=columns)

X = data.drop('class', axis=1).values
y = data['class'].values

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
