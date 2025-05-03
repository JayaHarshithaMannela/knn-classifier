import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the training and testing data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Handle missing values
train['Age'] = train['Age'].fillna(train['Age'].mean())  # Fill missing 'Age' with mean
test['Age'] = test['Age'].fillna(test['Age'].mean())  # Fill missing 'Age' with mean

train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])  # Fill missing 'Embarked' with mode
test['Embarked'] = test['Embarked'].fillna(test['Embarked'].mode()[0])  # Fill missing 'Embarked' with mode

test['Fare'] = test['Fare'].fillna(test['Fare'].mean())  # Fill missing 'Fare' with mean

# Encode categorical variables
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})

train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Drop unnecessary columns
train.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)
test.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)

# Prepare training and validation data
x = train.drop(['Survived', 'PassengerId'], axis=1)
y = train['Survived']
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_val)

# Evaluate the model
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))

# Plot Confusion Matrix
sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Make predictions on the test set
x_test = test.drop(['PassengerId'], axis=1)
test_pred = model.predict(x_test)

# Create the submission file
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': test_pred})
submission.to_csv("submission.csv", index=False)

print("Submission file created: submission.csv")
