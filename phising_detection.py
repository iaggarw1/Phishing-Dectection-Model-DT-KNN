import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
col_names = ['sender', 'receiver', 'date', 'subject', 'body', 'url', 'label']
dataset1 = pd.read_csv('Nazario_5.csv', header=None, names=col_names)
#dataset2 = pd.read_csv('Nazario_7_3.xlsx', header=None, names=col_names)

onehot = OneHotEncoder(handle_unknown='ignore')
onehot_transformed = pd.DataFrame(onehot.fit_transform(dataset1[['sender', 'receiver', 'date', 'subject', 'body', 'url']]).toarray())
dataset1 = pd.concat([dataset1, onehot_transformed], axis=1)
dataset1.drop(['sender', 'receiver', 'date', 'subject', 'body', 'url'], axis=1, inplace=True)

X = dataset1.iloc[:, 1:]
y = dataset1.iloc[:, :1]

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a StratifiedKFold object
strat_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Use the object to split your data
for train_index, test_index in strat_kfold.split(X, y.values.ravel()):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Decision Tree
decision_tree = DecisionTreeClassifier(random_state=42)

# KNN
knn = KNeighborsClassifier(n_neighbors=13)

# Define the base models
base_models = [('decision_tree', decision_tree), ('knn', knn)]

# Use a logistic regression as a meta-model
meta_model = LogisticRegression()

# Stacking Ensemble
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)

# Train the stacking model
stacking_model.fit(X_train, y_train.values.ravel())

# Make predictions on the test set
y_pred = stacking_model.predict(X_test)

# Assuming y_test and y_pred are your target test set and predictions respectively
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=1)  # or 'weighted'
recall = recall_score(y_test, y_pred, average='macro', zero_division=1)  # or 'weighted'
f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)  # or 'weighted'

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)