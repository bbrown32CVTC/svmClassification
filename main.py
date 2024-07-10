# This is a Python script for a SVM Classification.

import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Create the path to the dataset and set dataframe to the data
path_to_file = "../CSC 419 HW 3 Spambase 1000.csv"
df = pd.read_csv(path_to_file)

# # Test 1
# # Create X and y dataframes
# y = df['spam']
# X = df[['capital_run_length_longest', 'capital_run_length_total']]
#
# # Create train_test_split of 70% training and 30% testing data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# # Create SVC Model with a linear kernel and a C value of 1.0(Default)
# svc = svm.SVC(kernel='linear')
#
# # Train the model
# svc.fit(X_train, y_train)
#
# # Classify the data in the testing set using model.predict
# y_pred = svc.predict(X_test)
#
# # Evaluate the accuracy of the model using accuracy_score
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# print("Accuracy Percent:", round(accuracy*100, 2), "%")


# Test 2
# Create X and y dataframes
y = df['spam']
X = df[['capital_run_length_longest', 'capital_run_length_total']]

# Create train_test_split of 70% training and 30% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create SVC Model with a linear kernel and a C value of 500.0
svc = svm.SVC(C=50.0, kernel='linear')

# Train the model
svc.fit(X_train, y_train)

# Classify the data in the testing set using model.predict
y_pred = svc.predict(X_test)

# Evaluate the accuracy of the model using accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Accuracy Percent:", round(accuracy*100, 2), "%")


# # Test 3
# # Create X and y dataframes
# y = df['spam']
# X = df[['capital_run_length_longest', 'capital_run_length_total', 'char_freq_!']]
#
# # Create train_test_split of 70% training and 30% testing data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# # Create SVC Model with a linear kernel and a C value of 1.0(Default)
# svc = svm.SVC(kernel='linear')
#
# # Train the model
# svc.fit(X_train, y_train)
#
# # Classify the data in the testing set using model.predict
# y_pred = svc.predict(X_test)
#
# # Evaluate the accuracy of the model using accuracy_score
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# print("Accuracy Percent:", round(accuracy*100, 2), "%")
