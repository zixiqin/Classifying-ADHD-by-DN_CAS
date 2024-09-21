import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, make_scorer
import graphviz


dncas = pd.read_csv('ADHD+TD_dncas.csv')
cas_train_set, cas_test_set = train_test_split(dncas, test_size=0.2, random_state=1)

train = cas_train_set.iloc[:, 3:]
test = cas_test_set.iloc[:, 3:]

X_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]
X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]


# Convert labels to binary encoding, assuming "ADHD" as the positive class and "Normal" as the negative class
y_train = y_train.map({'ADHD': 1, 'TD': 0})
y_test = y_test.map({'ADHD': 1, 'TD': 0})

# Define the model and parameter grid
dt = DecisionTreeClassifier(random_state=42)
param_grid = {
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Define the evaluation metrics
scoring = {
    'accuracy': 'accuracy',
    'sensitivity': make_scorer(recall_score, pos_label=1),
    'specificity': make_scorer(recall_score, pos_label=0)
}

# Use grid search with 5-fold cross-validation to find the best parameters
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring=scoring, refit='accuracy', return_train_score=True)
grid_search.fit(X_train, y_train)

# Output the best parameters
print(f"Best parameters found: {grid_search.best_params_}")

# Evaluate the model on the test set using the best parameters
best_dt = grid_search.best_estimator_
y_pred = best_dt.predict(X_test)

# Calculate accuracy, specificity, and sensitivity
accuracy = accuracy_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred, pos_label=1)
specificity = recall_score(y_test, y_pred, pos_label=0)

print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

# Display the confusion matrix

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# Export the decision tree
dot_data = export_graphviz(best_dt, out_file=None, 
                           feature_names=X_train.columns,
                           class_names=['TD', 'ADHD'],
                           filled=True, rounded=True, 
                           special_characters=True,node_ids=True)  
graph = graphviz.Source(dot_data)  
graph.render("decision_tree_dncas") 
graph.view()  







X_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]
X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]

# Convert labels to binary encoding, assuming "ADHD" as the positive class and "Normal" as the negative class
y_train = y_train.map({'ADHD': 1, 'TD': 0})
y_test = y_test.map({'ADHD': 1, 'TD': 0})

# Define the model and parameter grid
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8]
}

# Define the evaluation metrics
scoring = {
    'accuracy': 'accuracy',
    'sensitivity': make_scorer(recall_score, pos_label=1),
    'specificity': make_scorer(recall_score, pos_label=0)
}

# Use grid search with 5-fold cross-validation to find the best parameters
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring=scoring, refit='accuracy', return_train_score=True)
grid_search.fit(X_train, y_train)

# Output the best parameters
print(f"Best parameters found: {grid_search.best_params_}")

# Evaluate the model on the test set using the best parameters
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# Calculate accuracy, specificity, and sensitivity
accuracy = accuracy_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred, pos_label=1)
specificity = recall_score(y_test, y_pred, pos_label=0)

print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

# Display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)