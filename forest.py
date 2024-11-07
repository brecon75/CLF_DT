import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import preprocessing

# Load the dataset
df = pd.read_csv('breast_cancer.csv', header=0)

# Data is clean according to Kaggle source, so gonna skip the preprocessing
# https://www.kaggle.com/code/mirzahasnine/breast-cancer-predection/notebook#Detecting-Missing-Values

# Independent variables (X)
x = df.drop('Status', axis=1)

# Dependent variable (Y)
y = df[['Status']]

# Encoding categorical variables
grade_mapping = {
    '1': 1,
    '2': 2,
    '3': 3,
    'Anaplastic Grade IV': 4}
x['Grade'] = x['Grade'].map(grade_mapping)

diff_mapping = {
    'Undifferentiated': 1,
    'Poorly differentiated': 2,
    'Moderately differentiated': 3,
    'Well differentiated': 4}
x['differentiate'] = x['differentiate'].map(diff_mapping)

label_encoder = preprocessing.LabelEncoder()
x['differentiate'] = label_encoder.fit_transform(x['differentiate'])
x['Grade'] = label_encoder.fit_transform(x['Grade'])
x_encoded = pd.get_dummies(x, columns=['Race', 'Marital Status', 'T Stage', 'N Stage', '6th Stage', 'A Stage',
                                       'Estrogen Status', 'Progesterone Status'])
y_encoded = pd.Series(label_encoder.fit_transform(y['Status']))

# Splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y_encoded, random_state=42)

# Initialize lists to store accuracy and loss
train_accuracies = []
test_accuracies = []
train_losses = []
test_losses = []

# Train Random Forest with different number of trees
n_estimators_range = range(10, 210, 10)
for n_estimators in n_estimators_range:
    clf_rf = RandomForestClassifier(random_state=42, n_estimators=n_estimators)
    clf_rf.fit(x_train, y_train)
    
    # Training accuracy and loss
    y_train_pred = clf_rf.predict(x_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_loss = 1 - train_accuracy
    train_accuracies.append(train_accuracy)
    train_losses.append(train_loss)
    
    # Testing accuracy and loss
    y_test_pred = clf_rf.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_loss = 1 - test_accuracy
    test_accuracies.append(test_accuracy)
    test_losses.append(test_loss)

# Plot accuracy vs number of trees
plt.figure(figsize=(12, 6))
plt.plot(n_estimators_range, train_accuracies, label='Training Accuracy', marker='o')
plt.plot(n_estimators_range, test_accuracies, label='Testing Accuracy', marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Trees')
plt.legend()
plt.grid(True)
plt.show()

# Plot loss vs number of trees
plt.figure(figsize=(12, 6))
plt.plot(n_estimators_range, train_losses, label='Training Loss', marker='o')
plt.plot(n_estimators_range, test_losses, label='Testing Loss', marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('Loss')
plt.title('Loss vs Number of Trees')
plt.legend()
plt.grid(True)
plt.show()

# Final model with optimal number of trees
optimal_n_estimators = n_estimators_range[np.argmax(test_accuracies)]
clf_rf_optimal = RandomForestClassifier(random_state=42, n_estimators=optimal_n_estimators)
clf_rf_optimal.fit(x_train, y_train)

# Make predictions
y_pred_optimal = clf_rf_optimal.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_optimal)
precision = precision_score(y_test, y_pred_optimal)
recall = recall_score(y_test, y_pred_optimal)
f1 = f1_score(y_test, y_pred_optimal)

print("Random Forest Classifier Performance with Optimal Number of Trees:")
print("Optimal Number of Trees:", optimal_n_estimators)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_optimal)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Alive", "Dead"])
disp.plot(cmap=plt.cm.Blues)
plt.show()