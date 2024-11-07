import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE

df = pd.read_csv('breast_cancer.csv', header=0)

x = df.drop('Status', axis=1)
y = df[['Status']]

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
x['differentiate']= label_encoder.fit_transform(x['differentiate'])
x['Grade']= label_encoder.fit_transform(x['Grade'])
x_encoded = pd.get_dummies(x, columns=['Race','Marital Status','T Stage','N Stage','6th Stage','A Stage',
                                       'Estrogen Status','Progesterone Status'])
y_encoded = pd.Series(label_encoder.fit_transform(y['Status'])) 


x_train, x_test, y_train, y_test = train_test_split(x_encoded, y_encoded, random_state=42)

#Data is imbalanced, so we use SMOTE to balance it
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
x_test_resampled, y_test_resampled = smote.fit_resample(x_test, y_test)

#Decision Tree Classifier where data was resampled using SMOTE
#Following tree has been fitted on resampled data and the corresponding confusion matrix is plotted
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt = clf_dt.fit(x_train_resampled, y_train_resampled)

plt.figure(figsize=(15, 7.5))
plot_tree(clf_dt, filled=True, rounded=True, 
          class_names=["Alive", "Dead"],
          feature_names=x_encoded.columns)
          
#The results in this confusion matrix are much better than the one using the original data
#Tree can be further improved through cost complexity pruning
y_pred = clf_dt.predict(x_test_resampled)
cm = confusion_matrix(y_test_resampled, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Alive", "Dead"])
disp.plot(cmap=plt.cm.Blues)

print("After SMOTE: ")
print("Accuracy: ",accuracy_score(y_test_resampled, y_pred))
print("Precision: ", precision_score(y_test_resampled, y_pred))

plt.show()