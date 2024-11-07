import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score
from sklearn import preprocessing

df = pd.read_csv('breast_cancer.csv', header=0)

# data is clean according to kaggle source, so gonna skip the preprocessing
# https://www.kaggle.com/code/mirzahasnine/breast-cancer-predection/notebook#Detecting-Missing-Values

#independent; X; used to make classifications
x = df.drop('Status', axis=1)

#dependent; Y; data we want to predict
y = df[['Status']]

## encoding
## x: race ; marital ; t stage; n stage; 6th stage;  a stage; estrogen; progesterone , these are one hot encoded
## x: grade ; differentiate , these are ordinal columns hence are label encoded
## y: status , this is label encoded as it is binary
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

## splitting data
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y_encoded, random_state=42)

## initial decision tree with no pruning
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt = clf_dt.fit(x_train, y_train)

# plt.figure(figsize=(15, 7.5))
# plot_tree(clf_dt, filled=True, rounded=True, 
#           class_names=["Alive", "Dead"],
#           feature_names=x_encoded.columns)

# y_pred = clf_dt.predict(x_test)
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Alive", "Dead"])
# disp.plot(cmap=plt.cm.Blues)

# print("Before CCP: ")
# print(accuracy_score(y_test, y_pred))
# print(precision_score(y_test, y_pred))
# print(" ")


## cost complexity pruning
path = clf_dt.cost_complexity_pruning_path(x_train,y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]   

# clf_dts = [] 
# for ccp_alpha in ccp_alphas:
#     clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
#     clf_dt.fit(x_train,y_train)
#     clf_dts.append(clf_dt)

# train_scores = [clf_dt.score(x_train,y_train) for clf_dt in clf_dts]
# test_scores = [clf_dt.score(x_test,y_test) for clf_dt in clf_dts]

# fig, ax = plt.subplots()
# ax.set_xlabel("alpha")
# ax.set_ylabel("accuracy")
# ax.set_title("accuracy vs alpha for training and testing sets")
# ax.plot(ccp_alphas, train_scores, marker='o', label="train", drawstyle="steps-post")
# ax.plot(ccp_alphas, test_scores, marker='o', label="test", drawstyle="steps-post")
# ax.legend()

## ideal alpha using cross validation
alpha_loop_values = [] 
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    scores = cross_val_score(clf_dt, x_train, y_train)
    alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])

alpha_results = pd.DataFrame(alpha_loop_values, columns=['alpha','mean_accuracy', 'std'])
# alpha_results.plot(x='alpha',
#                    y='mean_accuracy',
#                    yerr='std',
#                    marker = 'o',
#                    linestyle = '--')

## 0.003>alpha>0.0028 is the ideal range

ideal_ccp_alpha_series = alpha_results[(alpha_results['alpha']>0.0028) 
                                       & (alpha_results['alpha']<0.0030)]['alpha']  
ideal_ccp_alpha = float(ideal_ccp_alpha_series.iloc[0])   ## 0.0029329289682066585

## pruned tree on test data
clf_dt_pruned = DecisionTreeClassifier(random_state=42, ccp_alpha=ideal_ccp_alpha)
clf_dt_pruned = clf_dt_pruned.fit(x_test,y_test)

plt.figure(figsize=(15, 7.5))
plot_tree(clf_dt_pruned, filled=True, rounded=True, 
          class_names=["Alive", "Dead"],
          feature_names=x_encoded.columns)

y_pred_pruned = clf_dt_pruned.predict(x_test)
cm = confusion_matrix(y_test, y_pred_pruned)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Alive", "Dead"])
disp.plot(cmap=plt.cm.Blues)

print("After CCP: ")
print("Accuracy", accuracy_score(y_test, y_pred_pruned))
print("Precision", precision_score(y_test, y_pred_pruned))

plt.show()


