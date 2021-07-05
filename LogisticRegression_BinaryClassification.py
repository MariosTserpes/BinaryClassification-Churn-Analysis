'''
LOGISTIC REGRESSION 
'''


import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib
matplotlib.style.use('ggplot')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_sc

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler

'''
Model 1 for Logistic Classifier :  
a) FEATURES WITHOUT SPLITTING DUMMY(STATE) into columns equal to the unique values of the state
b) Using StandardScaled FEATURES
'''
X_features_normScaled #Norm Scaled 

np.random.seed(123)
X_norm_train, X_norm_test, y_norm_train, y_norm_test = train_test_split(X_features_normScaled, y_target, 
                                                    test_size = 0.3, random_state=123, stratify = y_target)
#Training Logistic Classifier | StandardScaled Data
logit_norm_data = LogisticRegression(max_iter = 5000) #crucial in order to converge
fit_logit_norm_data = logit_norm_data.fit(X_norm_train, y_norm_train)

y_norm_data_predictions = fit_logit_norm_data.predict(X_norm_test)
#y_norm_data_predictions = pd.DataFrame(y_norm_data_predictions, columns = ['predicted_classes_norm_data'])
print(f"Logistic Classifier using Standard Scaled data converge after -{logit_norm_data.n_iter_}- iterations")


'''
Model 2 for Logistic Classifier :  
a) FEATURES BY SPLITTING DUMMY(STATE) into columns equal to the unique values of the state, i.e 73 predictors
b) Using StandardScaled Features
'''
X_features_with_dummies_normScaled

np.random.seed(123)                                               
X_with_dummies_normScaled_train, X_with_dummies_normScaled_test, y_with_dummies_normScaled_train, y_with_dummies_normScaled_test = train_test_split(X_features_with_dummies_normScaled, y_target, 
                                                    test_size = 0.3, random_state=123, stratify = y_target) #Shuffle by default = True
#Training Logistic Classifier | Standard Scaled Data | 73 predictors
logit_with_dummies_normScaled_data = LogisticRegression(max_iter = 5000) #crucial in order to converge
fit_logit_with_dummies_normScaled_data = logit_with_dummies_normScaled_data.fit(X_with_dummies_normScaled_train, y_with_dummies_normScaled_train)

y_with_dummies_normScaled_data_predictions = fit_logit_with_dummies_normScaled_data.predict(X_with_dummies_normScaled_test)
#y_with_dummies_normScaled_data_predictions = pd.DataFrame(y_with_dummies_normScaled_data_predictions, columns = ['predicted_classes_nor_data'])
print(f"Logistic Classifier with 73 predictors and with norm scaled data, converge after -{logit_with_dummies_normScaled_data.n_iter_}- iterations")


# Efficiency of Logistic Classifier 

'''
Model 1 for Logistic Classifier :  
a) FEATURES WITHOUT SPLITTING DUMMY(STATE) into columns equal to the unique values of the state
b) Using StandardScaled FEATURES
'''
# Accuracy in train and in test set
logit_model1_accuracy_train = fit_logit_norm_data.score(X_norm_train, y_norm_train)
logit_model1_accuracy_test = fit_logit_norm_data.score(X_norm_test, y_norm_test)
print(f"For Logistic Classifier Model 1 using StandardScaler and 23 predictors.\n Accuracy in train set is {logit_model1_accuracy_train}.\n Accuracy in test set is {logit_model1_accuracy_test}.")
print("-"*100)
print(f"Number of mislabeled points on the test data set:\n{(y_norm_test != y_norm_data_predictions).sum()},out of {(len(X_norm_test))} total observations.")
print(f"{round(((y_norm_test != y_norm_data_predictions).sum()/(X_norm_test.shape[0])) * 100, 3)}%")


'''
Model 2 for Logistic Classifier :  
a) FEATURES by SPLITTING DUMMY(STATE) into columns equal to the unique values of the state
b) Using StandardScaled FEATURES
'''
# Accuracy in train and in test set
logit_model2_accuracy_train = fit_logit_with_dummies_normScaled_data.score(X_with_dummies_normScaled_train, y_with_dummies_normScaled_train)
logit_model2_accuracy_test = fit_logit_with_dummies_normScaled_data.score(X_with_dummies_normScaled_test, y_with_dummies_normScaled_test)
print(f"For Logistic Classifier Model 2 using StandardScaler and 73 predictors.\n Accuracy in train set is {logit_model2_accuracy_train}.\n Accuracy in test set is {logit_model2_accuracy_test}.")
print("-"*100)
print(f"Number of mislabeled points on the test data set:\n{(y_with_dummies_normScaled_test != y_with_dummies_normScaled_data_predictions).sum()},out of {(len(X_with_dummies_normScaled_test))} total observations.")
print(f"{round(((y_with_dummies_normScaled_test != y_with_dummies_normScaled_data_predictions).sum() / (X_with_dummies_normScaled_test.shape[0])) * 100, 3)} %")

'''
Cross Validation with 5-Folds for both Logistic Classifiers with 23 predictors and 73 predictors
'''
# 5 - Fold cross validation for Logistic Classifier 1
cross_validation_model1 = cross_val_score(fit_logit_norm_data, X_norm_train, y_norm_train, cv = 5, scoring = "accuracy")
cross_validation_model2 = cross_val_score(fit_logit_with_dummies_normScaled_data, X_with_dummies_normScaled_train, y_with_dummies_normScaled_train, cv = 5, scoring = "accuracy")
print(f"For Logistic Classifier Model 1 using StandardScaler and 23 predictors 5 - fold validation returns accuracy : {np.mean(cross_validation_model1)}.")
print(f"For Logistic Classifier Model 2 using StandardScaler and 73 predictors 5 - fold validation returns accuracy : {np.mean(cross_validation_model2)}.")



'''
Visualize for both models with 23 predictors and 73 predictors Accuracy in Train Set and 5 - Fold CV
'''

fig, axes = plt.subplots(1, 2, figsize = (12, 6))
fig.suptitle('Accuracy in Train set | 5-Fold Cross Validation Accuracy', fontweight = 'bold', fontsize = 14)

#5-Fold CV and Train Accuracy Logistic Classifier Model 1
x = ['Accuracy : 5-Fold CV', "Accuracy : Train Set"]
y_logit_model1 = [np.mean(cross_validation_model1), logit_model1_accuracy_train]

sns.barplot(x = x, y = y_logit_model1, ax = axes[0], palette = ['blue', 'lightblue'])
axes[0].set_title("Logit Classifier Model 1 : 23 predictors", fontweight = "bold");

#5-Fold CV and Train Accuracy Logistic Classifier Model 2
y_logit_model2 = [np.mean(cross_validation_model2), logit_model2_accuracy_train]

sns.barplot(x = x, y = y_logit_model2, ax = axes[1], palette = ['blue', 'lightblue'])
axes[1].set_title("Logit Classifier Model 2 : 73 predictors", fontweight = "bold");



'''
Logistic Regression Model : Confusion Matrix in Test Set
'''

# Confusion Matrix for Logistic Regression Model 1
confusion_matric_logit_model1 = confusion_matrix(y_norm_test, y_norm_data_predictions)
# Confusion Matrix for Logistic Regression Model 2
confusion_matric_logit_model2 = confusion_matrix(y_with_dummies_normScaled_test, y_with_dummies_normScaled_data_predictions)

#Visualize Confusion Matrix for both models
fig, axes = plt.subplots(1, 2, figsize = (12, 6))

# Confusion Matrix for Logistic Regression Model 1
sns.heatmap(confusion_matric_logit_model1, annot = True, fmt = 'd', cbar = True, cmap = 'Blues', ax = axes[0])
axes[0].set_xlabel('true label')
axes[0].set_ylabel('predicted label')
axes[0].set_title("Confusion Matrix : Logistic Regression Model 1", fontweight = "bold")

# Confusion Matrix for Logistic Regression Model 2
sns.heatmap(confusion_matric_logit_model2 , annot = True, fmt = 'd', cbar = True, cmap = 'Reds', ax = axes[1])
axes[1].set_xlabel('true label')
axes[1].set_ylabel('predicted label')
axes[1].set_title("Confusion Matrix : Logistic Regression Model 2", fontweight = "bold");



'''
Precision score and Recall score for Logistic Classifiers
'''
# Model 1
precision_score_logit_model1 = precision_score(y_norm_test, y_norm_data_predictions)
recall_score_logit_model1 = recall_score(y_norm_test, y_norm_data_predictions)

#Model 2
precision_score_logit_model2 = precision_score(y_with_dummies_normScaled_test, y_with_dummies_normScaled_data_predictions)
recall_score_logit_model2 = recall_score(y_with_dummies_normScaled_test, y_with_dummies_normScaled_data_predictions)

print(f"For Logistic Classifier 1 with 23 predictors and Standard Scaled Data PRECISON SCORE is [{precision_score_logit_model1}]")
print(f"For Logistic Classifier 1 with 23 predictors and Standard Scaled Data RECALL SCORE   is [{recall_score_logit_model1}]")
print("-" * 50)
print(f"For Logistic Classifier 2 with 73 predictors and Standard Scaled Data PRECISON SCORE is [{precision_score_logit_model2}]")
print(f"For Logistic Classifier 2 with 73 predictors and Standard Scaled Data RECALL SCORE   is [{recall_score_logit_model2}]")
print("-" * 50)


'''
plot_precision_recall_curve
'''
# Plot the Precision-Recall Curve fo Logit Classifier 1
plot_precision_recall_curve_model1 = plot_precision_recall_curve(fit_logit_norm_data,
                                                                 X_norm_test, y_norm_test)
# Plot the Precision-Recall Curve fo Logit Classifier 1
plot_precision_recall_curve_model2 = plot_precision_recall_curve(fit_logit_with_dummies_normScaled_data,
                                                                 X_with_dummies_normScaled_test, y_with_dummies_normScaled_test)

plot_precision_recall_curve_model1
plot_precision_recall_curve_model2;