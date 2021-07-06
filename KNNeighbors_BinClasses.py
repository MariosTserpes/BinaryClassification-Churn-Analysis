'''
K - Nearest Neighbors
'''

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib
matplotlib.style.use('ggplot')

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, plot_precision_recall_curve
from sklearn.metrics import f1_score, plot_roc_curve

'''
a) Error Rate for each K 
Model 1 : K-Nearest Neighbors with 23 predictors and Standard Scaled predictors.
'''
error_rate_train = []
error_rate = []
for k in range(1, 40):
    np.random.seed(123)
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_norm_train, y_norm_train)
    predictions = knn.predict(X_norm_test)
    error_rate.append(np.mean(y_norm_test != predictions))
    error_rate_train.append(np.mean(y_norm_train != knn.predict(X_norm_train)))
    
plt.figure( figsize = (12, 6))
plt.plot(range(1, 40), error_rate, color = "red", linestyle = "dashed", marker = "o", 
        markerfacecolor = "blue", markersize = 10, label = 'Mean Error Rate | Test Set')
plt.plot(range(1, 40), error_rate_train, color = "blue", linestyle = "dashed", marker = "o", 
        markerfacecolor = "red", markersize = 10, label = 'Mean Error Rate | Train Set')
plt.title('Mean Error Rate For Each K-Value for K-NN Model 1 | 23 Predictors', fontweight = "bold", fontsize = 15)
plt.xlabel("K Value")
plt.ylabel('Mean Error')
plt.legend()
plt.show();


'''
a) Error Rate for each K 
Model 2 : K-Nearest Neighbors with 73 predictors and Standard Scaled predictors.
'''

error_rate_model2_train = []
error_rate_model2 = []
for k in range(1, 40):
    np.random.seed(123)
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_with_dummies_normScaled_train, y_with_dummies_normScaled_train)
    predictions_model2 = knn.predict(X_with_dummies_normScaled_test)
    error_rate_model2.append(np.mean(y_with_dummies_normScaled_test != predictions_model2))
    error_rate_model2_train.append(np.mean(y_with_dummies_normScaled_train != knn.predict(X_with_dummies_normScaled_train)))
    
plt.figure( figsize = (12, 6))
plt.plot(range(1, 40), error_rate_model2, color = "red", linestyle = "dashed", marker = "o", 
        markerfacecolor = "blue", markersize = 7, label = 'Mean Error Rate | Test Set')
plt.plot(range(1, 40), error_rate_model2_train, color = "blue", linestyle = "dashed", marker = "o", 
        markerfacecolor = "red", markersize = 7, label = 'Mean Error Rate | Train Set')
plt.title('Mean Error Rate For Each K-Value for K-NN Model 2 | 73 Predictors', fontweight = "bold", fontsize = 15)
plt.xlabel("K Value")
plt.ylabel('Mean Error')
plt.legend()
plt.show();


'''
Accuracy Score for Train and Test Set | Model 1 +  23 Predictors
'''

accuracy_test = []
accuracy_train = []
for k in range(2, 40):
    np.random.seed(123)
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_norm_train, y_norm_train)
    preds = knn.predict(X_norm_test)
    accuracy_test.append(accuracy_score(y_norm_test, preds))
    accuracy_train.append(accuracy_score(y_norm_train, knn.predict(X_norm_train)))  

plt.figure( figsize = (12, 6))
plt.plot(range(2, 40), accuracy_test, label = 'Accuracy in test set')
plt.plot(range(2, 40), accuracy_train, label = 'Accuracy in train set')
plt.title('Accuracy Score in Test and Train Subset | Model 1 = 23 Predictors', fontweight = "bold", fontsize = 15)
plt.xlabel("K Values")
plt.ylabel("Accuracy Score")
plt.legend();

'''
Accuracy score for each K | Model 2 + 73 Predictors
'''

acc_score_model2_train = []
acc_score_model2 = []
for k in range(2, 40):
    np.random.seed(123)
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_with_dummies_normScaled_train, y_with_dummies_normScaled_train)
    predictions_model2 = knn.predict(X_with_dummies_normScaled_test)
    acc_score_model2.append(accuracy_score(y_with_dummies_normScaled_test, predictions_model2))
    acc_score_model2_train.append(accuracy_score(y_with_dummies_normScaled_train, knn.predict(X_with_dummies_normScaled_train)))
    
plt.figure( figsize = (12, 6))
plt.plot(range(2, 40), acc_score_model2, color = "red", label = 'Accuracy Score | Test Set')
plt.plot(range(2, 40), acc_score_model2_train, color = "blue",  label = 'Accuracy Score | Train Set')
plt.title('Accuracy Score For Each K-Value for K-NN Model 2 | 73 Predictors', fontweight = "bold", fontsize = 15)
plt.xlabel("K Value")
plt.ylabel('Accuracy Score')
plt.legend()
plt.show();


'''
A Function In order To Print Metrics For k-nn Model 1 | 23 Predictors
'''


def metrics_for_my_model1(neighbors):
    X_norm_train, X_norm_test, y_norm_train, y_norm_test
    print("Model has 23 Predictors and data have been scaled with Standard Scaler.")
    model = KNeighborsClassifier(n_neighbors = neighbors)
    model_fit = model.fit(X_norm_train, y_norm_train)
    print(f"Model with Neighbors : {neighbors}.")
    predictions = model.predict(X_norm_test)
    accuracy = accuracy_score(y_norm_test, predictions)
    print(f"Accuracy of Model is : {round(accuracy, 4)}")
    cross_validation_score = cross_val_score(model_fit, X_norm_train, y_norm_train, cv = 5, scoring = 'accuracy')
    print(f"Accuracy after 5-Fold Cross Validation is : {np.mean(cross_validation_score)}")
    prec_score = precision_score(y_norm_test, predictions)
    print(f"Precision Score is : {prec_score}")
    rec_score = recall_score(y_norm_test, predictions)
    print(f"Recall Score is : {rec_score}")
    confusion_matrix_results = confusion_matrix(y_norm_test, predictions)
    print(f"Confusion Matrix for {model} is :\n{confusion_matrix_results}")
    model_precision_recall_plot = plot_precision_recall_curve(model_fit, X_norm_test, y_norm_test)
    model_precision_recall_plot;


'''
5 - NN Model 1 | 23 Predictors
'''

np.random.seed(123)
metrics_for_my_model1(5)

'''
7-NN | Model 1 | 23 Predictors
'''
np.random.seed(123)
metrics_for_my_model1(7)

'''
9 - NN Model 1 | 23 Predictors
'''
np.random.seed(123)
metrics_for_my_model1(9)

'''
11 - NN Model 1 | 23 Predictors
'''
np.random.seed(123)
metrics_for_my_model1(11)


'''
A Function In order To Print Metrics For k-nn Model 2 | 73 Predictors
'''


def metrics_for_my_model2(neighbors):
    X_with_dummies_normScaled_train, X_with_dummies_normScaled_test,
    y_with_dummies_normScaled_train, y_with_dummies_normScaled_test
    print("Model has 73 Predictors and data have been scaled with Standard Scaler.")
    model = KNeighborsClassifier(n_neighbors = neighbors)
    model_fit = model.fit(X_with_dummies_normScaled_train, y_with_dummies_normScaled_train)
    print(f"Model with Neighbors : {neighbors}.")
    predictions = model.predict(X_with_dummies_normScaled_test)
    accuracy = accuracy_score(y_with_dummies_normScaled_test, predictions)
    print(f"Accuracy of Model is : {round(accuracy, 4)}")
    cross_validation_score = cross_val_score(model_fit, X_with_dummies_normScaled_train,
                                             y_with_dummies_normScaled_train, cv = 5, scoring = 'accuracy')
    print(f"Accuracy after 5-Fold Cross Validation is : {np.mean(cross_validation_score)}")
    prec_score = precision_score(y_norm_test, predictions)
    print(f"Precision Score is : {prec_score}")
    rec_score = recall_score(y_with_dummies_normScaled_test, predictions)
    print(f"Recall Score is : {rec_score}")
    confusion_matrix_results = confusion_matrix(y_with_dummies_normScaled_test, predictions)
    print(f"Confusion Matrix for {model} is :\n{confusion_matrix_results}")
    model_precision_recall_plot = plot_precision_recall_curve(model_fit, X_with_dummies_normScaled_test,
                                                              y_with_dummies_normScaled_test)
    model_precision_recall_plot;


'''
3 - NN | Model 2 | 73 Predictors
'''
metrics_for_my_model2(3)

'''
5 - NN | Model 2 | 73 Predictors
'''
metrics_for_my_model2(5)

'''
7 - NN | Model 2 | 73 Predictors
'''
metrics_for_my_model2(7)

'''
9 - NN | Model 2 | 73 Predictors
'''
metrics_for_my_model2(9)

