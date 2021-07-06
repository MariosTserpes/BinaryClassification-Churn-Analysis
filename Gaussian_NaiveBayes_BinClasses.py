import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib
matplotlib.style.use('ggplot')

from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
from sklearn.metrics import f1_score, plot_roc_curve, roc_curve,  plot_precision_recall_curve


'''
Naive Bayes Model 1 | 23 Predictors
'''

# instantiate the model
np.random.seed(123)
Naive_Bayes_Classifier_model1 = GaussianNB()
fit_Naive_Bayes_Classifier_model1 = Naive_Bayes_Classifier_model1.fit(X_norm_train, y_norm_train)

NaiveBayes_predictions_model1 = fit_Naive_Bayes_Classifier_model1.predict(X_norm_test)
NaiveBayes_predictions_prob_model1 = fit_Naive_Bayes_Classifier_model1.predict_proba(X_norm_test)
print(F"Model 1 and 23 Predictors => Naive Bayes Classfier assigned in each class:\n{pd.DataFrame(NaiveBayes_predictions_model1).value_counts()}")

#Accuracy Score 
print(f"On the  test set accuracy is : {accuracy_score(y_norm_test, NaiveBayes_predictions_model1)}")
print(f"On the train set accuracy is : {accuracy_score(y_norm_train, fit_Naive_Bayes_Classifier_model1.predict(X_norm_train))}")

#5 - Fold CV 
naiveBayes_cross_validation_model1 = cross_val_score(fit_Naive_Bayes_Classifier_model1, X_norm_train,
                                                    y_norm_train, cv = 5, scoring = 'accuracy')
print(f"Model 1 | 23 Predictors => Mean 5-Fold CV : [{np.mean(naiveBayes_cross_validation_model1)}].")
print(f"Each iteration of 5-Fold CV : {naiveBayes_cross_validation_model1}")


#Confusion Matrix
confusion_matrix_model1 = confusion_matrix(y_norm_test, NaiveBayes_predictions_model1)
plt.figure( figsize = (12, 6))
sns.heatmap(confusion_matrix_model1, annot = True, fmt = 'd', cbar = True, cmap = 'Reds')
plt.xlabel('Actual Classes')
plt.ylabel('Predicted Classes')
plt.title('Confusion Matrix Model 1 | 23 Predictors', fontweight = "bold");

# Precision
precision_score_NaiveBayes_model1 = precision_score(y_norm_test, NaiveBayes_predictions_model1)
print(f"Model 1 | 23 Predictors : Precision Score => {precision_score_NaiveBayes_model1}")
# Recall
recall_score_NaiveBayes_model1 = recall_score(y_norm_test, NaiveBayes_predictions_model1)
print(f"Model 1 | 23 Predictors :   Recall Score  => {recall_score_NaiveBayes_model1}")
# f1 score
f1_score_NaiveBayes_model1 = f1_score(y_norm_test, NaiveBayes_predictions_model1)
print(f"Model 1 | 23 Predictors :       f1 Score  => {f1_score_NaiveBayes_model1}")


#Class Probabilities | Model 1 | 23 Predictors
NaiveBayes_predictions_prob_model1 #Probabilities
NaiveBayes_predictions_prob_model1_df = pd.DataFrame(NaiveBayes_predictions_prob_model1, columns = ['Churner', 'Loyal'])

#Plotting Prob of Loyal Customers
fig, axes = plt.subplots(1, 2, figsize = (15, 6))
fig.suptitle('2-Class Predicted Probabilities for Model 1 | 23 Predictors', fontweight = 'bold', fontsize = 14)

#Predicted Probabilities class == 1
sns.histplot(NaiveBayes_predictions_prob_model1_df['Loyal'], ax = axes[0])
axes[0].set_title('Predicted Probabilities of Customers being Loyals', fontweight = "bold")

#Predicted Probabilities class == 1
sns.histplot(NaiveBayes_predictions_prob_model1_df['Churner'], ax = axes[1])
axes[1].set_title('Predicted Probabilities of Customers being Churners', fontweight = "bold");


'''
Naive Bayes Model 2 | 73 Predictors
'''

# instantiate the model
np.random.seed(123)
Naive_Bayes_Classifier_model2 = GaussianNB()
fit_Naive_Bayes_Classifier_model2 = Naive_Bayes_Classifier_model2.fit(X_with_dummies_normScaled_train, y_with_dummies_normScaled_train)

NaiveBayes_predictions_model2 = fit_Naive_Bayes_Classifier_model2.predict(X_with_dummies_normScaled_test)
NaiveBayes_predictions_prob_model2 = fit_Naive_Bayes_Classifier_model2.predict_proba(X_with_dummies_normScaled_test)
print(F"Model 2 and 73 Predictors => Naive Bayes Classfier assigned in each class:\n{pd.DataFrame(NaiveBayes_predictions_model2).value_counts()}")

#Accuracy Score 
print(f"On the  test set accuracy is : {accuracy_score(y_with_dummies_normScaled_test, NaiveBayes_predictions_model2)}")
print(f"On the train set accuracy is : {accuracy_score(y_with_dummies_normScaled_train, fit_Naive_Bayes_Classifier_model2.predict(X_with_dummies_normScaled_train))}")

#Confusion Matrix
confusion_matrix_model2 = confusion_matrix(y_with_dummies_normScaled_test, NaiveBayes_predictions_model2)
plt.figure( figsize = (12, 6))
sns.heatmap(confusion_matrix_model2, annot = True, fmt = 'd', cbar = True, cmap = 'Reds')
plt.xlabel('Actual Classes')
plt.ylabel('Predicted Classes')
plt.title('Confusion Matrix Model 2 | 73 Predictors', fontweight = "bold");

# Precision
precision_score_NaiveBayes_model2 = precision_score(y_with_dummies_normScaled_test, NaiveBayes_predictions_model2)
print(f"Model 2 | 73 Predictors : Precision Score => {precision_score_NaiveBayes_model2}")
# Recall
recall_score_NaiveBayes_model2 = recall_score(y_with_dummies_normScaled_test, NaiveBayes_predictions_model2)
print(f"Model 2 | 23 Predictors :   Recall Score  => {recall_score_NaiveBayes_model2}")
# f1 score
f1_score_NaiveBayes_model2 = f1_score(y_with_dummies_normScaled_test, NaiveBayes_predictions_model2)
print(f"Model 2 | 73 Predictors :       f1 Score  => {f1_score_NaiveBayes_model2}")
print(f"Good Performance for predicting the positive class 'Loyal Customer' out of actual positive outcomes.")
