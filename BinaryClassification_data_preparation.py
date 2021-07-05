import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib
matplotlib.style.use('ggplot')

from sklearn.preprocessing import StandardScaler, MinMaxScaler



'''
Outliers detection +-3 Standard Deviation Technique before scaling data.
The below procedure will be done only for continuous_quantitative_features.
'''

def outlier_removal(df, variable):
    upper_limit = df[variable].mean() + ( df[variable].std() * 3)
    lower_limit = df[variable].mean() - ( df[variable].std() * 3)
    return upper_limit, lower_limit

for column in data[continuous_quantitative_features].columns: # continuous_quantitative_features
    upper_limit, lower_limit = outlier_removal(data, column)
    print(f"Column : {data[column].name}")
    print(f"Upper Limit : {upper_limit}")
    print(f"Lower Limit : {lower_limit}")
    print(f"Number of Observation higher than +3 std from the mean : {len(data[data[column] > upper_limit])}")
    print(f"Number of Observation lower than -3 std from the mean  : {len(data[data[column] < lower_limit])}")
    print("-" * 30)

'''
In this step i will remove outliers of continuous_quantitative_features 
from our dataset
'''

for column in data[continuous_quantitative_features].columns:
    upper_limit, lower_limit = outlier_removal(data, column)
    outliers = (data[
          (data[column] > upper_limit)
        | (data[column] < lower_limit)
    ])
    data = data.drop(outliers[column].index)

print(f"After removing outliers for continuous_quantitative_features the shape of dataset is : {data.shape}")


'''
Features Preparation | Splitting | Scaling | 2 approaches
'''

X_features = data.iloc[ :, : -1] # Dropping Class column
y_target = data['class'] # [Churner = 0 & Loyal Customer = 1]
print(f"Columns of predictors = {X_features.shape[1]} and rows of predictors' frame = {X_features.shape[0]}")


'''
In this step i will create a second frame with predictors but  i will treat
column STATE as ORDINAL CATEGORICAL. Obviously, the number of predictors will be increased.
'''

X_features_with_dummies = X_features.copy() #a copy of X_features that contains +50 columns since getting dummies
X_features_with_dummies = pd.get_dummies(X_features_with_dummies, prefix = ['state'], columns = ['state'])
print(f"Columns of predictors = {X_features_with_dummies.shape[1]} and rows of predictors' frame = {X_features_with_dummies.shape[0]}")


'''
As I have mentioned before, we could exploit columns related to Minutes, Charges and Calls.
We can add more information in our dataset, utilizing daily characteristics of customers.
Hint : The below procedure will be done both X_features and X_features_with_dummies
'''
#For X_features
# a.) Daily Minutes b.) Daily Calls c.) Daily Charges
X_features["daily_minutes"] = X_features["total_day_minutes"] + X_features["total_eve_minutes"] + X_features["total_night_minutes"]
X_features["daily_calls"] = X_features["total_day_calls"] + X_features["total_eve_calls"] + X_features["total_night_calls"]
X_features["daily_charges"] = X_features["total_day_charge"] + X_features["total_eve_charge"] + X_features["total_night_charge"]


#For X_features_with_dummies
# a.) Daily Minutes b.) Daily Calls c.) Daily Charges
X_features_with_dummies["daily_minutes"] = X_features_with_dummies["total_day_minutes"] + X_features_with_dummies["total_eve_minutes"] + X_features_with_dummies["total_night_minutes"]
X_features_with_dummies["daily_calls"] = X_features_with_dummies["total_day_calls"] + X_features_with_dummies["total_eve_calls"] + X_features_with_dummies["total_night_calls"]
X_features_with_dummies["daily_charges"] = X_features_with_dummies["total_day_charge"] + X_features_with_dummies["total_eve_charge"] + X_features_with_dummies["total_night_charge"]

print(f"For X_features without dummies, columns of predictors = {X_features.shape[1]} and rows of predictors' frame = {X_features.shape[0]}")
print(f"For X_features_with_dummies, columns of predictors = {X_features_with_dummies.shape[1]} and rows of predictors' frame = {X_features_with_dummies.shape[0]}")


'''
Now, features will be scaled using StandardScaler and MinMaxScaler
for both X_features[23 columns] and X_features_with_dummies[73 columns]
'''

scaler_norm = StandardScaler()
scaler_minmax = MinMaxScaler()

#Features without getting dummies | StandardScaler
X_features_normScaled = scaler_norm.fit_transform(X_features.values)
X_features_normScaled = pd.DataFrame(X_features_normScaled, columns = X_features.columns.tolist())

#Features without getting dummies | MinMaxScaler
X_features_minMaxScaled = scaler_minmax.fit_transform(X_features.values)
X_features_minMaxScaled = pd.DataFrame(X_features_minMaxScaled, columns = X_features.columns.tolist())

#Features with dummies | StandardScaler
X_features_with_dummies_normScaled = scaler_norm.fit_transform(X_features_with_dummies.values)
X_features_with_dummies_normScaled = pd.DataFrame(X_features_with_dummies_normScaled, columns = X_features_with_dummies.columns.tolist())

#Features with dummies | MinMaxScaler
X_features_with_dummies_MinMaxScaled = scaler_minmax.fit_transform(X_features_with_dummies.values)
X_features_with_dummies_MinMaxScaled = pd.DataFrame(X_features_with_dummies_MinMaxScaled, columns = X_features_with_dummies.columns.tolist())
