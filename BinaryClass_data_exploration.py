import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib
matplotlib.style.use('ggplot')

# 2. Reading Dataset - Description of Variables
data = pd.read_csv("churn.csv")
print(f"Dataset's rows : {data.shape[0]}.\nDataset's columns : {data.shape[1]}.")

for column in data.columns:
    print(f"Column's name : {data[column].name}")
    print(f"Variable's type : {data[column].dtype}")
    print(f"Number of unique values : {data[column].nunique()}")
    print('-'*75)

#Loop(k)ing For Missing Values
for column in data.columns:
    print(f"Column's name : {data[column].name}")
    print(f"Number Of NaN values : {data[column].isnull().sum()}") #The Number of NaN values if there are NaN values
    print(f"Unique values : {data[column].unique()}") #In orded to be sured that there are not symbols [?, #, !]
    print("-"*20)

# 3. Exploratory Analysis

'''
1. Number of Targets in dataset
'''
plt.figure( figsize = (10, 10))

#Preparation For Visualization
labels_target = ['Churner', 'Loyal'] # Churner : 0 | No_chUrner : 1
sizes_labels  = data['class'].value_counts(sort = True) 
colors  = ["lightblue", "blue"]
explode = (0.15, 0)

plt.title('Number of Targets', fontsize = 18, fontweight = 'bold')
plt.pie(sizes_labels, explode = explode, labels = labels_target , colors = colors, autopct = '%1.1f%%', 
        shadow=True, startangle = 200, textprops = {"fontsize" : 20, "fontweight" : "bold"})
plt.show()

'''
2. Additional Visualizations
'''
fig, axes = plt.subplots(1, 4, figsize = (25, 10))
fig.suptitle('Insights per Targets', fontweight = 'bold', fontsize = 16)

#a. Calls of customers for services
sns.countplot(data['number_customer_service_calls'], hue = data['class'], palette = ['blue', 'lightblue'], ax = axes[0])
axes[0].set_title("number_customer_service_calls per targets", fontweight = "bold")
axes[0].legend(labels = ['Churners', 'Loyals'])

#b. International plan per target(churner or no churne)
sns.countplot(data['international_plan'], hue = data['class'], ax = axes[1], palette = ['blue', 'lightblue'])
axes[1].set_title("International plan per targets", fontweight = "bold")
axes[1].legend(labels = ['Churners', 'Loyals'])

#c. Voice mail plan per target
sns.countplot(data['voice_mail_plan'], hue = data['class'], ax = axes[2], palette = ['blue', 'lightblue'])
axes[2].set_title("Voice mail plan per targets", fontweight = "bold")
axes[2].legend(labels = ['Churners', 'Loyals'])

#d. Where do churners and loyals come from?
sns.countplot(data['area_code'], hue = data['class'], ax = axes[3], palette = ['blue', 'lightblue'])
axes[3].set_title("Are per targets", fontweight = "bold")
axes[3].legend(labels = ['Churners', 'Loyals']);


'''
3. Distributions for continuous numeric variables
'''

# Define Quantitative continuous features
continuous_quantitative_features = ['number_vmail_messages', 'total_day_minutes', 'total_day_calls', 'total_day_charge',
                                   'total_eve_minutes', 'total_eve_calls', 'total_eve_charge', 'total_night_minutes',
                                   'total_night_calls', 'total_night_charge','total_intl_minutes', 'total_intl_calls', 
                                   'total_intl_charge']

# A loop in order to visualize distributions per target class[0 = Churner, 1 = Loyal Customer]
chart_position = 1
plt.figure(figsize=(35,20))
for column in continuous_quantitative_features:
    plt.subplot(3, 5, chart_position)
    sns.kdeplot(data[column], hue = data['class'], shade = True, palette = ['orange', 'purple'])
    plt.xlabel(column, fontsize = 15, fontweight = 'bold')
    plt.xticks(fontsize = 10)
    chart_position += 1;


'''
Box plots continuous_quantitative_features
'''

chart_position = 1
plt.figure( figsize = (35, 20))
for column in continuous_quantitative_features:
    plt.subplot(3, 5, chart_position)
    sns.boxplot(data[column], color = 'lightblue')
    plt.xlabel(column, fontsize = 15, fontweight = 'bold')
    plt.xticks(fontsize = 10)
    chart_position += 1;


'''
Average number of churn for different combinations of
international plan and voice mail plan
'''
ax = plt.figure( figsize = (10, 5))
data.groupby(['international_plan',"voice_mail_plan"])['class'].mean().plot( figsize = (10, 5), 
                                    kind = "bar", color = "lightblue", edgecolor = 'white')
plt.title("Average Plot for Churn" , fontsize = 18, fontweight = 'bold')
plt.xlabel("international_plan, voice_mail_plan", size = 13, fontweight = 'bold')
plt.ylabel("Average of Churn", size = 13, fontweight = 'bold')
plt.grid( True );


# 3. Decriptive Analysis

'''
Descriptive Statistics
'''

statistics = data.describe().T
statistics['+3 standard Deviations'] = statistics['mean'] + (statistics['std'] * 3)
statistics['-3 standard Deviations'] = statistics['mean'] - (statistics['std'] * 3)
statistics

'''
Average Number of  class : 0 [i.e Churner] for particulart Features
'''
#Dropping Features state & account_length & area_code & phone_number & : -1 = Class column
# Mean Number Of Features For Churners
print( "Class : Churner")
mean_data_churners = round(data.iloc[:, 4 :-1][data["class"] == 0].mean(), 2).reset_index()
mean_data_churners.columns = ['Feature', 'Mean']
mean_data_churners = mean_data_churners.set_index('Feature')
mean_data_churners = mean_data_churners.sort_values(by = 'Mean', ascending = False)

'''
Average Number of  class : 1 [i.e Loyal Customers] for each Feature
'''


# Mean Number Of Features For Loyal Customers
print( "Class : Loyal Customers")
mean_data_loyals = round(data.iloc[:, 4 :-1][data["class"] == 1].mean(), 2).reset_index() # # Dropping columns : State and Class
mean_data_loyals.columns = ['Feature', 'Mean']
mean_data_loyals = mean_data_loyals.set_index('Feature')
mean_data_loyals = mean_data_loyals.sort_values(by = 'Mean', ascending = False)


'''
Visualization of Average Number for each Feature per Class[Churners = 0, Loyals = 1]
CHURNERS use most on the nights and evenings | LOYALS most on evenings and days.
'''

fig, axes = plt.subplots(1, 2, figsize = (25, 10))
fig.suptitle('Average Number for Particular Features per Class[Churners = 0, Loyals = 1]', fontweight = 'bold', fontsize = 20)

# Churners
sns.barplot(x = mean_data_churners.index, y = mean_data_churners['Mean'], ax = axes[0])
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation = 90, fontweight = "bold", fontsize = 17)
axes[0].set_title("Mean Value for particular Features for Churners[Class = 0]", fontweight = 'bold', fontsize = 17)

# Churners
sns.barplot(x = mean_data_loyals.index, y = mean_data_loyals['Mean'], ax = axes[1])
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation = 90, fontweight = "bold", fontsize = 17)
axes[1].set_title("Mean Value for particular Features for Loyals[Class = 1]", fontweight = 'bold', fontsize = 17);


'''
Correlation Analysis
'''

ax = plt.figure( figsize = (20, 10))
sns.heatmap(data.corr(), annot = True, cmap = "Blues", fmt = '.0%')
plt.title("Correlation Analysis", fontsize = 30, fontweight = "bold")
plt.show();