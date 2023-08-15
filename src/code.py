'''The code for Assignment 1 of Big Data Mining in Healthcare
Akshita Gupta - 2020491
Ayush Raje Chak - 2020502
Utkarsh Arora - 2020143

To get our highest ROC, we have applied various machine learning models and
techniques. This is the code for the best techniques we have been able to achieve.
'''

#Importing important libraries
import sys
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, RFE, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

warnings.filterwarnings("ignore")

#Importing the training and testing datasets
train_data = pd.read_csv(sys.argv[1])
x_test_data = pd.read_csv(sys.argv[2])
x_test = x_test_data.drop('ID',axis=1) #Cleaning the test dataset by dropping an extra column


#Extracting the independent (x) and dependent (y) variables
x_train = train_data.iloc[:,1:-1]
y_train = train_data.iloc[:,0]


#Given an external estimator that assigns weights to features (e.g., the
#coefficients of a linear model), the goal of recursive feature elimination
#(RFE) is to select features by recursively considering smaller and smaller sets
#of features. First, the estimator is trained on the initial set of features and
#the importance of each feature is obtained either through any specific
#attribute or callable. Then, the least important features are pruned from
#current set of features. That procedure is recursively repeated on the pruned
#set until the desired number of features to select is eventually reached.
lr = LogisticRegression()
rfe = RFE(lr) #It assigns weights to features using logistic regression
sf = rfe.fit(x_train, y_train)  # It learns relationship and transfrom the data
sc = x_train.columns[rfe.get_support(indices=True)] #This saves the columns names in sc variable
X = x_train[sc] #This will show the first few rows of selected features

mod_x_train, mod_x_test, mod_y_train, mod_y_test = train_test_split(X, y_train, test_size=0.2, random_state=4)


#Using a random forest classifier to predict target variable based on training
#data. The classifier uses multiple decision trees, each trained on a randomly
#sampled subset of features and samples, and aggregates their results to make
#final predictions. This method reduces overfitting and improves accuracy
#compared to single decision trees.
rf = RandomForestClassifier(random_state=56)
rf.fit(X,y_train)
RF_pred = rf.predict(x_test[sc])


#Using AdaBoost (Adaptive Boosting) to predict target variable based on training
#data. The algorithm combines weak classifiers (e.g., decision stumps) into a
#strong classifier by assigning weights to each sample based on previous
#classification results. Misclassified samples receive higher weights, which are
#then used to update the next weak classifier. The process continues until a
#predefined number of iterations is reached or until all samples are correctly
#classified. AdaBoost can be effective in handling imbalanced datasets and
#improving overall accuracy.
clf = AdaBoostClassifier(n_estimators=470)
clf.fit(X,y_train)
ADA_pred = clf.predict(x_test[sc])

#Creating the output csv file
df = pd.DataFrame(ADA_pred,columns=["Labels"])
output_df = pd.DataFrame()
output_df.insert(loc = 0,column = "ID",value = x_test_data['ID'])
output_df.insert(loc = 1,column = "Labels",value = df['Labels'])
output_df.to_csv('out.csv', index=False)

