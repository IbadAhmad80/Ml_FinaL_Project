"""
Created on Wed Aug 26 12:56:44 2020

@author: Ibad
"""

import pandas as pd
#Reading the Titanic Disaster Dataset CSV File
dataFrame=pd.read_csv('Titanic Disaster Dataset.csv')

#One Hot Encoding on Gender Attribute
dataFrame=pd.get_dummies(dataFrame, columns=['Gender'])
dataFrame=dataFrame.drop('Gender_female',axis=1)
dataFrame=dataFrame.rename(columns={'Gender_male':'male'})
#One Hot Encoding on Embarked Attribute
dataFrame=pd.get_dummies(dataFrame, columns=['Embarked'])
dataFrame=dataFrame.drop('Embarked_C',axis=1)

#Dropping missing values
dataFrame=dataFrame.dropna(axis=0,subset=None,inplace=False)

# Applying Principal Component Analysis for feature selection
from sklearn.decomposition import PCA
pca = PCA( n_components=4)
principalComponents = pca.fit_transform(dataFrame)
principalDf = pd.DataFrame(data = principalComponents  , columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4'])
finalDf = pd.concat([principalDf, dataFrame[['Survived']]], axis = 1)

# Doing 80-20% train_test split of Data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(finalDf.drop('Survived',axis=1), finalDf['Survived'], test_size=0.2)

# Balancing the Dataset using SEMOTE technique
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_train, Y_train= smote.fit_sample(X_train.astype('float'),Y_train)


#Normalizing data for KNN and ANN
from sklearn.preprocessing import StandardScaler
norm_whole_data= StandardScaler().fit_transform(finalDf.drop(['Survived'],axis=1))
norm_X_train= StandardScaler().fit_transform(X_train.iloc[:,:])
norm_X_test=  StandardScaler().fit_transform(X_test.iloc[:,:])

#Importing required modules for K fold cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kf = KFold(n_splits=10)

#Importing Evaluation Measures
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import metrics

#  Training Logistic Regression
from sklearn.linear_model import LogisticRegression
LG=LogisticRegression(solver='liblinear')
LG.fit(X_train,Y_train)
predictedLR=LG.predict(X_test)
print('Logistic Regression :',metrics.accuracy_score(Y_test,predictedLR))
print(confusion_matrix(Y_test,predictedLR))
print(classification_report(Y_test,predictedLR))

# Trainig Support Vector Machine
from sklearn.svm import SVC
SVC=SVC(C=100)
SVC.fit(X_train,Y_train)
predictedSVC=SVC.predict(X_test)
print('Support Vector Machine:',metrics.accuracy_score(Y_test,predictedSVC))
print(confusion_matrix(Y_test,predictedSVC))
print(classification_report(Y_test,predictedSVC))

# Training Random Forest
from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(n_estimators=300)
RF.fit(X_train,Y_train)
predictedRF=RF.predict(X_test)
print('Random Forest :',metrics.accuracy_score(Y_test,predictedRF))
print(confusion_matrix(Y_test,predictedRF))
print(classification_report(Y_test,predictedRF))


# Training Dicision Tree
from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier(criterion='entropy')
DT.fit(X_train,Y_train)
predictedDT= DT.predict(X_test)
print('Decision Tree :',metrics.accuracy_score(Y_test,predictedDT))
print(confusion_matrix(Y_test,predictedDT))
print(classification_report(Y_test,predictedDT))

# Training KNN
from sklearn.neighbors import KNeighborsClassifier
KNN= KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
KNN.fit(norm_X_train,Y_train)
predictedKNN = KNN.predict(norm_X_test)
print('KNN :',metrics.accuracy_score(Y_test,predictedKNN))
print(confusion_matrix(Y_test,predictedKNN))
print(classification_report(Y_test,predictedKNN))

# Training Guassain Naive Bayes
from sklearn.naive_bayes import GaussianNB
GNB=GaussianNB()
GNB.fit(X_train,Y_train)
predictedGuassianNB=GNB.predict(X_test)
print('Guassain NB :',metrics.accuracy_score(Y_test,predictedGuassianNB))
print(confusion_matrix(Y_test,predictedGuassianNB))
print(classification_report(Y_test,predictedGuassianNB))

# Training Bernoulli Naive Bayes
from sklearn.naive_bayes import BernoulliNB
BNB=BernoulliNB()
BNB.fit(X_train,Y_train)
predictedBernoulliNB=BNB.predict(X_test)
print('Bernoulli NB :',metrics.accuracy_score(Y_test,predictedBernoulliNB))
print(confusion_matrix(Y_test,predictedBernoulliNB))
print(classification_report(Y_test,predictedBernoulliNB))

# Training Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
SGD=SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)
SGD.fit(X_train,Y_train)
predictedSGD=SGD.predict(X_test)
print('SGD :',metrics.accuracy_score(Y_test,predictedSGD))
print(confusion_matrix(Y_test,predictedSGD))
print(classification_report(Y_test,predictedSGD))

# Training Stochastic Gradient Descent
from sklearn.neural_network import MLPClassifier
MLP=clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,), random_state=1)
MLP.fit(X_train,Y_train)
predictedMLP=MLP.predict(X_test)
print('MLP :',metrics.accuracy_score(Y_test,predictedMLP))
print(confusion_matrix(Y_test,predictedMLP))
print(classification_report(Y_test,predictedMLP))

#K fold cross validation on Logistic Regression
from sklearn.linear_model import LogisticRegression
testing_scoreLR=cross_val_score(LogisticRegression(solver='liblinear'),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10)
testing_meanScore_LR=testing_scoreLR.mean()
testing_precision_LR=cross_val_score(LogisticRegression(solver='liblinear'),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10,scoring='precision')
testing_recall_LR=cross_val_score(LogisticRegression(solver='liblinear'),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10,scoring='recall')
testing_f1_LR=cross_val_score(LogisticRegression(solver='liblinear'),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10,scoring='f1')

#K fold cross validation on Support Vector Machine
from sklearn.svm import SVC
testing_scoreSVM=cross_val_score(SVC(C=100),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10)
testing_meanScore_SVM=testing_scoreSVM.mean()
testing_precision_SVM=cross_val_score(SVC(C=100),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10,scoring='precision')
testing_recall_SVM=cross_val_score(SVC(C=100),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10,scoring='recall')
testing_f1_SVM=cross_val_score(SVC(C=100),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10,scoring='f1')

#K fold cross validation on Random Forest
from sklearn.ensemble import RandomForestClassifier
testing_scoreRF=cross_val_score(RandomForestClassifier(n_estimators=300),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10)
testing_meanScore_RF=testing_scoreRF.mean()
testing_precision_RF=cross_val_score(RandomForestClassifier(n_estimators=300),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10,scoring='precision')
testing_recall_RF=cross_val_score(RandomForestClassifier(n_estimators=300),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10,scoring='recall')
testing_f1_RF=cross_val_score(RandomForestClassifier(n_estimators=300),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10,scoring='f1')

#K fold cross validation on Dicision Tree
from sklearn.tree import DecisionTreeClassifier
testing_scoreDT=cross_val_score( DecisionTreeClassifier(criterion='entropy'),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10)
testing_meanScore_DT=testing_scoreDT.mean()
testing_precision_DT=cross_val_score(DecisionTreeClassifier(criterion='entropy'),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10,scoring='precision')
testing_recall_DT=cross_val_score(DecisionTreeClassifier(criterion='entropy'),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10,scoring='recall')
testing_f1_DT=cross_val_score(DecisionTreeClassifier(criterion='entropy'),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10,scoring='f1')

#K fold cross validation with KNN
from sklearn.neighbors import KNeighborsClassifier
testing_scoreKNN=cross_val_score( KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2),norm_whole_data,finalDf['Survived'],cv=10)
testing_meanScore_KNN=testing_scoreKNN.mean()
testing_precision_KNN=cross_val_score( KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10,scoring='precision')
testing_recall_KNN=cross_val_score( KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10,scoring='recall')
testing_f1_KNN=cross_val_score( KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10,scoring='f1')

#K fold cross validation on Guassain Naive Bayes
from sklearn.naive_bayes import GaussianNB
testing_scoreGuassianNB=cross_val_score( GaussianNB(),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10)
testing_meanScore_GuassianNB=testing_scoreGuassianNB.mean()
testing_precision_GuassianNB=cross_val_score( GaussianNB(),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10,scoring='precision')
testing_recall_GuassianNB=cross_val_score( GaussianNB(),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10,scoring='recall')
testing_f1_GuassianNB=cross_val_score( GaussianNB(),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10,scoring='f1')

#K fold cross validation on Bernoulli Naive Bayes
from sklearn.naive_bayes import BernoulliNB
testing_scoreBernoulliNB=cross_val_score( BernoulliNB(),X_test,Y_test,cv=10)
testing_meanScore_BernoulliNB=testing_scoreBernoulliNB.mean()
testing_precision_BernoulliNB=cross_val_score(BernoulliNB(),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10,scoring='precision')
testing_recall_BernoulliNB=cross_val_score(BernoulliNB(),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10,scoring='recall')
testing_f1_BernoulliNB=cross_val_score(BernoulliNB(),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10,scoring='f1')

#K fold cross validation on Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
testing_scoreSGD=cross_val_score( SGDClassifier(loss="hinge", penalty="l2", max_iter=1000),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10)
testing_meanScore_SGD=testing_scoreSGD.mean()
testing_precision_SGD=cross_val_score(SGDClassifier(loss="hinge", penalty="l2", max_iter=1000),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10,scoring='precision')
testing_recall_SGD=cross_val_score(SGDClassifier(loss="hinge", penalty="l2", max_iter=1000),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10,scoring='recall')
testing_f1_SGD=cross_val_score(SGDClassifier(loss="hinge", penalty="l2", max_iter=1000),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10,scoring='f1')

#K fold cross validation on Multi Layered Perceptron
from sklearn.neural_network import MLPClassifier
testing_scoreMLP=cross_val_score(  MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,), random_state=1),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10)
testing_meanScore_MLP=testing_scoreMLP.mean()
testing_precision_MLP=cross_val_score( MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,), random_state=1),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10,scoring='precision')
testing_recall_MLP=cross_val_score( MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,), random_state=1),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10,scoring='recall')
testing_f1_MLP=cross_val_score( MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,), random_state=1),finalDf.drop(['Survived'],axis=1),finalDf['Survived'],cv=10,scoring='f1')

# Drawing table for K folds Cross Validation Report
from prettytable import PrettyTable
x=PrettyTable()  
x.field_names = ["Model", "Max Score", "Avg Score"]
x.add_row(["Logistic Regression", testing_scoreDT.max(), testing_meanScore_DT])
x.add_row(["Support Vector Machine",  testing_scoreSVM.max(), testing_meanScore_SVM])
x.add_row(["Guassian Naive Bayes",  testing_scoreGuassianNB.max(), testing_meanScore_GuassianNB])
x.add_row(["Bernoulli Naive Bayes", testing_scoreBernoulliNB.max(), testing_meanScore_BernoulliNB])
x.add_row(["K nearest Neighbours",  testing_scoreKNN.max(),  testing_meanScore_KNN])
x.add_row(["Stochastic Gradient Descent", testing_scoreSGD.max(),testing_meanScore_SGD])
x.add_row(["Decision Tree",  testing_scoreDT.max(),testing_meanScore_DT])
x.add_row(["Random Forest", testing_scoreRF.max(), testing_meanScore_DT])
x.add_row(["Multi layered Pereceptron", testing_scoreMLP.max(), testing_meanScore_MLP])
print(x)

# Drawing table for Train Test Split Report
from prettytable import PrettyTable    
y = PrettyTable()
y.field_names = ["Model", "Score"]
y.add_row(["Logistic Regression", metrics.accuracy_score(Y_test,predictedLR)])
y.add_row(["Support Vector Machine",  metrics.accuracy_score(Y_test,predictedSVC)])
y.add_row(["Guassian Naive Bayes",  metrics.accuracy_score(Y_test,predictedGuassianNB)])
y.add_row(["Bernoulli Naive Bayes", metrics.accuracy_score(Y_test,predictedBernoulliNB)])
y.add_row(["K nearest Neighbours",  metrics.accuracy_score(Y_test,predictedKNN)])
y.add_row(["Stochastic Gradient Descent",metrics.accuracy_score(Y_test,predictedSGD)])
y.add_row(["Decision Tree",  metrics.accuracy_score(Y_test,predictedDT)])
y.add_row(["Random Forest", metrics.accuracy_score(Y_test,predictedRF)])
y.add_row(["Multi layered Pereceptron", metrics.accuracy_score(Y_test,predictedMLP)])
print(y)


