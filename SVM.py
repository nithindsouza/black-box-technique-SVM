#################################Problem 1 #############################
#importing required package
import pandas as pd
import numpy as np

#loading dataset
salary_train = pd.read_csv("C:/Users/hp/Desktop/SVM assi/SalaryData_Train.csv")
salary_test = pd.read_csv("C:/Users/hp/Desktop/SVM assi/SalaryData_Test.csv")

#EDA
salary_train.describe()
salary_test.describe()

#label encoding the data
from sklearn import preprocessing
L_enc = preprocessing.LabelEncoder()
salary_train.columns
salary_train['workclass'] = L_enc.fit_transform(salary_train['workclass'])
salary_train['education'] = L_enc.fit_transform(salary_train['education'])
salary_train['maritalstatus'] = L_enc.fit_transform(salary_train['maritalstatus'])
salary_train['occupation'] = L_enc.fit_transform(salary_train['occupation'])
salary_train['relationship'] = L_enc.fit_transform(salary_train['relationship'])
salary_train['race'] = L_enc.fit_transform(salary_train['race'])
salary_train['sex'] = L_enc.fit_transform(salary_train['sex'])
salary_train['native'] = L_enc.fit_transform(salary_train['native'])

salary_test.columns
salary_test['workclass'] = L_enc.fit_transform(salary_test['workclass'])
salary_test['education'] = L_enc.fit_transform(salary_test['education'])
salary_test['maritalstatus'] = L_enc.fit_transform(salary_test['maritalstatus'])
salary_test['occupation'] = L_enc.fit_transform(salary_test['occupation'])
salary_test['relationship'] = L_enc.fit_transform(salary_test['relationship'])
salary_test['race'] = L_enc.fit_transform(salary_test['race'])
salary_test['sex'] = L_enc.fit_transform(salary_test['sex'])
salary_test['native'] = L_enc.fit_transform(salary_test['native'])

#preparing dataset
train_X = salary_train.iloc[:, :13]
train_y = salary_train.iloc[:, 13]
test_X  = salary_test.iloc[:, :13]
test_y  = salary_test.iloc[:, 13]

#building SVM model
# kernel = linear
from sklearn.svm import SVC
model_linear = SVC(kernel = "linear")
#fitting the model
model_linear.fit(train_X, train_y)
#predicting on tests data
pred_test_linear = model_linear.predict(test_X)

#model evaluation accuracy
np.mean(pred_test_linear == test_y)

# kernel = rbf (improving model)
model_rbf = SVC(kernel = "rbf")
#fitting the model
model_rbf.fit(train_X, train_y)
#predicting on test data
pred_test_rbf = model_rbf.predict(test_X)
#model evaluation (accuracy)
np.mean(pred_test_rbf==test_y)

##########################################Problem 2####################
#importing required package
import pandas as pd
import numpy as np

forestfire_data = pd.read_csv("C:/Users/hp/Desktop/SVM assi/forestfires.csv")
forestfire_data.describe()

#droping first 2 columns since dummies area lready created in dataset
forestfire_data = forestfire_data.iloc[:,2:]

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

train,test = train_test_split(forestfire_data, test_size = 0.20)

#preparing dataset
train_X = train.iloc[:, :28]
train_y = train.iloc[:, 28]
test_X  = test.iloc[:, :28]
test_y  = test.iloc[:, 28]

#EDA
train.describe()
test.describe()

#building SVM model
# kernel = linear
from sklearn.svm import SVC
model_linear = SVC(kernel = "linear")
#fitting the model
model_linear.fit(train_X, train_y)
#predicting on tests data
pred_test_linear = model_linear.predict(test_X)

#model evaluation accuracy
np.mean(pred_test_linear == test_y)

# kernel = rbf (improving model)
model_rbf = SVC(kernel = "rbf")
#fitting the model
model_rbf.fit(train_X, train_y)
#predicting on test data
pred_test_rbf = model_rbf.predict(test_X)
#model evaluation (accuracy)
np.mean(pred_test_rbf==test_y)
######################################END#######################################