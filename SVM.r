############################Problem 1################################# 
# Load the Dataset
salary_train <- read.csv("C://Users//hp//Desktop//SVM assi//SalaryData_Train.csv", stringsAsFactors = TRUE)
salary_test <- read.csv("C://Users//hp//Desktop//SVM assi//SalaryData_Test.csv", stringsAsFactors = TRUE)

summary(salary_train)
summary(salary_test)


#Training a model on the data
#Begin by training a simple linear SVM
install.packages("kernlab")
library(kernlab)

salary_classifier <- ksvm(Salary ~ ., data = salary_train, kernel = "vanilladot")

#Evaluating model performance
#predictions on testing dataset
salary_predictions <- predict(salary_classifier, salary_test)

table(salary_predictions, salary_test$Salary)
agreement <- salary_predictions == salary_test$Salary
table(agreement)
prop.table(table(agreement))

#Improving model performance
salary_classifier_rbf <- ksvm(Salary ~ ., data = salary_train, kernel = "rbfdot")
salary_predictions_rbf <- predict(salary_classifier_rbf, salary_test)
agreement_rbf <- salary_predictions_rbf == salary_test$Salary
table(agreement_rbf)
prop.table(table(agreement_rbf))

####################################Problem 2################################
# Load the Dataset
library(readr)
forestfires <- read.csv("C://Users//hp//Desktop//SVM assi//forestfires.csv", stringsAsFactors = TRUE)

summary(forestfires)

# Partition Data into train and test data and also droping first 2 columns since already dummies are created in dataset
forestfires_train <- forestfires[1:362, 3:31]
forestfires_test  <- forestfires[363:517, 3:31]

#Training a model on the data
#Begin by training a simple linear SVM
install.packages("kernlab")
library(kernlab)

forestfires_classifier <- ksvm(size_category ~ ., data = forestfires_train, kernel = "vanilladot")

#Evaluating model performance
#predictions on testing dataset
forestfires_predictions <- predict(forestfires_classifier, forestfires_test)

table(forestfires_predictions, forestfires_test$size_category)
agreement <- forestfires_predictions == forestfires_test$size_category
table(agreement)
prop.table(table(agreement))

#Improving model performance
forestfires_classifier_rbf <- ksvm(size_category ~ ., data = forestfires_train, kernel = "rbfdot")
forestfires_predictions_rbf <- predict(forestfires_classifier_rbf, forestfires_test)
agreement_rbf <- forestfires_predictions_rbf == forestfires_test$size_category
table(agreement_rbf)
prop.table(table(agreement_rbf))

#############################END#############################################