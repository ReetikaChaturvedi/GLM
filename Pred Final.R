#Logistic regression

## Importing the necessary libraries
rm(list=ls(all=TRUE))
library(dplyr)
library(data.table)

# Imported the dataset
train<- mnist_train
test<- mnist_test

train <- fread("train") 
a=data.frame(names(train))
a
names(train)[1] <- "label" 

##PART A
train$e_0 <- ifelse(train$label == 0, 1, 0) 
#train$e_0
train$e_1 <- ifelse(train$label == 1, 1, 0) 
#train$e_1
train$e_2 <- ifelse(train$label == 2, 1, 0) 
train$e_3 <- ifelse(train$label == 3, 1, 0) 
train$e_4 <- ifelse(train$label == 4, 1, 0) 
train$e_5 <- ifelse(train$label == 5, 1, 0) 
train$e_6 <- ifelse(train$label == 6, 1, 0) 
train$e_7 <- ifelse(train$label == 7, 1, 0) 

train$e_8 <- ifelse(train$label == 8, 1, 0) 
train$e_9 <- ifelse(train$label == 9, 1, 0) 


##PART B
glm_train <- glm(label ~. ,data = train)
glm_train
summary<-summary(glm_train)
summary

##PART C
library(tidyverse)
s_train <- sample_n(train, 10000)
s_train


#
#CREATING random samples subsetting with all labels
train_0 <-  subset(s_train, select = -c(label,e_1,e_2,e_3,e_4,e_5,e_6,e_7,e_8,e_9))
train_0
dim(train_0)
train_1 <-  subset(s_train, select = -c(label,e_0,e_2,e_3,e_4,e_5,e_6,e_7,e_8,e_9))
train_2 <-  subset(s_train, select = -c(label,e_1,e_0,e_3,e_4,e_5,e_6,e_7,e_8,e_9))
train_3 <-  subset(s_train, select = -c(label,e_1,e_2,e_0,e_4,e_5,e_6,e_7,e_8,e_9))
train_4 <-  subset(s_train, select = -c(label,e_1,e_2,e_3,e_0,e_5,e_6,e_7,e_8,e_9))
train_5 <-  subset(s_train, select = -c(label,e_1,e_2,e_3,e_4,e_0,e_6,e_7,e_8,e_9))
train_6 <-  subset(s_train, select = -c(label,e_1,e_2,e_3,e_4,e_5,e_0,e_7,e_8,e_9))
train_7 <-  subset(s_train, select = -c(label,e_1,e_2,e_3,e_4,e_5,e_6,e_0,e_8,e_9))
train_8 <-  subset(s_train, select = -c(label,e_1,e_2,e_3,e_4,e_5,e_6,e_7,e_0,e_9))
train_9 <-  subset(s_train, select = -c(label,e_1,e_2,e_3,e_4,e_5,e_6,e_7,e_8,e_0))


#checking test data
dim(test)
dim(test)
b<-test[,-1]
#b
length(b)

##building regression models and predicting probabilities using test data


glm_0 <-  glm(e_0 ~. ,data = train_0,family=binomial(link=logit))
glm_0

e_0_pred= predict(glm_0, b,type = "response")
e_0_pred


############# product-1(trouser)###############

glm_1 <-  glm(e_1 ~. ,data = train_1)
e_1_pred= predict(glm_1, b,type = "response")
e_1_pred

############# product-2(pull over)################

glm_2 <-  glm(e_2 ~. ,data = train_2)
e_2_pred= predict(glm_2, b)



############# product-3(dress)###################

glm_3 <-  glm(e_3 ~. ,data = train_3)
e_3_pred= predict(glm_3, b)


############# product-4(coat)########################

glm_4 <-  glm(e_4 ~. ,data = train_4)
e_4_pred= predict(glm_4, b)


############# product-5(Sandal)#####################

glm_5 <-  glm(e_5 ~. ,data = train_5)
e_5_pred= predict(glm_5, b)

############# product-6(shirt)#####################

glm_6 <-  glm(e_6 ~. ,data = train_6)
e_6_pred= predict(glm_6, b)


############# product-7(sneaker)#################

glm_7 <-  glm(e_7 ~. ,data = train_7)
e_7_pred= predict(glm_7, b)


############# product-8(bag)#####################

glm_8 <-  glm(e_8 ~. ,data = train_8)
e_8_pred= predict(glm_8, b)
e_8_pred

############# product-9(Ankleboot)##############

glm_9 <-  glm(e_9 ~. ,data = train_9)
e_9_pred= predict(glm_9, b)



#####softmax####

z<- cbind(e_0_pred,e_1_pred,e_2_pred,e_3_pred,e_4_pred,e_5_pred,e_6_pred,e_7_pred,e_8_pred,e_9_pred)
z
dim(z)
exp(z)
##############softmax <- exp(z)/sum(exp(z))################

nr <- exp(z)
nr
dr <- apply(nr,MARGIN = 1,FUN = sum)
dr
sfmax<-nr/dr
sfmax
sfmax[1,]
sum(sfmax[1,])
colnames(sfmax)<-c("0","1","2","3","4","5","6","7","8","9")

#most probable product for the each row item
maxprob <-colnames(sfmax)[apply(sfmax,1,which.max)]
maxprob

#confusion matrix
library(caret)
cfm<-confusionMatrix(factor(test$V1),factor(maxprob))
cfm
