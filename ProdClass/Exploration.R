setwd("/users/pwyatt/Documents/GitHub/Kaggle/ProdClass/")

library(caret)
library(gbm)
library(glmnet)
library(ada)

train<-read.csv("train.csv")
test<-read.csv("test.csv")

table(train$target)

#mod<-train(train[,grepl("feat",colnames(train))],train$target,method = "ada")

LogLoss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}

LogLoss <- function(actual, predicted, eps=1e-15) {
  predicted[predicted < eps] <- eps;
  predicted[predicted > 1 - eps] <- 1 - eps;
  -1/nrow(actual)*(sum(actual*log(predicted)))
}

#grid<-expand.grid(n.trees=c(5,10),interaction.depth=c(1),shrinkage=.1)
#model<-train(train[,grepl("feat",colnames(train))], train$target, method = "gbm", tuneGrid=grid)
model<-gbm.fit(train[,grepl("feat",colnames(train))], train$target, distribution = "multinomial")
preds<-predict(model,test,n.trees=100,type="prob")
