#initial test submission to http://www.kaggle.com/c/afsis-soil-properties/
#adapted from example code at http://afsiskaggle.qed.ai/
setwd("C:\\Users\\User\\Documents\\Misc\\Kaggle\\AfricanSoil")
library(randomForest)
trainingdata <- read.csv("training.csv")
testdata <- read.csv("sorted_test.csv")

soil_properties <- c("Ca", "P", "pH", "SOC", "Sand")


names(trainingdata)[2656:2670]
mir_measurements <- trainingdata[, 2:2655]
mir_der <- mir_measurements - cbind(NA, mir_measurements)[, -(dim(mir_measurements)[2]+1)]
train <- cbind(trainingdata[, 3580:3595], mir_der[,-1])
mir_measurements <- trainingdata[, 2671:3579]
mir_der <- mir_measurements - cbind(NA, mir_measurements)[, -(dim(mir_measurements)[2]+1)]
train <- cbind(train, mir_der[, -1])

mir_measurements <- testdata[, 2:2655]
mir_der <- mir_measurements- cbind(NA, mir_measurements)[, -(dim(mir_measurements)[2]+1)]
test <- cbind(testdata[, 3580:3595], mir_der[,-1])
mir_measurements <- testdata[, 2671:3579]
mir_der <- mir_measurements- cbind(NA, mir_measurements)[, -(dim(mir_measurements)[2]+1)]
test <- cbind(test, mir_der[, -1])

predictions <- as.data.frame(matrix(0,nrow(test),5))
i<-1
for(soil_property in soil_properties){
  model <- randomForest(train, trainingdata[, soil_property], ntree = 30)
  predictions[,i] <- cbind(predictions, predict(model,test))
  i<-i+1
}

colnames(predictions) <-  soil_properties
cbind(PIDN=as.character(testdata[,1]), predictions)[1,]
write.csv(cbind(PIDN= as.character(testdata[,1]), predictions), "predictions.csv", row.names=FALSE)

