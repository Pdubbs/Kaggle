library(gbm)
library(Matrix)
library(lubridate)

setwd("/Users/Pwyatt/Documents/GitHub/Kaggle/Walmart")
train <- read.csv("train.csv")
test <- read.csv("test.csv")
weather <- read.csv("weather.csv")
keys <- read.csv("key.csv")
samp <- read.csv("sampleSubmission.csv")

weather$snowfall <- as.character(weather$snowfall)
weather$snowfall[weather$snowfall=="T"] <- .01
weather$preciptotal <- as.character(weather$preciptotal)
weather$preciptotal[weather$preciptotal=="T"] <- .01
weather$depart_missing <- 0
weather$depart_missing[weather$depart=="M"] <- 1

for(i in c(3:12,14:18,20)){
  weather[,i] <- as.numeric(as.character(weather[,i]))
  weather[is.na(weather[,i]),i] <- mean(weather[,i],na.rm=TRUE)
}


preds<-NULL
for(item in unique(test$item_nbr)){
  train_sub <- train[train$item_nbr==item,]
  train_sub <- merge(train_sub,keys)
  train_sub <- merge(train_sub,weather)
  test_sub <- test[test$item_nbr==item,]
  test_sub <- merge(test_sub,keys)
  test_sub <- merge(test_sub,weather)
  test_sub$units <- 0

  mdat <- rbind(train_sub,test_sub)
  mdat$month <- factor(month(mdat$date))
  modelMat <- model.matrix(units ~ .-date-1,mdat)

  trainMat <- modelMat[1:nrow(train_sub),]
  target <- train_sub$units
  table(target) #crazy outliers
  sdt <- sd(target[target>0])
  mt <- mean(target[target>0])
  target[target>(mt+4*sdt)] <- (mt+4*sdt)
  hist(target) #soooooo many 0s
  hist(target[target>0]) #super log distributed
  target <- log(target+.1)
  hist(target) #didn't know why I thought that'd be different
  hist(target[target>-2]) #more normalish

  mod <- gbm.fit(trainMat, target, distribution = "gaussian")

  testMat <- modelMat[(nrow(train_sub)+1):nrow(modelMat),]
  test_pred <- (exp(predict(mod,testMat,n.trees=100))-.1)
  test_pred[test_pred<0] <- 0
  test_pred <- cbind(test_pred, test_sub$store_nbr)
  test_pred <- data.frame(test_pred)
  test_pred$date <- test_sub$date
  test_pred$item_nmbr <- item
  colnames(test_pred) <- c("pred", "store_nbr", "date", "item")
  preds <- rbind(preds,test_pred)
}

ids <- paste(preds$store_nbr,preds$item,preds$date,sep="_")
sub <- data.frame(cbind(ids,preds$pred))
colnames(sub) <- c("id","units")
write.csv(sub,"sub_gbm_g.csv",row.names=FALSE)


