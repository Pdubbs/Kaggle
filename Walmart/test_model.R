library(gbm)
library(Matrix)
library(lubridate)

setwd("/Users/Pwyatt/Documents/GitHub/Kaggle/Walmart")

#read in the data
train <- read.csv("train.csv")
test <- read.csv("test.csv")
weather <- read.csv("weather.csv")
keys <- read.csv("key.csv")
samp <- read.csv("sampleSubmission.csv")

#recode trace amounts as a small number
weather$snowfall <- as.character(weather$snowfall)
weather$snowfall[weather$snowfall=="T"] <- .01
weather$preciptotal <- as.character(weather$preciptotal)
weather$preciptotal[weather$preciptotal=="T"] <- .004
weather$depart_missing <- 0
weather$depart_missing[weather$depart=="M"] <- 1

#really shitty imputation; factor values get recoded as character and numeric, then the NA values (where there was a value that could not get encoed as numeric) get replaced by means
for(i in c(3:12,14:18,20)){
  weather[,i] <- as.numeric(as.character(weather[,i]))
  weather[is.na(weather[,i]),i] <- mean(weather[,i],na.rm=TRUE)
}

#loop predictions item-by item because I'm working on a macbook air
#hierarchical store-level effects should be included eventually
preds<-NULL
for(item in unique(test$item_nbr)){
  train_sub <- train[train$item_nbr==item,] #select the item
  train_sub <- merge(train_sub,keys) #merge keys to the training set so that I know what station matches up to a store
  train_sub <- merge(train_sub,weather) #merge weather into the set
  test_sub <- test[test$item_nbr==item,] #same shit with test
  test_sub <- merge(test_sub,keys)
  test_sub <- merge(test_sub,weather)
  test_sub$units <- 0 #add units so I can stack them (so model matrix works right)

  mdat <- rbind(train_sub,test_sub)
  mdat$month <- factor(month(mdat$date))
  modelMat <- model.matrix(units ~ .-date-1,mdat) #model matrix turns factors into binaries

  trainMat <- modelMat[1:nrow(train_sub),]
  target <- train_sub$units
  table(target) #crazy outliers exist in the total dataset, I want to control for that
  sdt <- sd(target[target>0])
  mt <- mean(target[target>0])
  target[target>(mt+4*sdt)] <- (mt+4*sdt)
  hist(target) #soooooo many 0s
  hist(target[target>0]) #the data overall is log distributed, later I'll do box-cox or choose transformation on the fly
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


