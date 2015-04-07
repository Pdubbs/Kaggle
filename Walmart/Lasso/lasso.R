library(glmnet)
library(Matrix)
library(lubridate)

setwd("/Users/Pwyatt/Documents/GitHub/Kaggle/Walmart")

#read in the data
train <- read.csv("train.csv")
test <- read.csv("test.csv")
keys <- read.csv("key.csv")
samp <- read.csv("sampleSubmission.csv")
train$date <- ymd(train$date)
test$date <- ymd(test$date)

source("/Users/Pwyatt/Documents/GitHub/Kaggle/Walmart/weather_cleaning.R")

#loop predictions item-by item because I'm working on a macbook air
#hierarchical store-level effects should be included eventually
preds<-NULL
for(item in unique(test$item_nbr)){
  train_sub <- train[train$item_nbr==item,] #select the item
  train_sub <- merge(train_sub,keys) #merge keys to the training set so that I know what station matches up to a store
  train_sub <- merge(train_sub,weather) #merge weather into the set
  test_sub <- test[test$item_nbr==item,] #same procedure with test
  test_sub <- merge(test_sub,keys)
  test_sub <- merge(test_sub,weather)
  test_sub$units <- 0 #add units so I can stack them (so model matrix works right)
  
  mdat <- rbind(train_sub,test_sub)
  
  mdat$month <- factor(month(mdat$date))
  mdat$wday <- wday(mdat$date) #weekend shopping spree!
  
  modelMat <- model.matrix(units ~ .-date-1,mdat) #model matrix turns factors into binaries
  
  trainMat <- modelMat[1:nrow(train_sub),]
  target <- train_sub$units
  sdt <- sd(target[target>0]) #crazy outliers exist in the total dataset, I want to control for that
  mt <- mean(target[target>0])
  target[target>(mt+4*sdt)] <- (mt+4*sdt)
  target <- log(target+.1) #the data overall is log distributed, later I'll do box-cox or choose transformation on the fly for each item
  
  mod <- cv.glmnet(trainMat, target, family= "gaussian") #probably will end up choosing gaussian or poisson on the fly
  
  testMat <- modelMat[(nrow(train_sub)+1):nrow(modelMat),]
  test_pred <- (exp(predict(mod,testMat))-.1)
  print(paste("item:", item)) 
  print(table(test_pred))
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
write.csv(sub,"lasso.csv",row.names=FALSE)


