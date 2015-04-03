library(glmnet)
library(Matrix)

setwd("/Users/Pwyatt/Documents/GitHub/Kaggle/Walmart")
train <- read.csv("train.csv")
test <- read.csv("test.csv")
weather <- read.csv("weather.csv")
keys <- read.csv("key.csv")
samp <- read.csv("sampleSubmission.csv")

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
  for(i in c(6:15,17:23)){
    mdat[,i] <- as.numeric(as.character(mdat[,i]))
    mdat[is.na(mdat[,i]),i] <- mean(mdat[,i],na.rm=TRUE)
  }

  modelMat <- model.matrix(units ~ . -1,mdat)

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

  mod <- glmnet(trainMat, target, family="gaussian")

  testMat <- modelMat[(nrow(train_sub)+1):nrow(modelMat),]
  test_pred <- (exp(predict(mod,testMat))-.1)[,ncol(mod$beta)]
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
write.csv(sub,"sub.csv",row.names=FALSE)


