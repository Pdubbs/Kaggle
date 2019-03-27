library(dplyr)
library(caret)
setwd('~/Git/Kaggle/career-con-2019')
source('helper_functions.R')
sample_sub <- read.csv('data/sample_submission.csv')
train_x <- read.csv('data/X_train.csv')
train_y <- read.csv('data/y_train.csv')
test_x <- read.csv('data/X_test.csv')

head(train_x)
length(unique(train_x$series_id))
head(train_y)
table(train_y$group_id,train_y$surface)

train_x <- make_trainvars(train_x)
train_mod <- data.frame(
  train_x[,-1] %>%
  group_by(series_id) %>%
  summarise_all( list(~min(.), ~max(.), ~mean(.), ~sd(.), ~abs_range(.)) )
)
uniques <- sapply(train_mod,function(x) length(unique(x)))
drops <- names(uniques[uniques==1])
train_mod <- train_mod[,!colnames(train_mod)%in%drops]

control <- trainControl(method="repeatedcv"
                       , number=10
                       , repeats=3)
grid <- expand.grid(.mtry=3:sqrt(ncol(train_mod)))
metric <- "Accuracy"

set.seed(3231)
mod <- train(train_mod[,-1]
            ,train_y$surface
            ,method = "rf"
            ,metric = metric
            ,trControl = control
            ,tuneGrid = grid
            ,preProcess = "YeoJohnson"
            ,ntree=1000)

euler_test_orient <- quaternion_mat(test_x,'orientation')
test_x <- make_trainvars(test_x)
test_mod <- data.frame(
  test_x[,-1] %>%
  group_by(series_id) %>%
  summarise_all( list(~min(.), ~max(.), ~mean(.), ~sd(.), ~abs_range(.)) )
)
test_mod <- test_mod[,!colnames(test_mod)%in%drops]

sub <- sample_sub
preds <- predict(mod,test_mod[,-1])
sub$surface <- preds
write.csv(sub,"submission.csv",row.names = FALSE)