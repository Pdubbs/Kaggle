library(dplyr)
library(caret)
setwd('~/Git/Kaggle/career-con-2019')
sample_sub <- read.csv('data/sample_submission.csv')
train_x <- read.csv('data/X_train.csv')
train_y <- read.csv('data/y_train.csv')
test_x <- read.csv('data/X_test.csv')

head(train_x)
length(unique(train_x$series_id))
head(train_y)
table(train_y$group_id,train_y$surface)

train_mod <- train_x[,-1] %>%
  group_by(series_id) %>%
  summarise_all(list(~min(.), ~max(.), ~mean(.), ~sd(.)))

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
            ,tuneGrid = tunegrid
            ,ntree=500)

test_mod <- test_x[,-1] %>%
  group_by(series_id) %>%
  summarise_all(list(~min(.), ~max(.), ~mean(.), ~sd(.)))
sub <- sample_sub
preds <- predict(mod,test_mod)
sub$surface <- preds
write.csv(sub,"submission.csv",row.names = FALSE)
