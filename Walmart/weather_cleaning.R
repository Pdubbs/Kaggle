library(lubridate)

weather <- read.csv("weather.csv")
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

weather$month <- factor(month(mdat$date))
weather$wday <- wday(mdat$date) #weekend shopping spree!

