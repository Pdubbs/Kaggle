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

weather$month <- factor(month(weather$date))
weather$wday <- factor(wday(weather$date)) #weekend shopping spree!


##Create "daylightMins" variable
weather$sunriseMins <- as.numeric(substr(weather$sunrise,1,2))*60 + 
  as.numeric(substr(weather$sunrise,3,4)) 

weather$sunsetMins <- as.numeric(substr(weather$sunset,1,2))*60 + 
  as.numeric(substr(weather$sunset,3,4)) 

weather$daylightMins <- weather$sunsetMins - weather$sunriseMins 