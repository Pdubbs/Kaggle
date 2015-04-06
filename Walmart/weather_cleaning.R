library(lubridate)
library(plyr)

weather <- read.csv("weather.csv")
#recode trace amounts as a small number
weather$snowfall <- as.character(weather$snowfall)
weather$snowfall[weather$snowfall=="T"] <- .01
weather$preciptotal <- as.character(weather$preciptotal)
weather$preciptotal[weather$preciptotal=="T"] <- .004
weather$depart_missing <- 0
weather$depart_missing[weather$depart=="M"] <- 1
weather$date<-ymd(weather$date)

#really shitty imputation; factor values get recoded as character and numeric, then the NA values (where there was a value that could not get encoed as numeric) get replaced by means
for(i in c(3:12,14:18,20)){
  weather[,i] <- as.numeric(as.character(weather[,i]))
  weather[is.na(weather[,i]),i] <- mean(weather[,i],na.rm=TRUE)
}

weather$month <- factor(month(weather$date))
weather$wday <- wday(weather$date) #weekend shopping spree!

#variables about the future forecast:
weather$nw_precip<-
  apply(weather,1,function(x) { 
    sum(weather$preciptotal[weather$date > ymd(x["date"]) & 
                              weather$date < (ymd(x["date"])+days(7)) &
                              weather$station_nbr == as.numeric(x["station_nbr"])] 
                            )}
    )

weather<-ddply(weather,.(date,station_nbr),mutate, 
         nw_precip = sum(weather$preciptotal[weather$date > date &
                                               weather$date < (date+days(7)) &
                                               weather$station_nbr == station_nbr]),
         nw_snow = sum(weather$snowfall[weather$date > date &
                                          weather$date < (date+days(7)) &
                                          weather$station_nbr == station_nbr]),
         nw_high = max(weather$tmax[weather$date > date &
                                      weather$date < (date+days(7)) &
                                      weather$station_nbr == station_nbr]),
         nw_low = sum(weather$tmin[weather$date > date &
                                      weather$date < (date+days(7)) &
                                      weather$station_nbr == station_nbr])
         )

weather$nw_high[!is.finite(weather$nw_high)]<-mean(weather$nw_high[is.finite(weather$nw_high)],na.rm=TRUE)
