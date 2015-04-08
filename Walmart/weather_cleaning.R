library(lubridate)
library(plyr)
library(data.table)

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
weather$wday <- factor(wday(weather$date)) #weekend shopping spree! #SW changed this to factor

#variables about the future forecast:
weather$nw_precip<-
  apply(weather,1,function(x) { 
    sum(weather$preciptotal[weather$date > ymd(x["date"]) & 
                              weather$date < (ymd(x["date"])+days(7)) &
                              weather$station_nbr == as.numeric(x["station_nbr"])] 
                            )}
    )
    
##DDply PREDICTIONS

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

##Create "daylightMins" variable
weather$sunriseMins <- as.numeric(substr(weather$sunrise,1,2))*60 + 
  as.numeric(substr(weather$sunrise,3,4)) 

weather$sunsetMins <- as.numeric(substr(weather$sunset,1,2))*60 + 
  as.numeric(substr(weather$sunset,3,4)) 

weather$daylightMins <- weather$sunsetMins - weather$sunriseMins 

##WEATHER EVENT BINARIES

weather$codesum <- as.character(weather$codesum)

weatherEventList <- c()

for (i in 1:nrow(weather)) {
  if (length(weather$codesum[i] > 0)) {
    weatherLineSplit <- strsplit(weather$codesum[i]," ")
    for (j in 1:length(weatherLineSplit)) {
      if (!is.na(weatherLineSplit[[1]][j])) {
        weatherEventList <- c(weatherEventList,weatherLineSplit[[1]][j])   
      }
 
    }
  }
}

weatherEventList <- unique(weatherEventList)

#http://stackoverflow.com/questions/13201081/how-to-subset-from-a-long-list?rq=1

weatherEventList <- weatherEventList[sapply(weatherEventList,nchar)>1] 

#http://stackoverflow.com/questions/18214395/r-add-empty-columns-to-a-dataframe-with-specified-names-from-a-vector

weather[,weatherEventList] <- 0

#http://stackoverflow.com/questions/7227976/using-grep-in-r-to-find-strings-as-whole-words-but-not-strings-as-part-of-words

for (event in weatherEventList) {
  for (i in 1:nrow(weather)) {
    if (grepl(paste0("\\b",event,"\\b"),weather$codesum[i])) {
    weather[i,event] <- 1
    }
  }
}

setnames(weather,"RA","Rain")
setnames(weather,"UP","Unknown_Precipitation")
setnames(weather,"FG+","Heavy_Fog")
setnames(weather,"SN","Snow")
setnames(weather,"BR","Mist")
setnames(weather,"FG","Fog")
setnames(weather,"TSRA","Thunderstorm_Rain")
setnames(weather,"TS","Thunderstorm")
setnames(weather,"DZ","Drizzle")
setnames(weather,"FZRA","Freezing_Rain")
setnames(weather,"FZDZ","Freezing_Drizzle")
setnames(weather,"HZ","Haze")
setnames(weather,"BLDU","Blowing_Widespread_Dust")
setnames(weather,"DU","Widespread_Dust")
setnames(weather,"FG","Fog")
setnames(weather,"BCFG","Patches_Fog")
setnames(weather,"MIFG","Shallow_Fog")
setnames(weather,"HZ","Haze")
setnames(weather,"FZFG","Freezing_Fog")
setnames(weather,"FU","Smoke")
setnames(weather,"VCTS","Vicinity_Thunderstorm")
setnames(weather,"TSSN","Thunderstorm_Snow")
setnames(weather,"SG","Snow_Grains")
setnames(weather,"VCFG","Vicinity_Fog")
setnames(weather,"PRFG","Partial_Fog")
setnames(weather,"BLSN","Blowing_Snow")
setnames(weather,"GS","Small_Hail_or_Snow_Pellets")
