# ------------------------------------------------------------------ #
# This script does the following:
# 1) estimate internal noise, using data from control/training task
# 2) transform stimuli in internal noise units (correcting for bias)
# 3) remove sjs that have large bias & poor performance (2 in total)
# Matteo Lisi, 2019
# ------------------------------------------------------------------ #

# clear workspace
rm(list=ls())

# set directory
setwd("D:/xiangmu/scripts/R")

# load everything (noise estimate made based also on control task)
d <- read.table("D:/xiangmu/data/behavior_for_behavior/data_for_behavior.csv",header=T,sep=",")# D:/xiangmu/data/behavior/idsess/data.csv #D:/xiangmu/data/behavior/idsess/data.csv
tapply(d$decision,list(d$decision,d$task),length)

# this select only decisions with **equal prior probabilities**
# (both subjectively and objectively)
# i.e. during control task or 430 decision in dual-decision task
d$keep <- ifelse(d$decision==1 | d$dual==0, 1, 0)#（修改了0变成1[改回来了]）
tapply(d$keep, list(d$task, d$decision,d$dual), sum) # sanity check

# exclude the rest
d <- d[d$keep==1,]
d$acc <- as.numeric(d$acc)
# load functions
source("./R/noise_estimation_functions.R")

# ------------------------------------------------------------------ #
# do estimation 
d_noise <- {}
for(i in unique(d$id)){
  cat("/n",i)
  FT <- estimateNoise(d[d$id==i,])
  
  d_line <- data.frame(t(FT))
  d_line$id <- i
  
  d_noise <- rbind(d_noise, d_line)
  write.table(d_noise, file="D:/xiangmu/data/behavior_for_behavior/noise_estimates.txt")#D:/xiangmu/data/behavior/idsess/noise_estimates.txt
}

# display noise estimates
data.frame(d_noise$id, round(d_noise[,c(1,4)],digits=2))
# d_noise <- read.table("D:/Lisi/data_34/noise_estimates.txt", header=T)

# ------------------------------------------------------------------ #
# now convert the stimuli in units of internal noise
# ------------------------------------------------------------------ #
# now convert the stimuli in units of internal noise
d <- read.table("D:/xiangmu/data/behavior_for_behavior/data_for_behavior.csv",header=T,sep=",")#D:/xiangmu/data/behavior/idsess/data.csv
d$s <- NA
d$sign_s <- NA
for(i in unique(d$id)){
  d_line <- d_noise[which(d_noise$id==i),]
  
  # convert stimuli to noise sd (430 bias is subtracted)
  d$s[d$id==i] <- ifelse(d$task[d$id==i]=="motion",
                         (d$signed_coherence[d$id==i]  - d_line$bias_m) / d_line$sigma_m,
                         (d$signed_mu[d$id==i]  - d_line$bias_o) / d_line$sigma_o)
  d$sign_s[d$id==i] <- sign(d$s[d$id==i])
}

# ------------------------------------------------------------------ #
# check participant biases
tapply(d_noise$bias_o, d_noise$id, mean)  # high bias: "sj400"
tapply(d_noise$bias_m, d_noise$id, mean)  # high bias: "sj1500"


# # "large" in the sense that it is more than 2SD from group mean
# excl_bias_o <- d_noise$id[abs((d_noise$bias_o-mean(d_noise$bias_o))/sd(d_noise$bias_o))>2.5]
# excl_bias_m <- d_noise$id[abs((d_noise$bias_m-mean(d_noise$bias_m))/sd(d_noise$bias_m))>2.5]
# if (!(is.integer(excl_bias_o) && length(excl_bias_o) == 0)){
#   d <- d[d$id!=excl_bias_o,]
# }
# if (!(is.integer(excl_bias_m) && length(excl_bias_m) == 0)){
#   d <- d[d$id!=excl_bias_m,]
# }



# ------------------------------------------------------------------ #
# save dataset noise units
write.table(d, file="D:/xiangmu/data/behavior_for_behavior/data_nu.txt", quote=F, row.names=F, sep=";")## ------------------------------------------------------------------ #D:/xiangmu/data/behavior/idsess/data_nu.txt
