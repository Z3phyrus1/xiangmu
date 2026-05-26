# ------------------------------------------------------------------ #
# This script does the following:
# 1) calculate model predictions 
# 2) save them in the dataset
# Matteo Lisi, 2020
# ------------------------------------------------------------------ #

rm(list=ls())
setwd("D:/xiangmu/scripts/R")
source("./R/code_all_models.R")

# ------------------------------------------------------#
# load data
d <- read.table("D:/xiangmu/data/behavior_for_behavior/data_nu.txt",header=T, sep=";")#D:/reysy/EEG/data/data_nu.txt

# exclude control sessions
d <- d[d$dual==1,]

# # ------------------------------------------------------#
# # put the data frames in a "wide" format
# d <- data.frame(s1=d$s[d$decision=="1"], s2=d$s[d$decision=="2"], r1=d$rr[d$decision=="1"], r2=d$rr[d$decision=="2"], acc1=d$acc[d$decision=="1"],acc2=d$acc[d$decision=="2"], id=d$id[d$decision=="1"],seq=d$seq, id_sess=d$id_sess, task=ifelse(is.na(d$meanTilt[d$decision=="1"]),"M-O","O-M"), dataset=1, rt1=d$tResp[d$decision=="1"],rt2=d$tResp[d$decision=="2"])
# ------------------------------------------------------#

# 构造唯一key，辅助匹配且保证顺序
d$key <- paste(d$id_sess, d$block, d$trial, d$id, d$seq, sep="_")

d1 <- d[d$decision==1 | d$decision=="1", ]
d2 <- d[d$decision==2 | d$decision=="2", ]

# 只保留双方都有的key，且保证合并后顺序等同于d1原排序
common_keys <- intersect(d1$key, d2$key)
d1 <- d1[match(common_keys, d1$key), ]
d2 <- d2[match(common_keys, d2$key), ]

# wide格式，顺序和d1一致
d <- data.frame(
  s1 = d1$s,
  s2 = d2$s,
  r1 = d1$rr,
  r2 = d2$rr,
  acc1 = d1$acc,
  acc2 = d2$acc,
  id = d1$id,
  seq = d1$seq,
  id_sess = d1$id_sess,
  task = ifelse(is.na(d1$meanTilt),"M-O","O-M"),
  dataset = 1,
  rt1 = d1$tResp,
  rt2 = d2$tResp
)

# ------------------------------------------------------#
# retrieve estimated parameters values
dfit <- read.table("D:/xiangmu/data/behavior_for_behavior/fit_par.txt",header=T)#D:/reysy/EEG/data/fit_par.txt

dfit$n_par <- ifelse(dfit$model=="bayes",0, ifelse(dfit$model=="metanoise" | dfit$model=="sub1", 1, ifelse(dfit$model=="sub2",2,4)))

# ------------------------------------------------------#
# preallocate

# probability of "right choice" r2, conditioned on observed R1
d$p_r2_discrete  <- NA
d$p_r2_bayesian <- NA
d$p_r2_optimal <- NA
d$p_r2_fixed <- NA
d$p_r2_naive <- NA

# probability of correct r2, conditioned on observed R1
d$p_c2_discrete  <- NA
d$p_c2_bayesian <- NA
d$p_c2_optimal <- NA
d$p_c2_fixed <- NA
d$p_c2_naive <- NA

# ------------------------------------------------------#
# compute trial-by-trial predictions

d$abs1 <- abs(d$s1)
d$abs2 <- abs(d$s2)

# unify level sets of factors
d$id <- factor(d$id)
dfit$id <- factor(dfit$id)

for(s in unique(d$id)){
  
  # ----------------------------------------------------- #
  # load parameters
  mnoise <- dfit$m[dfit$model=="metanoise" & dfit$id==s]
  sub2_1 <- dfit$w1[dfit$model=="sub2" & dfit$id==s]
  sub2_2 <- dfit$theta2[dfit$model=="sub2" & dfit$id==s]
  fixBias <- dfit$fixed_shift[dfit$model=="sub1" & dfit$id==s]
  
  # index trials of subjects with parameters selected above
  dsind <- which(d$id==s)
  
  # ----------------------------------------------------- #
  ### unconditioned probability of "right choice"
  
  d[dsind,]$p_r2_discrete <- p_r2_sub2(data.frame(s1=d[dsind,]$abs1,r1=d[dsind,]$acc1, s2=ifelse(d[dsind,]$acc1==1,d[dsind,]$abs2,-d[dsind,]$abs2)), c(sub2_1, sub2_2))
  d[dsind,]$p_r2_bayesian <- predict_r2_metanoise(data.frame(s1=d[dsind,]$abs1,r1=d[dsind,]$acc1, s2=ifelse(d[dsind,]$acc1==1,d[dsind,]$abs2,-d[dsind,]$abs2)), mnoise)
  d[dsind,]$p_r2_fixed <- pnorm(ifelse(d[dsind,]$acc1==1,d[dsind,]$abs2,-d[dsind,]$abs2), mean=fixBias, sd=1)
  d[dsind,]$p_r2_naive <- pnorm(ifelse(d[dsind,]$acc1==1,d[dsind,]$abs2,-d[dsind,]$abs2), mean=0, sd=1)
  d[dsind,]$p_r2_optimal <- predict_r2_metanoise(data.frame(s1=d[dsind,]$abs1,r1=d[dsind,]$acc1, s2=ifelse(d[dsind,]$acc1==1,d[dsind,]$abs2,-d[dsind,]$abs2)), 1)
  
  # ----------------------------------------------------- #
  # unconditioned probability of correct r2
  
  #
  d[dsind,]$p_c2_fixed <- ifelse(d[dsind,]$acc1==1, d[dsind,]$p_r2_fixed, 1-d[dsind,]$p_r2_fixed)
  d[dsind,]$p_c2_naive <- ifelse(d[dsind,]$acc1==1, d[dsind,]$p_r2_naive, 1-d[dsind,]$p_r2_naive)
  d[dsind,]$p_c2_optimal <- ifelse(d[dsind,]$acc1==1, d[dsind,]$p_r2_optimal, 1-d[dsind,]$p_r2_optimal)
  d[dsind,]$p_c2_discrete  <- ifelse(d[dsind,]$acc1==1, d[dsind,]$p_r2_discrete, 1-d[dsind,]$p_r2_discrete)
  d[dsind,]$p_c2_bayesian <- ifelse(d[dsind,]$acc1==1, d[dsind,]$p_r2_bayesian , 1-d[dsind,]$p_r2_bayesian)
  
}

# ------------------------------------------------------#
# store results
write.table(d, file="D:/xiangmu/data/behavior_for_behavior/data_wide_wmPred.txt",quote=F,sep=";",row.names=F)#D:/reysy/EEG/data/data_wide_wmPred.txt




