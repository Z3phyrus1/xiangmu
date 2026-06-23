# ------------------------------------------------------------------ #
# This script does the model-free (model-agnostic) analyses reported in the paper
# (e.g. logistic regression of prop. correct responses)
# Matteo Lisi, 2020
# ------------------------------------------------------------------ #

rm(list=ls())
setwd("D:/xiangmu/scripts/R")
source("./R/code_all_models.R")

# load data
d <- read.table("D:/xiangmu/data/behavior_for_behavior/data_nu.txt",header=T,sep=";")

# ------------------------------------------------------#
# some summary statistics
dag <- aggregate(acc ~ decision + id + dual, d, mean)
aggregate(acc ~ decision + dual, dag, mean)
mean(d$acc[d$decision==1])

# ------------------------------------------------------#
# as reported in the text, data collection was done in two different sites
# (UCL and City University) with similar equipment, yielding two datasets. 
# The information about which subject belong to which dataset is included in 
# the last two digits of the subject code
d$dataset <- substr(as.character(d$id), start=nchar(as.character(d$id))-1, stop=nchar(as.character(d$id)))
# In particular:
# - dataset collected at City: d$dataset== "00"
# - dataset collected at UCL : d$dataset== "01"

# ------------------------------------------------------#
# 'model free' test
library(lme4)
d$decision_f <- d$decision-1
d$condition_f <- ifelse(d$dual==0,0,1)
d$task_f <- ifelse(d$task=="orientation",0,1)
d$abs_s <- abs(d$s)

m0 <- lmList(acc ~ abs_s + decision_f*condition_f| id, d, family=binomial(logit))

# decision 2 vs 1 at control task
t.test(coef(m0)[,3])
round(exp(mean(coef(m0)[,3])),digits=2)
round(exp(mlisi::bootMeanCI(coef(m0)[,3], nsim=20000)),digits=2) # library mlisi available at https://github.com/mattelisi/mlisi

library(BayesFactor)
1/ttestBF(x = coef(m0)[,3], rscale=1)

# decision 2 vs 1 at dual decision task
t.test(coef(m0)[,5]+coef(m0)[,3])
round(mean(exp(coef(m0)[,5]+coef(m0)[,3])),digits=2)
round(exp(mlisi::bootMeanCI(coef(m0)[,5]+coef(m0)[,3], nsim=20000)),digits=2)
ttestBF(x = coef(m0)[,5]+coef(m0)[,3], rscale=1)

# there is also a main effect of condition, suggesting some learning
t.test(coef(m0)[,4])

# ------------------------------------------------------#
# now test for the effect of acc1
d <- read.table(file="D:/xiangmu/data/behavior_for_behavior/data_wide_wmPred.txt",header=T, sep=";")
str(d)

m0 <- lmList(r2 ~ s2 + acc1  | id, d, family=binomial(logit))
coef(m0)

t.test(coef(m0)[,3])
round(exp(mean(coef(m0)[,3])),digits=2)
round(exp(mlisi::bootMeanCI(coef(m0)[,3])),digits=2)

ttestBF(x = coef(m0)[,3], rscale=1)

# test the effect of abs1
m0 <- lmList(r2 ~ s2 + abs1 + abs1:acc1  | id, d, family=binomial(logit))
coef(m0)

# change in response probability after a wrong response
t.test(coef(m0)[,3])
round(exp(mean(coef(m0)[,3])),digits=2)
round(exp(mlisi::bootMeanCI(coef(m0)[,3])),digits=2)
ttestBF(x = coef(m0)[,3], rscale=1)

# change in response probability after a correct response
t.test(coef(m0)[,4]+coef(m0)[,3])
round(exp(mean(coef(m0)[,4]+coef(m0)[,3])),digits=2)
round(exp(mlisi::bootMeanCI(coef(m0)[,4]+coef(m0)[,3])),digits=2)
ttestBF(x = coef(m0)[,4]+coef(m0)[,3], rscale=1)




