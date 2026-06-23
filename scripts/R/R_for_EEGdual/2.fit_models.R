# ------------------------------------------------------------------ #
# This script does the following:
# 1) fit models in individual data
# 2) save parameters and goodness-of-fit indicators
# Matteo Lisi, 2019
# ------------------------------------------------------------------ #

rm(list=ls())
setwd("D:/xiangmu/scripts/R")
source("./R/code_all_models.R")
source("./R/additional_optimal_policy_d2.R")

# load everything
d <- read.table("D:/xiangmu/data/behavior_for_behavior/data_nu.txt",header=T, sep=";")#D:/xiangmu/data/behavior/id/data_nu.txt
str(d)

## exclude control sessions
d <- d[d$dual==1,]

# convert decision order to factor
d$decision <- as.character(d$decision)


# wrapper function that fit models given d_s
fit_models <- function(d){
  d_s <- data.frame(s1=d$s[d$decision=="1"], s2=d$s[d$decision=="2"], r1=d$rr[d$decision=="1"], r2=d$rr[d$decision=="2"])
  fctrl <- list(maxit=100000000)
  logLikOpti <- l_bayes_nf(d_s) # likelihood Bayesian model
  sub1 <- optim(par = -.5, l_sub1_nf, d=d_s, hessian=T, method="Brent", lower=-10, upper=0, control= fctrl)
  sub2 <- fit_sub_model(d_s, nlevels=2,hess=T,maxb=10)
  
  meta_fit <- optimx::optimx(par=1, l_bayes_metanoise, d=d_s, method=c("bobyqa"), lower=0.1, upper=10, control=list(maxfun=10000000))
  
  ## if you wanted to calculate likelihood of optimal policy for sub2 model
  # sub2_opti <- optimal_policy_d2(2*max(abs(c(d_s$s1,d_s$s1))))
  # L_sub2_opti <- l_sub2_nf(sub2_opti, d_s)
  
  model <- c("bayes","sub1","sub2","metanoise")#,"sub2-optimal")
  logLik <- -c(logLikOpti, sub1$value, sub2$value, meta_fit$value)#, L_sub2_opti)
  n_par <- c(0, 1, 2, 1)#, 0, 0)
  AIC <- 2*n_par -2*logLik
  relModLik <- exp(-(AIC - min(AIC))/2)
  w <- relModLik / sum(relModLik)
  
  fixed_shift <- c(NA, sub1$par, NA,  NA)#, NA)
  w1 <- c(NA, NA, sub2$par[1],  NA)#,  sub2_opti[1])
  theta2  <- c(NA, NA, sub2$par[2],  NA)#,  sub2_opti[2])
  m <-  c(NA, NA, NA, unlist(meta_fit[1,1]))#, NA)
  
  d_fit <- data.frame(model, logLik, n_par, AIC, relModLik, fixed_shift, w1, theta2, m, w)
  return(d_fit)
}

# ## This version of the wrapper function also fit the discrete-3 model
# # wrapper function that fit models given d_s
# fit_models_s3 <- function(d){
#   d_s <- data.frame(s1=d$s[d$decision=="1"], s2=d$s[d$decision=="2"], r1=d$rr[d$decision=="1"], r2=d$rr[d$decision=="2"])
#   fctrl <- list(maxit=100000000)
#   logLikOpti <- l_bayes_nf(d_s) # likelihood Bayesian model
#   sub1 <- optim(par = -.5, l_sub1_nf, d=d_s, hessian=T, method="Brent", lower=-10, upper=0, control= fctrl)
#   sub2 <- fit_sub_model(d_s, nlevels=2,hess=T,maxb=10)
#   sub3 <- fit_sub_model(d_s, nlevels=3,hess=T,maxb=10)
#   
#   meta_fit <- optimx::optimx(par=1, l_bayes_metanoise, d=d_s, method=c("bobyqa"), lower=0.1, upper=10, control=list(maxfun=10000000))
#   
#   model <- c("bayes","sub1","sub2","sub3","metanoise")#,"sub2-optimal","sub3-optimal")
#   logLik <- -c(logLikOpti, sub1$value, sub2$value, sub3$value, meta_fit$value)#, L_sub2_opti, L_sub3_opti)
#   n_par <- c(0, 1, 2, 4, 1)#, 0, 0)
#   AIC <- 2*n_par -2*logLik
#   relModLik <- exp(-(AIC - min(AIC))/2)
#   w <- relModLik / sum(relModLik)
#   
#   fixed_shift <- c(NA, sub1$par, NA, NA, NA)#, NA, NA)
#   w1 <- c(NA, NA, sub2$par[1], sub3$par[1], NA)#,  sub2_opti[1], sub3_opti[1])
#   theta2  <- c(NA, NA, sub2$par[2], sub3$par[2], NA)#,  sub2_opti[2], sub3_opti[2])
#   w2  <- c(NA, NA, NA, sub3$par[3], NA)#, NA, sub3_opti[3])
#   theta3 <- c(NA, NA, NA, sub3$par[4],  NA)#, NA, sub3_opti[4])
#   m <-  c(NA, NA, NA, NA, unlist(meta_fit[1,1]))#, NA, NA)
#   #m <-  c(NA, NA, NA, NA, meta_fitB$par)#, NA, NA)
#   
#   d_fit <- data.frame(model, logLik, n_par, AIC, relModLik, fixed_shift, w1, w2, theta2, theta3, m, w)
#   return(d_fit)
# }

# do fitting
d_fit <- NULL
for(i in unique(d$id)){
  
  d_id <- fit_models(d[d$id==i,])
  # d_id <- fit_models_s3(d[d$id==i,])
  d_id$id <- i
  
  # test overall difference in prop. correct d1 vs d2
  d_s <- d[d$id==i,]
  n_ <- nrow(d_s)/2
  macc1 <- mean(d_s$acc[d_s$decision==1])
  macc2 <- mean(d_s$acc[d_s$decision==2])
  crosstab_acc <- matrix(c(macc1*n_, macc2*n_, n_-macc1*n_, n_-macc2*n_),2,2,dimnames=list(c("d1","d2"),c(1,0)))
  bin_test <- prop.test(crosstab_acc, alternative = "less", correct=F)
  
  d_id$acc.stat <- bin_test$statistic
  d_id$acc.p.value <- bin_test$p.value
  
  d_fit <- rbind(d_fit, d_id)
  
  cat(i," done! moving on...\n\t")
}
cat("\ncompleted!")
write.table(d_fit, "D:/xiangmu/data/behavior_for_behavior/fit_par.txt",row.names=F)#D:/xiangmu/data/behavior/id/fit_par.txt

