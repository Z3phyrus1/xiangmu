# ------------------------------------------------------#
# functions for fitting psychometric functions and estimating 
# model-averaged parameters
# (The main function is 'estimate_noise()' at the bottom of the script)
#
# Matteo Lisi, 2020
# ------------------------------------------------------#

source("D:/xiangmu/scripts/R/R/code_all_models.R")

# functions for fitting models with lapses
psy_3par <- function(x, mu ,sigma, lambda){
  lambda + (1-2*lambda) * 0.5 * (1+erf((x-mu)/(sqrt(2)*sigma)))
}

lnorm_3par <- function(p,d,task){
  if(task=="orientation"){
    nL <- -sum(log(psy_3par(d$signed_mu[d$rr==1], p[1] ,p[2] ,p[3]))) - 
      sum(log(1 - psy_3par(d$signed_mu[d$rr==0], p[1] ,p[2], p[3])))
  }else{
    nL <- -sum(log(psy_3par(d$signed_coherence[d$rr==1], p[1] ,p[2] ,p[3]))) - 
      sum(log(1 - psy_3par(d$signed_coherence[d$rr==0], p[1] ,p[2], p[3])))
  }
  return(nL)
}
lnorm_nb_yl <- function(p,d,task){
  if(task=="orientation"){
    nL <- -sum(log(psy_3par(d$signed_mu[d$rr==1], 0 ,p[1] ,p[2]))) - 
      sum(log(1 - psy_3par(d$signed_mu[d$rr==0], 0 ,p[1], p[2])))
  }else{
    nL <- -sum(log(psy_3par(d$signed_coherence[d$rr==1], 0 ,p[1] ,p[2]))) - 
      sum(log(1 - psy_3par(d$signed_coherence[d$rr==0], 0 ,p[1], p[2])))
  }
  return(nL)
}

# wrapper functions (fit models and return parameters and AIC values)
# the functions are named according to:
# nb_nl: No bias - No lapse rate
# yb_nl: Yes bias - No lapse rate
# nb_yl: No bias - Yes lapse rate
# yb_yl: Yes bias - Yes lapse rate
fit_nb_nl <- function(d,task){
  if(task=="orientation"){
    m <- glm(rr~0+signed_mu,family=binomial(link=probit), d)
  }else{
    m <- glm(rr~0+signed_coherence,family=binomial(link=probit), d)
  }
  eta <- unname(1/coef(m))
  return(list(eta=eta, bias=0, lambda=0, aic=AIC(m)))
}

fit_yb_nl <- function(d,task){
  if(task=="orientation"){
    m <- glm(rr~signed_mu,family=binomial(link=probit), d)
  }else{
    m <- glm(rr~signed_coherence,family=binomial(link=probit), d)
  }
  eta <- unname(1/coef(m)[2])
  bias <- -unname(coef(m)[1] / coef(m)[2])
  return(list(eta=eta, bias=bias, lambda=0, aic=AIC(m)))
}

fit_nb_yl <- function(d,task){
  if(task=="orientation"){
    start_p <- c(mean(abs(d$signed_mu),na.rm=T), 0)
    LB <- min(abs(d$signed_mu))/2
    UB <- max(abs(d$signed_mu))
  }else{
    start_p <- c(mean(abs(d$signed_coherence),na.rm=T), 0)
    LB <- min(abs(d$signed_coherence))/2
    UB <- max(abs(d$signed_coherence))
  }
  lpb <- c(LB, 0)
  upb <- c(UB, 0.5)
  m <- optimx::optimx(par = start_p, lnorm_nb_yl , d=d,  task=task, method="bobyqa", lower =lpb, upper =upb)
  cat(m$ierr)
  eta <- unlist(unname(m[1]))
  lambda <- unlist(unname(m[2]))
  aic <- 2*2 + 2*unlist(unname(m[3]))
  return(list(eta=eta, bias=0, lambda=lambda, aic=aic))
}

fit_yb_yl <- function(d,task){
  if(task=="orientation"){
    start_p <- c(0, mean(abs(d$signed_mu)), 0)
    lpb <- c(-max(abs(d$signed_mu)), min(abs(d$signed_mu))/2, 0)
    upb <- c(max(abs(d$signed_mu)), max(abs(d$signed_mu)), 0.5)
  }else{
    start_p <- c(0, mean(abs(d$signed_coherence)), 0)
    lpb <- c(-max(abs(d$signed_coherence)), min(abs(d$signed_coherence))/2, 0)
    upb <- c(max(abs(d$signed_coherence)), max(abs(d$signed_coherence)), 0.5)
  }
  m <- optimx::optimx(par = start_p, lnorm_3par , d=d,  task=task, method="bobyqa", lower =lpb, upper =upb)
  cat(m$ierr)
  bias <- unlist(unname(m[1]))
  eta <- unlist(unname(m[2]))
  lambda <- unlist(unname(m[3]))
  aic <- 2*3 + 2*unlist(unname(m[4]))
  return(list(eta=eta, bias=bias, lambda=lambda, aic=aic))
}

# wrapper function that do the model averaging
estimateNoise <- function(d){
  
  nbnl_1 <- fit_nb_nl(d[d$task=="orientation",],task="orientation")
  ybnl_1 <- fit_yb_nl(d[d$task=="orientation",],task="orientation")
  nbyl_1 <- fit_nb_yl(d[d$task=="orientation",],task="orientation")
  ybyl_1 <- fit_yb_yl(d[d$task=="orientation",],task="orientation")
  deltas <- c(nbnl_1$aic, ybnl_1$aic, nbyl_1$aic, ybyl_1$aic) - min(c(nbnl_1$aic, ybnl_1$aic, nbyl_1$aic, ybyl_1$aic))
  weights <- exp(-0.5*deltas) / sum(exp(-0.5*deltas))
  eta_1 <- sum(weights * c(nbnl_1$eta, ybnl_1$eta, nbyl_1$eta, ybyl_1$eta))
  bias_1 <- sum(weights * c(nbnl_1$bias, ybnl_1$bias, nbyl_1$bias, ybyl_1$bias))
  lapse_1 <- sum(weights * c(nbnl_1$lambda, ybnl_1$lambda, nbyl_1$lambda, ybyl_1$lambda))
  
  nbnl_2 <- fit_nb_nl(d[d$task=="motion",],task="motion")
  ybnl_2 <- fit_yb_nl(d[d$task=="motion",],task="motion")
  nbyl_2 <- fit_nb_yl(d[d$task=="motion",],task="motion")
  ybyl_2 <- fit_yb_yl(d[d$task=="motion",],task="motion")
  deltas <- c(nbnl_2$aic, ybnl_2$aic, nbyl_2$aic, ybyl_2$aic) - min(c(nbnl_2$aic, ybnl_2$aic, nbyl_2$aic, ybyl_2$aic))
  weights <- exp(-0.5*deltas) / sum(exp(-0.5*deltas))
  eta_2 <- sum(weights * c(nbnl_2$eta, ybnl_2$eta, nbyl_2$eta, ybyl_2$eta))
  bias_2 <- sum(weights * c(nbnl_2$bias, ybnl_2$bias, nbyl_2$bias, ybyl_2$bias))
  lapse_2 <- sum(weights * c(nbnl_2$lambda, ybnl_2$lambda, nbyl_2$lambda, ybyl_2$lambda))
  
  outV <- c(eta_1, bias_1, lapse_1, eta_2, bias_2, lapse_2)
  names(outV) <- c("sigma_o", "bias_o", "lapse_o", "sigma_m", "bias_m", "lapse_m")
  return(outV)
}
