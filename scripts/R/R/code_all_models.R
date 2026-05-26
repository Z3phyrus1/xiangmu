#-------------------------------------------------------------------------------------#
#
# Modelling functions for dual-decision task
# 
# Matteo Lisi, updated 2020
#
#-------------------------------------------------------------------------------------#
# 
# Notes:
# - hereafter 'eta' usually indicate the standard deviation of internal noise
#
#
#-------------------------------------------------------------------------------------#
# general

# error function (required by the other functions below)
# erf <- function(x)
# {
# 	# from: numerical recipes in C, 2nd ed
# 	t <- 1/(1+0.5*abs(x))
# 	tau <- t*exp(-x^2 -1.26551223 + 1.00002368*t + 0.37409196*t^2 + 0.09678418*t^3 - 0.18628806*t^4 + 0.27886807*t^5 - 1.13520398*t^6 + 1.48851587*t^7 - 0.82215223*t^8 + 0.17087277*t^9)
# 	return(ifelse(x>=0, 1-tau, tau -1))
# }

# alternative formulation in R
erf <- function(x) 2 * pnorm(x * sqrt(2)) - 1
erf.inv <- function(x) qnorm((x + 1)/2)/sqrt(2)

# truncated Gaussian
library(truncnorm)
# these are not necessary anymore, we use package truncnorm
# dtruncnorm <- function(x, mu=0, sigma=1, a=-Inf,b=Inf){
#   epsilon <- (x - mu)/sigma
#   alpha <- (a - mu)/sigma
#   beta <- (b - mu)/sigma
#   Z <- pnorm(beta) - pnorm(alpha)
#   return(ifelse(x>=a & x<b, dnorm(epsilon)/(sigma*Z), 0))
# }
# ptruncnorm <- function(x,mu=0,sigma=1, a=-Inf, b=Inf){
#   epsilon <- (x - mu)/sigma
#   alpha <- (a - mu)/sigma
#   beta <- (b - mu)/sigma
#   Z <- pnorm(beta) - pnorm(alpha)
#   cP <- (pnorm(epsilon) -pnorm(alpha)) / Z 
#   #return(cP)
#   return(ifelse(x<=a, 0, ifelse(x>b, 1, cP))) 
# }


#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
# Functions for likelihood computations and model estimation
# 
# Note: functions ending with '_nf' (short for 'noise_fixed') should be used when
# the noise is fixed to sigma = 1
#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#

# negative log-likelihood for simple 1 fit
# i.e. this would be the likelihood of the data under a "naive" model 
# in which the observer do not make use of the task structure at all
l_simple <- function (eta, d) 
{
  -sum(pnorm(d$s1[which(d$r1==1)]/eta, lower.tail = TRUE, log.p = TRUE)) - sum(pnorm(d$s1[which(d$r1==0)]/eta, lower.tail = FALSE, log.p = TRUE))
}

#-------------------------------------------------------------------------------------#
### ideal observer model

### optimized likelihood function (for response 2); use table lookup instead of numerical integration

# load lookup tables
if(file.exists("tabindex.rds") & file.exists("tabplus.rds") & file.exists("tabminus.rds")){
  t_plus <- readRDS(file="tabplus.rds")
  t_minus <- readRDS(file="tabminus.rds")
  t_index <- readRDS(file="tabindex.rds")
}else if(file.exists("./R/tabindex.rds") & file.exists("./R/tabplus.rds") & file.exists("./R/tabminus.rds")){
  t_plus <- readRDS(file="./R/tabplus.rds")
  t_minus <- readRDS(file="./R/tabminus.rds")
  t_index <- readRDS(file="./R/tabindex.rds")
}else{
  warning("Lookup table not available: run the script 'make_lookup_table.R' before loading the functions.")
}

intF <- function(x,y,tab,ind) # x = s1; y = s2
{
  ti <- findInterval(c(x,y), ind)
  x1 <- ind[ti[1]]; x2 <- ind[ti[1]+1]
  y1 <- ind[ti[2]]; y2 <- ind[ti[2]+1]
  (1/((x2-x1)*(y2-y1))) * (tab[ti[1],ti[2]]*(x2-x)*(y2-y) + tab[ti[1]+1,ti[2]]*(x-x1)*(y2-y) + tab[ti[1],ti[2]+1]*(x2-x)*(y-y1) + tab[ti[1]+1,ti[2]+1]*(x-x1)*(y-y1))
}

Fplus <- function(x,s1,s2) erf(x + s2) * exp(-(x-s1)^2)
Fminus <- function(x,s1,s2) erf(x - s2) * exp(-(x-s1)^2)

# likelihood of the second response, given the stimuli and the first response 
# (slower, not optimized, perform numerical integration of the likelihood for each trial)
# p_r2 <- function(obs, eta)
# {
# THETA <- function(x) ifelse(x<0,1,0)
# p_d2 <- function(x,s,eta) 0.5*(1 - erf((x-s)/(eta*sqrt(2))) )

# p_theta <- function(x, obs, eta){
# if(obs$r1==1){
# (THETA(x)/pnorm(obs$s1/eta))    * (1/sqrt(2*pi*eta^2)) * exp( -0.5 * ((-x-obs$s1)/eta)^2)
# }else if(obs$r1==0){
# (THETA(x)/(1-pnorm(obs$s1/eta))) * (1/sqrt(2*pi*eta^2)) * exp( -0.5 * ((x-obs$s1)/eta)^2)
# }
# }

# if(obs$r2==1){
# integrand <- function(x) p_theta(x, obs, eta) * p_d2(x,obs$s2,eta)
# }else if(obs$r2==0){
# integrand <- function(x) p_theta(x, obs, eta) * (1-p_d2(x,obs$s2,eta))
# }
# return(integrate(integrand, lower=-Inf, upper= Inf)$value)
# }
# l_bayes <- function (eta, d) # OLD
# {
# L1 <- -sum(pnorm(d$s1[which(d$r1==1)]/eta, lower.tail = TRUE, log.p = TRUE)) - sum(pnorm(d$s1[which(d$r1==0)]/eta, lower.tail = FALSE, log.p = TRUE))
# L2 <- vector(length=nrow(d),mode="numeric")
# for(i in 1:nrow(d)){
# L2[i] <- p_r2(d[i,], eta)
# }
# L <- L1 - sum(log(L2))
# return(L)
# }

### these are optimized functions that use table lookup & linear interpolation to solve the integral
p_r2 <- function(obs, eta, tab_plus=t_plus, tab_minus=t_minus, tab_index=t_index)
{
  s1_t <- obs$s1/(eta*sqrt(2))
  s2_t <- obs$s2/(eta*sqrt(2))
  if(all(findInterval(c(s1_t,s2_t),range(tab_index))==c(1,1))){
    if(obs$r1==1)
    {
      P <- 0.5 + (1/(2*pnorm(obs$s1/eta))) * intF(s1_t , s2_t, tab_plus, tab_index)
    }else{
      P <- 0.5 - (1/(2*(1-pnorm(obs$s1/eta)))) * intF(s1_t , s2_t, tab_minus, tab_index)
    }
  }else{
    if(obs$r1==1)
    {
      integrand <- function(x) Fplus(x, s1_t, s2_t)
      P <- 0.5 + (1/(2*pnorm(obs$s1/eta)))	 * (1/sqrt(pi)) * integrate(integrand, lower= 0, upper= Inf)$value
    }else{
      integrand <- function(x) Fminus(x, s1_t, s2_t)
      P <- 0.5 - (1/(2*(1-pnorm(obs$s1/eta)))) * (1/sqrt(pi)) * integrate(integrand, lower= -Inf, upper= 0)$value
    }
  }
  P <- ifelse(obs$r2==1,P,1-P)
  return(P)
}

# likelihood computation for optimal model
l_bayes <- function (eta, d, tab_plus=t_plus, tab_minus=t_minus, tab_index=t_index) 
{
  # uncommenting this will result in the likelihood being calculated from both decisions
  #L1 <- -sum(pnorm(d$s1[which(d$r1==1)]/eta, lower.tail = TRUE, log.p = TRUE)) - sum(pnorm(d$s1[which(d$r1==0)]/eta, lower.tail = FALSE, log.p = TRUE))
  L2 <- vector(length=nrow(d),mode="numeric")
  for(i in 1:nrow(d)){
    L2[i] <- p_r2(d[i,], eta, tab_plus, tab_minus, tab_index)
  }
  #L <- L1 - sum(log(L2))
  L <- - sum(log(L2))
  return(L)
}

# predicted probability of 2nd choice = 'right'
p_r2_bayes <- function (d, eta=1, tab_plus=t_plus, tab_minus=t_minus, tab_index=t_index) 
{
  d$r2 <- 1
  P2 <- vector(length=nrow(d),mode="numeric")
  for(i in 1:nrow(d)){
    P2[i] <- p_r2(d[i,], eta, tab_plus, tab_minus, tab_index)
  }
  return(P2)
}

# probability of second correct decision
p_acc2_bayes <- function (d, eta=1, tab_plus=t_plus, tab_minus=t_minus, tab_index=t_index) 
{
  d$r2 <- ifelse(d$s2>0,1,0)
  P2 <- vector(length=nrow(d),mode="numeric")
  for(i in 1:nrow(d)){
    P2[i] <- p_r2(d[i,], eta, tab_plus, tab_minus, tab_index)
  }
  return(P2)
}

# nf: noise-fixed (eta=1) 
l_bayes_nf <- function(d, tab_plus=t_plus, tab_minus=t_minus, tab_index=t_index) 
{
  return(l_bayes(1, d, tab_plus=t_plus, tab_minus=t_minus, tab_index=t_index))
}

# this gives the likelihood of a single trial
# (output as probability and not log-probability)
l_bayes_singleTrial <- function (eta, d) 
{
  L <- vector(length=nrow(d),mode="numeric")
  for(i in 1:nrow(d)){
    if(d$r1[i]==1){
      L[i] <- p_r2(d[i,], eta) * pnorm(d$s1[i]/eta, lower.tail = TRUE)
    }else{
      L[i] <- p_r2(d[i,], eta) * pnorm(d$s1[i]/eta, lower.tail = FALSE)
    }
  }
  return(L)
}

# simulate ideal observer model
sim_bayes <- function(obs, eta, rep = 1)
{
  obs <- obs[rep(seq_len(nrow(obs)), rep), ]
  obs$rep <- sort(rep_len(1:rep, nrow(obs)))
  ir1 <- obs$s1 + rnorm(length(obs$s1),0,eta) # internal response r1
  obs$r1 <- (sign(ir1)+1)/2 # ifelse(ir1>0,1,0) # decision 1
  acc1 <- sign(ir1) * sign(obs$s1)
  obs$s2 <- acc1 * abs(obs$s2)
  theta2 <- -abs(ir1)
  ir2 <- obs$s2 + rnorm(length(obs$s2),0,eta)
  obs$r2 <- (sign(ir2-theta2)+1)/2
  obs$rep <- factor(obs$rep)
  return(obs)
}

sim_bayes_nf <- function(obs, rep = 1)
{
  return(sim_bayes(obs, 1, rep))
}

#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
# Bayesian model with metacognitive "noise"
# 
# here the observer use Bayesian computations
# but mise-estimate the variability of her/his internal noise

# this assume eta == 1 (noise-fix analysis)
p_r2_metanoise <- function(obs, m)
{
  if(obs$r1==1)
  {
    integrand <- function(x) erf((x/m + obs$s2)/sqrt(2)) * exp(-0.5*((x-obs$s1))^2)
    P <- 0.5 + (1/(2*pnorm(obs$s1)))	 * (1/(sqrt(2*pi))) * integrate(integrand, lower= 0, upper= Inf)$value
  }else{
    integrand <- function(x) erf((obs$s2 - x/m)/sqrt(2)) * exp(-0.5*((x-obs$s1))^2)
    P <- 0.5 + (1/(2*pnorm(obs$s1,lower.tail=FALSE))) * (1/(sqrt(2*pi))) * integrate(integrand, lower= -Inf, upper= 0)$value
  }
  P <- ifelse(obs$r2==1,P,1-P)
  return(P)
}


l_bayes_metanoise <- function (m, d) 
{
  #L1 <- -sum(pnorm(d$s1[which(d$r1==1)], lower.tail = TRUE, log.p = TRUE)) - sum(pnorm(d$s1[which(d$r1==0)], lower.tail = FALSE, log.p = TRUE))
  L2 <- vector(length=nrow(d),mode="numeric")
  for(i in 1:nrow(d)){
    L2[i] <- p_r2_metanoise(d[i,], m)
  }
  #L <- L1 - sum(log(L2))
  L <- - sum(log(L2))
  return(L)
}

# predicted probability of 2nd choice = 'right'
predict_r2_metanoise <- function (d, m) 
{
  d$r2 <- 1
  P2 <- vector(length=nrow(d),mode="numeric")
  for(i in 1:nrow(d)){
    P2[i] <- p_r2_metanoise(d[i,], m)
  }
  return(P2)
}

predict_acc2_metanoise <- function (d, m) 
{
  d$r2 <-  ifelse(d$s2>0,1,0)
  P2 <- vector(length=nrow(d),mode="numeric")
  for(i in 1:nrow(d)){
    P2[i] <- p_r2_metanoise(d[i,], m)
  }
  return(P2)
}

sim_metanoise <- function(obs, eta, m, rep = 1)
{
  obs <- obs[rep(seq_len(nrow(obs)), rep), ]
  obs$rep <- sort(rep_len(1:rep, nrow(obs)))
  ir1 <- obs$s1 + rnorm(length(obs$s1),0,eta) # internal response r1
  obs$r1 <- (sign(ir1)+1)/2 # ifelse(ir1>0,1,0) # decision 1
  acc1 <- sign(ir1) * sign(obs$s1)
  obs$s2 <- acc1 * abs(obs$s2)
  theta2 <- -abs(ir1)/m
  ir2 <- obs$s2 + rnorm(length(obs$s2),0,eta)
  obs$r2 <- (sign(ir2-theta2)+1)/2
  obs$rep <- factor(obs$rep)
  return(obs)
}


#-------------------------------------------------------------------------------------#

### suboptimal model # 1 criterion # likelihood function
# par [1] noise
# par [2] criterion shift
l_sub1 <- function (par, d) 
{
  L2 <- -sum(pnorm((d$s2[which(d$r2==1)]-par[2])/par[1], lower.tail = TRUE, log.p = TRUE)) - sum(pnorm((d$s2[which(d$r2==0)]-par[2])/par[1], lower.tail = FALSE, log.p = TRUE))
  return(L2)
}

### noise-fixed version
# only one parameter; used to find criterion shift when stimuli 
# are already transformed in units of noise
l_sub1_nf <- function (par, d) 
{
  par <- c(1,par)
  return(l_sub1(par, d))
}

p_r2_sub1 <- function(obs, par)
{
  if(length(par)<2){
    par <- c(1, unlist(par))
  }
  return(pnorm((obs$s2-par[2])/par[1]))
}

p_acc2_sub1 <- function(obs, par)
{
  if(length(par)<2){
    par <- c(1, unlist(par))
  }
  return(ifelse(obs$s2>0,pnorm((obs$s2-par[2])/par[1]),1-pnorm((obs$s2-par[2])/par[1])))
}

# suboptimal model # 1 criterion # predict both responses
# to simulate with noise fixed just add 1 at the beginning of
# parameters vector
sim_sub1 <- function(obs, par, rep = 1)
{
  obs <- obs[rep(seq_len(nrow(obs)), rep), ]
  obs$rep <- sort(rep_len(1:rep, nrow(obs)))
  ir1 <- obs$s1 + rnorm(length(obs$s1),0,par[1]) # internal response r1
  obs$r1 <- (sign(ir1)+1)/2 # ifelse(ir1>0,1,0) # decision 1
  acc1 <- sign(ir1) * sign(obs$s1)
  obs$s2 <- acc1 * abs(obs$s2)
  ir2 <- obs$s2 + rnorm(length(obs$s2),0,par[1])
  obs$r2 <- (sign(ir2-par[2])+1)/2
  obs$rep <- factor(obs$rep)
  return(obs)
}

sim_sub1_nf <- function(obs, par, rep = 1)
{
  return(sim_sub1(obs, c(1,par), rep))
}


#-------------------------------------------------------------------------------------#

### suboptimal model # 2 criterion/levels of confidence

# par[1] noise standard deviation
# par[2] threshold for confidence 
# par[3] criterion adjustment

psub2_r2 <- function(obs, par) 
{
  tr1 <- obs$r1*2 -1
  p_c <- (1/(2*(1-obs$r1 + tr1*pnorm(obs$s1/par[1])))) * (1- erf((par[2]-tr1*obs$s1)/(sqrt(2)*par[1])))
  tr2 <- obs$r2*2 -1
  P<-((p_c*(1-obs$r2 + tr2*pnorm((obs$s2-par[3])/par[1])) + (1-p_c)*(1-obs$r2 + tr2*pnorm(obs$s2/par[1]))))
  
}

l_sub2 <- function (par, d) 
{
  L <- -sum(log(psub2_r2(d, par)))
  return(L)
}

l_sub2_nf <- function (par, d) 
{
  return(l_sub2(c(1,par), d))
}

p_r2_sub2 <- function(obs, par)
{
  if(length(par)<3){
    par <- c(1, unlist(par))
  }
  obs$r2 <- 1
  return(psub2_r2(obs, par))
}

p_acc2_sub2 <- function(obs, par)
{
  if(length(par)<3){
    par <- c(1, unlist(par))
  }
  obs$r2 <- ifelse(obs$s2>0,1,0)
  return(psub2_r2(obs, par))
}


# simulate discrete model with 2 lvl
sim_sub2 <- function(obs, par, rep = 1)
{
  obs <- obs[rep(seq_len(nrow(obs)), rep), ]
  obs$rep <- sort(rep_len(1:rep, nrow(obs)))
  ir1 <- obs$s1 + rnorm(length(obs$s1),0,par[1]) # internal response r1
  obs$r1 <- (sign(ir1)+1)/2 # ifelse(ir1>0,1,0) # decision 1
  acc1 <- sign(ir1) * sign(obs$s1)
  obs$s2 <- acc1 * abs(obs$s2)
  
  # set criterion according to confidence level
  theta2 <- ((sign(abs(ir1) - abs(par[2]))+1)/2)*par[3] # ifelse(abs(ir1)>par[2],par[3],0)
  
  ir2 <- obs$s2 + rnorm(length(obs$s2),0,par[1])
  obs$r2 <- (sign(ir2-theta2)+1)/2
  obs$rep <- factor(obs$rep)
  return(obs)
}
sim_sub2_nf <- function(obs, par, rep = 1)
{
  return(sim_sub2(obs, c(1,par), rep))
}


#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#

### suboptimal model # 3 levels of confidence

# par[1] noise standard deviation
# par[2] threshold for confidence (1)
# par[3] criterion adjustment (if confidence = 1)
# par[4] threshold for confidence (2) [additive difference from conf th 1]
# par[5] criterion adjustment (if confidence = 2) [additive]

psub3_r2 <- function(obs, par) 
{
  c2 <- par[2]+par[4]
  th2 <- par[3]+par[5]
  c1 <- par[2]
  th1 <- par[3]
  
  # virtually "invert" stimulus such that (+) correspond to the side choosen
  # s1_ <- ifelse(obs$r1==1, obs$s1, -obs$s1) 
  s1_ <- ((obs$r1*2)-1)*obs$s1
  
  require(truncnorm)
  ptruncnorm <- truncnorm::ptruncnorm
  dtruncnorm <- truncnorm::dtruncnorm
  
  p_c2 <- 1 - ptruncnorm(c2, mean=s1_,sd=par[1], a=0)
  p_c1 <-  1 - ptruncnorm(c1, mean=s1_,sd=par[1], a=0) - p_c2
  p_c0 <-  1 - (p_c1 + p_c2)
  
  P_ <- p_c0*pnorm(0,mean=obs$s2,sd=par[1],lower.tail=F) + p_c1*pnorm(th1,mean=obs$s2,sd=par[1],lower.tail=F) + p_c2*pnorm(th2,mean=obs$s2,sd=par[1],lower.tail=F)
  return(ifelse(obs$r2==1,P_, 1-P_))
}

l_sub3 <- function (par, d) 
{
  L <- - sum(log(psub3_r2(d, par)))
  return(L)
}
l_sub3_nf <- function (par, d) 
{
  return(l_sub3(c(1,par), d))
}

p_r2_sub3 <- function(obs, par)
{
  if(length(par)<5){
    par <- unname(c(1, unlist(par)))
  }
  obs$r2 <- 1
  return(psub3_r2(obs, par))
}

p_acc2_sub3 <- function(obs, par)
{
  if(length(par)<5){
    par <- unname(c(1, unlist(par)))
  }
  obs$r2 <- ifelse(obs$s2>0,1,0)
  return(psub3_r2(obs, par))
}

sim_sub3 <- function(obs, par, rep = 1)
{
  c2 <- par[2]+par[4]
  th2 <- par[3]+par[5]
  
  obs <- obs[rep(seq_len(nrow(obs)), rep), ]
  obs$rep <- sort(rep_len(1:rep, nrow(obs)))
  ir1 <- obs$s1 + rnorm(length(obs$s1),0,par[1]) # internal response r1
  obs$r1 <- (sign(ir1)+1)/2 # ifelse(ir1>0,1,0) # decision 1
  acc1 <- sign(ir1) * sign(obs$s1)
  obs$s2 <- acc1 * abs(obs$s2)
  
  # set criterion according to confidence level
  theta2 <- ifelse(abs(ir1)>c2,th2,ifelse(abs(ir1)>par[2],par[3],0))
  
  ir2 <- obs$s2 + rnorm(length(obs$s2),0,par[1])
  obs$r2 <- (sign(ir2-theta2)+1)/2
  obs$rep <- factor(obs$rep)
  return(obs)
}
sim_sub3_nf <- function(obs, par, rep = 1)
{
  return(sim_sub3(obs, c(1,par), rep))
}



#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
# create a wrapper function that allow to fit all suboptimal models using 
# bound contrained optimization (bobyqa method)

fit_sub_model<-function(d, nlevels=2, hess=F, maxb=15){
  ub <- c(maxb, 0, maxb, 0, maxb, 0)
  lb <- c(0, -maxb, 0, -maxb, 0, -maxb)
  switch(nlevels,
         {n_par<-1
         stop("nlevels=1 not implemented. Use directly Brent method in optim() to estimate fixed-bias model.")
         },
         {n_par<-2
         par<-c(1, -1)
         Lfun<-function(par,d){l_sub2_nf(par,d)}
         },
         {n_par<-4
         par<-c(0.5, -0.5, 0.5, -0.5)
         Lfun<-function(par,d){l_sub3_nf(par,d)}
         },
         {n_par<-6
         par<-c(0.2, -0.2, 0.3, -0.4, 0.5, -0.5)
         Lfun<-function(par,d){l_sub4_nf(par,d)}
         })
  
  fit <- optimx::optimx(par=par, Lfun, d=d, 
                        method=c("bobyqa"), lower=lb[1:n_par], upper=ub[1:n_par], 
                        control=list(maxfun=100000000))
  out_ <- {}
  out_$par <- unlist(matrix(fit[1,1:n_par],1,n_par))
  out_$value <- fit$value
  
  if(hess){
    out_$hessian <- numDeriv::hessian(Lfun, x=out_$par, d=d)
  }
  
  return(out_)
}


