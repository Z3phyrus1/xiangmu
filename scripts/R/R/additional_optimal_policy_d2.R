# ------------------------------------------------------#
# some additional functions, can be used to compute optimal policies for discrete models
# Matteo Lisi, 2019
# ------------------------------------------------------#


# unconditional probability of correct response in models discrete-2
p_correct_s1s2 <- function(par, s1, s2){
  p_cor_d1 <- pnorm(abs(s1))
  p_c1 <- p_r2_sub2(data.frame(s1=abs(s1), r1=1, s2=abs(s2)), par)
  p_c0 <- 1-p_r2_sub2(data.frame(s1=abs(s1), r1=0, s2=-abs(s2)), par)
  p_c <- p_cor_d1*p_c1 + (1-p_cor_d1)*p_c0
  return(p_c)
}

# integrate over stimuli in range R
p_correct_R <- function(par, R){
  # require(cubature)
  integrand <- function(x) p_correct_s1s2(par,x[1],x[1])
  p_c <- cubature::adaptIntegrate(integrand, lowerLimit = c(0, 0), upperLimit = c(R, R))$integral
  return(p_c)
}

optimal_policy_d2 <- function(R){
  FUN <- function(par) -log(p_correct_R(par,R))
  par0 <- c(1, -1)
  opti <- optimx::optimx(par=par0, FUN, method=c("bobyqa"), lower=c(0,-2*R), upper=c(R,0), control=list(maxfun=100000000))
  par <- unlist(matrix(opti[1,1:2],1,2))
  return(par)
}

# unconditional probability of correct response in models discrete-3
p_correct_s1s2_sub3 <- function(par, s1, s2){
  p_cor_d1 <- pnorm(abs(s1))
  p_c1 <- p_r2_sub3(data.frame(s1=abs(s1), r1=1, s2=abs(s2)), par)
  p_c0 <- 1-p_r2_sub3(data.frame(s1=abs(s1), r1=0, s2=-abs(s2)), par)
  p_c <- p_cor_d1*p_c1 + (1-p_cor_d1)*p_c0
  return(p_c)
}

# integrate over stimuli in range R
p_correct_R_sub3 <- function(par, R){
  # require(cubature)
  integrand <- function(x) p_correct_s1s2_sub3(par,x[1],x[1])
  p_c <- cubature::adaptIntegrate(integrand, lowerLimit = c(0, 0), upperLimit = c(R, R))$integral
  return(p_c)
}

optimal_policy_d2_sub3 <- function(R){
  FUN <- function(par) -log(p_correct_R_sub3(par,R))
  par0 <- c(0.5, -0.5, 0.5, -0.5)
  opti <- optimx::optimx(par=par0, FUN, method=c("bobyqa"), lower=c(0,-2*R,0,-2*R), upper=c(R,0,R,0), control=list(maxfun=100000000))
  par <- unlist(matrix(opti[1,1:4],1,4))
  return(par)
}


# optimal policy for non-probabilistic observer
p_correct_s1s2_fixd <- function(par, s1, s2){
  p_cor_d1 <- pnorm(abs(s1))
  p_c1 <- pnorm(abs(s2), mean=par)
  p_c0 <- 1-pnorm(-abs(s2), mean=par)
  p_c <- p_cor_d1*p_c1 + (1-p_cor_d1)*p_c0
  return(p_c)
}

# integrate over stimuli in range R
p_correct_R_fixd <- function(par, R){
  # require(cubature)
  integrand <- function(x) p_correct_s1s2_fixd(par,x[1],x[1])
  p_c <- cubature::adaptIntegrate(integrand, lowerLimit = c(0, 0), upperLimit = c(R, R))$integral
  return(p_c)
}

optimal_policy_fixd <- function(R){
  FUN <- function(par) -log(p_correct_R_fixd(par,R))
  opti <- optim(par=-1, FUN, method=c("Brent"), lower=-2*R, upper=0)
  return(opti$par)
}

