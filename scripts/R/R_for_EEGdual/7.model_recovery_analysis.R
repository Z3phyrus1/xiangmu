# ------------------------------------------------------#
# model recovery analysis
# Matteo Lisi, 2018
# - now presented as Extended Data
# - paramaters now sampled from their multivariate distribution
# ------------------------------------------------------#

rm(list=ls())
setwd("D:/xiangmu/scripts/R")
source("./R/code_all_models.R")
library(MASS)

# ------------------------------------------------------#
# average fitted parameters
dfit <- read.table("D:/xiangmu/data/behavior_for_behavior/fit_par.txt",header=T)

dfit$best_m <- NA
for(i in unique(dfit$id)){
  dfit$best_m[dfit$id==i] <-as.character(dfit[dfit$id==i,]$model[which(dfit$AIC[dfit$id==i]==min(dfit$AIC[dfit$id==i]))])
}
tapply(dfit$best_m,dfit$best_m,length)


m_mean <- mean(dfit$m,na.rm=T)
m_sd <- sd(dfit$m,na.rm=T)

# sample from multivariate distribution for discrete model
MU_s2 <- apply(dfit[dfit$model=="sub2",c("w1","theta2")],2,mean,na.rm=T)
SD_s2 <- cov(dfit[dfit$model=="sub2" ,c("w1","theta2")],use="pairwise.complete.obs")

# ------------------------------------------------------#
# simulation parameters
n_id <- 25
n_trial <- 600
R <- 2.5

# settings
fctrl <- list(maxit=100000000)
maxb <- 15 # max bounbary for criteria of heuristic models (units of internal noise)
ub <- c(maxb, 0, maxb, 0, maxb, 0)
lb <- c(0, -maxb, 0, -maxb, 0, -maxb)
opt_method <- "bobyqa" # alternatively "L-BFGS-B"

# ------------------------------------------------------#
# run simulation
sim_data <- {}
sim_fit <- {}

n_par <- c(0 , 1, 2)

cat("Model recovery simulation started on ",date(),"\n")

for(id in 1:n_id){
  d <- data.frame(s1=runif(n_trial,min=-R,max=R), s2=runif(n_trial,min=-R,max=R))
  
  mnoise_i <- -1
  while(mnoise_i<=0){ mnoise_i <- rnorm(1, mean=m_mean,sd=m_sd)}
  
  sub2_1_i <- -1
  sub2_2_i <-  1
  while(sub2_1_i<=0 | sub2_2_i> 0){
    par_s2 <- mvrnorm(n=1, mu = MU_s2,Sigma=SD_s2)
    sub2_1_i <- par_s2[1]
    sub2_2_i <- par_s2[2]
  }
  
  # generative: Bayesian
  d1 <- sim_bayes(d, 1, rep = 1)
  logLikOpti <- l_bayes_nf(d1)
  meta_fit <- optimx::optimx(par=1, l_bayes_metanoise, d=d1, 
                             method=c("bobyqa"), lower=0.1, upper=15, 
                             control=list(maxfun=100000000))
  sub2 <- fit_sub_model(d1,nlevels=2,hess=F,maxb=maxb)
  logLik <- -c(logLikOpti, meta_fit$value, sub2$value)
  AIC <- 2*n_par -2*logLik
  d1_fit <- data.frame(generative="Bayesian",
                       fitted=c("Bayesian","biased-Bayesian","discrete"),
                       L = logLik, AIC = AIC,
                       relModLik = exp(-(AIC - min(AIC))/2),
                       AkaikeWeight = exp(-(AIC - min(AIC))/2) / sum(exp(-(AIC - min(AIC))/2)),
                       par1 = c(NA, meta_fit$p1, sub2$par[1]),
                       par2 = c(NA, NA, sub2$par[2]),
                       gpar1 = c(NA, mnoise_i , sub2_1_i),
                       gpar2 = c(NA, NA, sub2_2_i),id=id)
  
  # generative: biases-Bayesian
  d2 <- sim_metanoise(d, 1, mnoise_i, rep = 1)
  logLikOpti <- l_bayes_nf(d2)
  meta_fit <- optimx::optimx(par=1, l_bayes_metanoise, d=d2, 
                             method=c("bobyqa"), lower=0.1, upper=15, 
                             control=list(maxfun=100000000))
  sub2 <- fit_sub_model(d2,nlevels=2,hess=F,maxb=maxb)
  logLik <- -c(logLikOpti, meta_fit$value, sub2$value)
  AIC <- 2*n_par -2*logLik
  d2_fit <- data.frame(generative="biased-Bayesian",
                       fitted=c("Bayesian","biased-Bayesian","discrete"),
                       L = logLik, AIC = AIC,
                       relModLik = exp(-(AIC - min(AIC))/2),
                       AkaikeWeight = exp(-(AIC - min(AIC))/2) / sum(exp(-(AIC - min(AIC))/2)),
                       par1 = c(NA, meta_fit$p1, sub2$par[1]),
                       par2 = c(NA, NA, sub2$par[2]),
                       gpar1 = c(NA, mnoise_i , sub2_1_i),
                       gpar2 = c(NA, NA, sub2_2_i),id=id)
  
  
  # generative: discrete-2
  d3 <- sim_sub2(d, c(1, sub2_1_i, sub2_2_i), rep = 1)
  logLikOpti <- l_bayes_nf(d3)
  meta_fit <- optimx::optimx(par=1, l_bayes_metanoise, d=d3, 
                             method=c("bobyqa"), lower=0.1, upper=15, 
                             control=list(maxfun=100000000))
  sub2 <- fit_sub_model(d3,nlevels=2,hess=F,maxb=maxb)
  logLik <- -c(logLikOpti, meta_fit$value, sub2$value)
  AIC <- 2*n_par -2*logLik
  d3_fit <- data.frame(generative="discrete (2 lvl)",
                       fitted=c("Bayesian","biased-Bayesian","discrete"),
                       L = logLik, AIC = AIC,
                       relModLik = exp(-(AIC - min(AIC))/2),
                       AkaikeWeight = exp(-(AIC - min(AIC))/2) / sum(exp(-(AIC - min(AIC))/2)),
                       par1 = c(NA, meta_fit$p1, sub2$par[1]),
                       par2 = c(NA, NA, sub2$par[2]),
                       gpar1 = c(NA, mnoise_i , sub2_1_i),
                       gpar2 = c(NA, NA, sub2_2_i),id=id)
  
  # store results
  d1$generative <- "Bayesian"
  d2$generative <- "biased-Bayesian"
  d3$generative <- "discrete (2 lvl)"
  d1$id <- id
  d2$id <- id
  d3$id <- id
  sim_data <- rbind(sim_data, d1, d2, d3)
  sim_fit <- rbind(sim_fit, d1_fit, d2_fit, d3_fit)
  save(sim_data, sim_fit, file="D:/xiangmu/data/behavior_for_behavior/model_recovery_res.RData")
  cat("\tsim id: ",id," completed;\n")
}
cat("Model recovery simulation completed on ",date(),"\n")

aggregate(L~fitted+generative,sim_fit, mean, na.rm=T)


# ------------------------------------------------------#
load("D:/xiangmu/data/behavior_for_behavior/model_recovery_res.RData")

sub_2_col <- rgb(80,80,255,maxColorValue=255)
bayes_col <- rgb(255,80,80,maxColorValue=255)
meta_col <- rgb(255,150,0,maxColorValue=255)

d_plot_2 <- aggregate(relModLik~generative+fitted, sim_fit, mean, na.rm=T)
d_plot_2$L <- d_plot_2$relModLik
d_plot_2$L_se <- aggregate(relModLik~generative+fitted, sim_fit, bootMeanSE)$relModLik
d_plot_2$cl <- factor(c(rep(1,3),rep(2,3),rep(3,3)))
levels(d_plot_2$generative) <-  paste("data generated from\n",levels(d_plot_2$generative))

library(ggplot2)
nice_theme <- theme_bw()+theme(text=element_text(family="Helvetica",size=9),panel.border=element_blank(),strip.background = element_rect(fill="white",color="white",size=0),strip.text=element_text(size=rel(0.8)),panel.grid.major.x=element_blank(),panel.grid.major.y=element_blank(),panel.grid.minor=element_blank(),axis.line.x=element_line(size=.4),axis.line.y=element_line(size=.4),axis.text.x=element_text(size=7,color="black"),axis.text.y=element_text(size=7,color="black"),axis.line=element_line(size=.4), axis.ticks=element_line(color="black"))

pl1 <- ggplot(d_plot_2, aes(y=fitted, x=L, xmin=L-L_se, xmax=L+L_se, color=cl))+geom_vline(xintercept=0,lty=2,size=0.4)+geom_errorbarh(height=0,size=7)+geom_point(pch="I",size=6.5,color="black")+labs(y="fitted model",x=expression(paste("likelihood (of model ",italic("m"),"),  exp[",-1/2 ("AIC"[italic("m")] - "min AIC"),"]")))+scale_color_manual(values=c(bayes_col, meta_col, sub_2_col),guide=F)+nice_theme+ theme(panel.grid.major.y = element_line(colour="grey", size=0.2,linetype=1))+scale_x_continuous(breaks=seq(0,1,0.5))+facet_grid(.~generative)

ggsave(plot=pl1, filename="D:/xiangmu/figures/behavior_for_behavior/model_recovery.pdf", device="pdf",width=7.5,height=1.7)

