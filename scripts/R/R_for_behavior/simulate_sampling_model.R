# ------------------------------------------------------#
# this script simulate the Bayesian sampling observer
# on the "true" dataset, conditioned on the observed
# pattern of first choices
#
# The standard deviation of the internal noise is adjusted for 
# each model so as to achieve the same observed
# first decision accuracy of the ideal with sigma=1
#
# Matteo Lisi, 2019
# updated 04/202
# ------------------------------------------------------#

rm(list=ls())
setwd("D:/11.13")
source("D:/11.13/nhb_data_code/R/code_all_models.R")
library(truncnorm)

# ------------------------------------------------------#
# helpers

# probability of r1=1
p_r1_sampling <- function(obs, eta, n){
  k1 <- ceiling(n/2)
  p <- 0
  for(k_i in k1:n){
    integrand <- function(x) pnorm(x/eta)^k_i * (1-pnorm(x/eta))^(n-k_i) * dnorm(x,obs$s1,eta)
    p <- p + choose(n,k_i) * integrate(integrand, lower=-Inf, upper= Inf)$value
  }
  return(p)
}

# likelihood of first choices
l_sampling_d1 <- function (eta, d, n) 
{
  L <- vector(length=nrow(d),mode="numeric")
  for(i in 1:nrow(d)){
    if(d$r1[i]==1){
      L[i] <- p_r1_sampling(d[i,], eta, n)
    }else{
      L[i] <- 1- p_r1_sampling(d[i,], eta, n)
    }
  }
  return( -sum(log(L)) )
}

# proportion of samples exceeding threshold
prop_above_th <- function(v, th) {sum(v>=th)/length(v)}

# simulate sampling model; not conditioned on observed pattern of R1
sim_sampling <- function(obs, eta, nsamples=3 ,rep = 1){
  obs <- obs[rep(seq_len(nrow(obs)), rep), ]
  obs$rep <- sort(rep_len(1:rep, nrow(obs)))
  ir1 <- obs$s1 + rnorm(length(obs$s1),0,eta)
  ir1_s <- {} # sampled resposes (add a column for each sample)
  for(s in 1:nsamples){
    ir1_s <- cbind(ir1_s, ir1 + rnorm(length(ir1),0,eta))
  }
  sc1 <- apply(ir1_s, 1, prop_above_th, 0)
  obs$r1 <- ifelse(sc1==0.5, rbinom(1,1,0.5), ifelse(sc1>0.5,1,0))
  
  # stimulus for second decision
  acc1 <- (obs$r1*2 -1) * sign(obs$s1)
  obs$s2 <- acc1 * abs(obs$s2)
  
  # confidence and criterion shift
  c1 <- ifelse(sc1 >= (1-sc1), sc1, (1-sc1))
  theta2 <- sqrt(2) * erf.inv(-2*c1 + 1)
  
  # second decision
  ir2 <- obs$s2 + rnorm(length(obs$s2),0,eta)
  ir2_s <- {} 
  for(s in 1:nsamples){
    ir2_s <- cbind(ir2_s, ir2 + rnorm(length(ir2),0,eta))
  }
  # tababove <- {}; for(i in 1:nsamples){ tababove <- cbind(tababove, ir2[,i]>=theta2) }
  # all((ir2>=theta2) == tababove)
  sc2 <- apply(ir2_s>=theta2, 1, sum)/nsamples
  obs$r2 <- ifelse(sc2==0.5, rbinom(1,1,0.5), ifelse(sc2>0.5,1,0))
  obs$rep <- factor(obs$rep)
  return(obs)
}


# simulate sampling conditioned on observed R1
# require truncnorm package
sim_sampling_cond <- function(obs, eta, nsamples=3 ,rep = 1){
  obs <- obs[rep(seq_len(nrow(obs)), rep), ]
  obs$rep <- sort(rep_len(1:rep, nrow(obs)))
  
  # use truncated normal to draw samples conditioned on observed choices
  a_i <- ifelse(obs$r1==1,0,-Inf)
  b_i <- ifelse(obs$r1==1,Inf,0)
  ir1 <- rtruncnorm(nrow(obs),a=a_i,b=b_i,mean=obs$s1,sd=eta)
  
  ir1_s <- {} # sampled resposes (add a column for each sample)
  n_samples_r1side <- ceiling(nsamples/2)
  # need to ensure that the majority of samples is on the correct side of threshold
  for(s in 1:n_samples_r1side){
    ir1_s <- cbind(ir1_s, rtruncnorm(nrow(obs),a=a_i,b=b_i,mean=ir1,sd=eta))
  }
  for(s in 1:(nsamples-n_samples_r1side)){
    ir1_s <- cbind(ir1_s, rnorm(nrow(obs),mean=ir1,sd=eta))
  }
  sc1 <- apply(ir1_s, 1, prop_above_th, 0)
  # r1_sim <- ifelse(sc1==0.5, rbinom(1,1,0.5), ifelse(sc1>0.5,1,0))
  # all(r1_sim==obs$r1) # sanity check
  
  # stimulus for second decision
  acc1 <- (obs$r1*2 -1) * sign(obs$s1)
  obs$s2 <- acc1 * abs(obs$s2)
  
  # confidence and criterion shift
  c1 <- ifelse(sc1 >= (1-sc1), sc1, (1-sc1))
  theta2 <- sqrt(2) * erf.inv(-2*c1 + 1)
  
  # second decision
  ir2 <- obs$s2 + rnorm(length(obs$s2),0,eta)
  ir2_s <- {} 
  for(s in 1:nsamples){
    ir2_s <- cbind(ir2_s, ir2 + rnorm(length(ir2),0,eta))
  }
  # tababove <- {}; for(i in 1:nsamples){ tababove <- cbind(tababove, ir2[,i]>=theta2) }
  # all((ir2>=theta2) == tababove)
  sc2 <- apply(ir2_s>=theta2, 1, sum)/nsamples
  obs$r2 <- ifelse(sc2==0.5, rbinom(1,1,0.5), ifelse(sc2>0.5,1,0))
  obs$rep <- factor(obs$rep)
  return(obs)
}

# ------------------------------------------------------#
# finally simulate models on the empirical dataset
n_samples <- c(2,3,5,10,20)
n_sim <- 10
library(tidyr)

d <- read.table(file="D:/reysy/dual-decision-task-master/dual-decision-task-master/data/data_wide_wmPred.txt", sep=";", header=T)

#
fctrl <- list(maxit=100000000)

#
d_sall <- {}
acc_i <- {}
acc_s <- matrix(NA,length(unique(d$id)), length(n_samples))

# use the same breaks as in previous plots
cut_points <- c(0, 0.5, 1, 1.5, 2, 20)
break_s1 <- cut_points # quantile(d$abs1, probs=seq(0,1,length.out=8))

cat("\nBegin simulation of sampling model...")
c_i <- 0
for(i in unique(d$id)){
  
  c_i <- c_i + 1
  
  d_i <- with(d[d$id==i,], data.frame(s1=s1, r1=r1, s2=s2, r2=r2, acc1=acc1))
  acc_i <-c(acc_i, mean(d_i$acc1))
  d_i$s1bin <- cut(abs(d_i$s1),break_s1, right=F)
  
  cat("\n\nconditioned on subject ",i,", samples: ")
  
  n_i <- 0
  
  for(n in n_samples){
    
    n_i <- n_i + 1
    cat(n," ")
    
    fit_samp <- optim(par=1, l_sampling_d1, d=d_i, n=n, method="Brent", upper=5, lower=0, control=fctrl)
    d_ss <- sim_sampling_cond(d_i, fit_samp$par, n, n_sim)
    d_sc <- sim_sampling(d, fit_samp$par, n, n_sim)
    acc_s[c_i, n_i] <- mean(d_sc$s2>=0)
    rm(d_sc)
    d_s <- aggregate(cbind(r2,abs(s1)) ~ s1bin, d_ss, mean)
    colnames(d_s)[3] <- "s1"
    
    d_s$id <- i
    d_s$n <- n
    d_s$noise <- fit_samp$par
    d_sall <- rbind(d_sall, d_s)
    rm(d_ss)
  }
  write.table(d_sall, file="D:/reysy/dual-decision-task-master/dual-decision-task-master/data/sampling_sim.txt",row.names=F)
}
cat("\n\n...done!")

d_sall <- read.table("D:/reysy/dual-decision-task-master/dual-decision-task-master/data/sampling_sim.txt", header=T)
str(d_sall)

# ------------------------------------------------------#
# do plot with observed data
library(ggplot2)
library(mlisi)
nice_theme <- theme_bw()+theme(text=element_text(family="Helvetica",size=9),panel.border=element_blank(),strip.background = element_rect(fill="white",color="white",size=0),strip.text=element_text(size=rel(0.8)),panel.grid.major.x=element_blank(),panel.grid.major.y=element_blank(),panel.grid.minor=element_blank(),axis.line.x=element_line(size=.4),axis.line.y=element_line(size=.4),axis.text.x=element_text(size=7,color="black"),axis.text.y=element_text(size=7,color="black"),axis.line=element_line(size=.4), axis.ticks=element_line(color="black"))
library(viridis)
bayes_col <- rgb(255,80,80,maxColorValue=255)


cut_points <- c(0, 0.5, 1, 1.5, 2, 20)
d$bin_s1 <- cut(d$abs1, breaks=cut_points) # this is for observed data
d$x_s1 <- cut(d$abs1, breaks=cut_points) # model



d_m_1 <- aggregate(cbind(p_r2_optimal, abs1) ~ x_s1 + id, d, mean)
d_m <- aggregate(cbind(p_r2_optimal,abs1) ~ x_s1, d_m_1, mean)
m_ci <- function(v,alpha=0.05){
  se_value <- sd(v) / sqrt(length(v))
  ci <- 2 * se_value
  return(ci)
}
d_m$se <- aggregate(p_r2_optimal ~ x_s1, d_m_1, m_ci)$p_r2_optimal
colnames(d_m) <- c("bin_s1","r2","s1","se")
d_m$model <-"Bayesian (optimal)"
d_m$n <- Inf

d_sam <- aggregate(cbind(s1,r2)~s1bin+n,d_sall,mean)
d_sam$se <- aggregate(r2~s1bin+n,d_sall,m_ci)$r2
d_sam$model <- paste("n samples",d_sam$n)
colnames(d_sam)[1] <- "bin_s1"
d_sam$model <- reorder(d_sam$model, d_sam$n)

d_plot <- rbind(d_sam, d_m)

dag <- aggregate(cbind(abs1, r2) ~ bin_s1 + id, d, mean)
dag2 <- aggregate(cbind(abs1,  r2) ~ bin_s1 , dag, mean)
dag2$r2_se <- aggregate(r2 ~  bin_s1, dag, bootMeanSE)$r2
dag2$model <-"observed\ndata"
colnames(dag2) <- c("bin_s1", "s1", "r2", "se","model" )

model_plot <- "Bayesian sampler"
pdf(paste("D:/reysy/dual-decision-task-master/dual-decision-task-master/fig/1",model_plot,"_r2.pdf",sep=""),width=3.3,height=2.2)
ggplot(d_plot,aes(x=s1, y=r2, color=model))+nice_theme+geom_line(size=0.8)+coord_cartesian(xlim=c(0,2.5),ylim=c(0.65,1))+labs(x =bquote(atop(paste("discriminability [",sigma,"]"), 1^"st"~"decision")), y=expression(paste(p(italic("choose right")),", ",2^"nd"," decision ") ))+scale_color_manual(values=c(viridis_pal(direction=-1,option="D")(5),bayes_col),guide=guide_legend(title=NULL,keyheight=0.8,label.theme = element_text(size=7,angle=0)))+geom_line(data=dag2,color="black")+geom_point(data=dag2,color="black",pch=21)+geom_errorbar(data=dag2,aes(ymin=r2-se,ymax=r2+se),width=0,color="black")+ggtitle(model_plot)+theme(plot.title = element_text(size = 8))
dev.off()






