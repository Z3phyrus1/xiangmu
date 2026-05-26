# ------------------------------------------------------#
# Analysis of response times
# Matteo Lisi, 2019
# ------------------------------------------------------#
# clear
rm(list=ls())
setwd("D:/xiangmu/scripts/R")

# ------------------------------------------------------#
# some plotting libraries and settings
sub_2_col <- rgb(80,80,255,maxColorValue=255)
bayes_col <- rgb(255,80,80,maxColorValue=255)
opti_col <- "black"
library(ggplot2)
library(viridis)
library(mlisi)
nice_theme <- theme_bw()+theme(text=element_text(family="Helvetica",size=9),panel.border=element_blank(),strip.background = element_rect(fill="white",color="white",size=0),strip.text=element_text(size=rel(0.8)),panel.grid.major.x=element_blank(),panel.grid.major.y=element_blank(),panel.grid.minor=element_blank(),axis.line.x=element_line(size=.4),axis.line.y=element_line(size=.4),axis.text.x=element_text(size=7,color="black"),axis.text.y=element_text(size=7,color="black"),axis.line=element_line(size=.4), axis.ticks=element_line(color="black"))

setwd("D:/xiangmu/scripts/R")
source("./R/code_all_models.R")
library(ggpubr)

# ------------------------------------------------------#
# load data and model parameters
dfit <- read.table("D:/xiangmu/data/behavior_for_behavior/fit_par.txt",header=T)#D:/reysy/EEG/data/fit_par.txt
d <- read.table(file="D:/xiangmu/data/behavior_for_behavior/data_wide_wmPred.txt", sep=";", header=T)#D:/reysy/EEG/data/data_wide_wmPred.txt

# check range of RTs
range(d$rt1)
range(d$rt2)

# log transform
d$logrt2 <- log(d$rt2)
d$logrt1 <- log(d$rt1)

# cut offs
round(mean(d$rt1>5 | d$rt2>5)*100, digits=2)
d <- d[d$rt2<=5,]
d <- d[d$rt1<=5,]

# # add stimulus duration to obtain full response time
# (i.e. from the stimulus onset)
d$rt1 <- d$rt1 + 0.3
d$rt2 <- d$rt2 + 0.3

# ------------------------------------------------------#
# applyt cut-offs also to d1, and compare RTs in decision 1 nad 2
d$abs1 <- abs(d$s1)
d$abs2 <- abs(d$s2)

dag <- aggregate(cbind(rt1,rt2)~id,d, mean)
summary(dag)
t.test(dag$rt1,dag$rt2,paired=T)

round(c(mean(dag$rt1),sd(dag$rt1)),digits=2)
round(c(mean(dag$rt2),sd(dag$rt2)),digits=2)

cut_points <- c(0, 0.5, 1, 1.5, 2, 20)
d$bin_s1 <- cut(d$abs1, breaks=cut_points)
d$bin_s2 <- cut(d$abs2, breaks=cut_points)

dag1 <- aggregate(cbind(rt1,logrt1, abs1) ~ bin_s1 + id, d, mean)
dag01 <- aggregate(cbind(rt1,logrt1, abs1) ~ bin_s1, dag1, mean)
dag01$rt1_se <- aggregate(rt1 ~ bin_s1, dag1, bootMeanSE)$rt1
dag01$decision <- "1st decision"

dag1 <- aggregate(cbind(rt2,logrt2, abs2) ~ bin_s2 + id, d, mean)
dag02 <- aggregate(cbind(rt2,logrt2, abs2) ~ bin_s2, dag1, mean)
dag02$rt2_se <- aggregate(rt2 ~ bin_s2, dag1, bootMeanSE)$rt2
dag02$decision <- "2nd decision"

colnames(dag01) <- c("bin", "rt", "logrt", "abs_s", "se", "decision")
colnames(dag02) <- c("bin", "rt", "logrt", "abs_s", "se", "decision")
dag <- rbind(dag01, dag02)

prt01 <- ggplot(dag, aes(x=abs_s, y=rt, ymin=rt-se, ymax=rt+se, color=decision))+geom_line()+geom_point()+geom_errorbar(width=0.05)+nice_theme+labs(x=expression(paste("discriminability [",sigma,"]") ),y="response time [sec]")+scale_color_manual(values=c("dark grey","black"),name=NULL)+ theme(legend.box.background = element_blank(),legend.position=c(.7,.9))+scale_y_continuous(limits=c(0.5, 1))
dir.create("D:/xiangmu/figures/behavior_idsess", recursive = TRUE)
pdf("D:/xiangmu/figures/behavior_for_behavior/rt_01_430.pdf",width=2,height=2.4)
prt01
dev.off()

## do the same but for control conditions
# load data
d_c <- read.table("D:/xiangmu/data/behavior_for_behavior/data_nu.txt",header=T, sep=";")#D:/reysy/EEG/data/data_nu.txt
d_c <- d_c[d_c$dual==0,]#改了一下，原为0

# put the data frames in a "wide" format
d_c <- data.frame(s1=d_c$s[d_c$decision=="1"], s2=d_c$s[d_c$decision=="2"], r1=d_c$rr[d_c$decision=="1"], r2=d_c$rr[d_c$decision=="2"], acc1=d_c$acc[d_c$decision=="1"],acc2=d_c$acc[d_c$decision=="2"], id=d_c$id[d_c$decision=="1"], task=ifelse(is.na(d_c$meanTilt[d_c$decision=="1"]),"M-O","O-M"), dataset=1, rt1=d_c$tResp[d_c$decision=="1"],rt2=d_c$tResp[d_c$decision=="2"])

d_c <- d_c[d_c$rt2<=5,]
d_c <- d_c[d_c$rt1<=5,]

# # add stimulus duration (form stimulus onset)
d_c$rt1 <- d_c$rt1 + 0.3
d_c$rt2 <- d_c$rt2 + 0.3

d_c$abs1 <- abs(d_c$s1)
d_c$abs2 <- abs(d_c$s2)
d_c$logrt2 <- log(d_c$rt2)
d_c$logrt1 <- log(d_c$rt1)

dag <- aggregate(cbind(rt1,rt2)~id,d_c, mean)
summary(dag)
t.test(dag$rt1,dag$rt2,paired=T)
library(BayesFactor)
1/ttestBF(x=dag$rt1,y=dag$rt2,paired=T, rscale=1)
round(c(mean(dag$rt1),sd(dag$rt1)),digits=2)
round(c(mean(dag$rt2),sd(dag$rt2)),digits=2)

# ------------------------------------------------------#
# functions to calculate 'expected' confidence level
# ------------------------------------------------------#
# probability of observer being in "confident" state conditioned on first stimulus
# AND on the first decision
pc_ds1 <- function(obs, par)
{
  s1_ <- ((obs$r1*2)-1)*obs$s1
  p_c <- 1 - ptruncnorm(par[2], mu=s1_,sigma=par[1], a=0)
}

# probability of observer being in "confident" state conditioned on first stimulus
# first decision, AND second stimulus and decision
pc_ds1_ds2 <- function(obs, par)
{
  s1_ <- ((obs$r1*2)-1)*obs$s1
  p_c <- 1 - ptruncnorm(par[2], mu=s1_,sigma=par[1], a=0)
  
  tr2 <- obs$r2*2 -1
  p_d2_c <- (1-obs$r2 + tr2*pnorm((obs$s2-par[3])/par[1]))
  p_d2 <-((p_c*p_d2_c + (1-p_c)*(1-obs$r2 + tr2*pnorm(obs$s2/par[1]))))
  
  # compute prob. of being confident using bayes theorem
  P <- (p_d2_c * p_c) / p_d2
  return(P)
}

# ------------------------------------------------------#
# average confidence of ideal model
prior_theta2 <- function(theta2, s1,d1){
  bar_theta2 <- ifelse(sign(d1*2 -1)==sign(s1),-abs(s1),abs(s1))
  return(dtruncnorm(theta2, mu=bar_theta2, sigma=1, a=-Inf,b=0))
}
unn_posterior_theta2 <- function(theta2, s1, s2, d1, d2){
  if(d2==1){
    p_r1 <- prior_theta2(theta2, s1,d1) * pnorm(s2,mean=theta2)
  }else{
    p_r1 <- prior_theta2(theta2, s1,d1) * (1-pnorm(s2,mean=theta2))
  }
  return(p_r1)
}
norm_constant_post_theta2 <- function(s1, s2, d1, d2){
  if(d2==1){
    integrand <- function(theta2) prior_theta2(theta2, s1,d1) * pnorm(s2,mean=theta2)
  }else{
    integrand <- function(theta2) prior_theta2(theta2, s1,d1) * (1-pnorm(s2,mean=theta2))
  }
  return(integrate(integrand, lower=-Inf, upper= Inf)$value)
}
posterior_theta2 <- function(theta2, s1, s2, d1, d2){
  return(unn_posterior_theta2(theta2, s1, s2, d1, d2)/norm_constant_post_theta2(s1, s2, d1, d2))
}

mean_ideal_confidence <- function(s1, s2, d1, d2){
  c1 <- rep(NA,length(s1))
  for(i in 1:length(s1)){
    mean_theta2 <- integrate(function(x) x*posterior_theta2(x,s1[i],s2[i],d1[i],d2[i]), lower=-Inf, upper= Inf)$value
    c1[i]<-pnorm(-mean_theta2)
  }
  return(c1)
}

# ------------------------------------------------------#
# compute probability of being confident for each trial
d <- read.table(file="D:/xiangmu/data/behavior_for_behavior/data_wide_wmPred.txt", sep=";", header=T)#D:/reysy/EEG/data/data_wide_wmPred.txt
d$p_conf <- NA
d$p_conf_ideal <- NA
for(s_i in unique(d$id)){
  s2_1 <- dfit$w1[dfit$model=="sub2" & dfit$id==s_i]
  s2_2 <- dfit$theta2[dfit$model=="sub2" & dfit$id==s_i]
  d$p_conf[d$id==s_i] <- pc_ds1_ds2(d[d$id==s_i,], c(1, s2_1, s2_2))
  d$p_conf_ideal[d$id==s_i] <- mean_ideal_confidence(d$s1[d$id==s_i], d$s2[d$id==s_i], d$r1[d$id==s_i],d$r2[d$id==s_i])
}

d$logrt2 <- log(d$rt2)
d$logrt1 <- log(d$rt1)
write.table(d, "D:/xiangmu/data/behavior_for_behavior/all_RTanalysis.txt",row.names=F, sep="/t")#D:/reysy/EEG/data/all_RTanalysis.txt
#d <- read.table("./data/all_RTanalysis.txt",header=T, sep="/t")

# cut offs
round(mean(d$rt1>5 | d$rt2>5)*100, digits=2)

d <- d[d$rt2<=5,]
d <- d[d$rt1<=5,]

# # add stimulus duration (form stimulus onset)
d$rt1 <- d$rt1 + 0.3
d$rt2 <- d$rt2 + 0.3

# # ------------------------------------------------------#
# # because responses were not recorded during stimulus presentation
# # in this case the responses are effectively censored at 0.3 sec
# # set "hard" boundary at 0.31

# prepare variables
d$abs1 <- abs(d$s1)
d$abs2 <- abs(d$s2)
d$logoddconf <- log(d$p_conf/(1-d$p_conf))
d$logoddconf_ideal <- log(d$p_conf_ideal/(1-d$p_conf_ideal))
d$logrt2 <- log(d$rt2)
d$highconf <- ifelse(d$p_conf>0.5,1,0)

d$abs1_squared <- d$abs1^2
d$abs2_squared <- d$abs2^2

## Analysis
library(lme4)
m0 <- lmList(logrt2 ~ abs1 + abs2 + abs1*abs2 + acc1 + acc2 | id, d)
coef(m0)
t.test(coef(m0)[,2])
t.test(coef(m0)[,3])
t.test(coef(m0)[,4])
t.test(coef(m0)[,5])
d$m0_residu <- residuals(m0)
d$pred_logrt2 <- predict(m0)
d$pred_rt2 <- exp(d$pred_logrt2)

library(gridExtra)
library(viridis)

### binning by the confidence levels
n_bin <- 20
d$conf_ideal_bin <- NA
d$conf_ideal_bin <- as.character(cut(d$p_conf_ideal, breaks=quantile(d$p_conf_ideal, prob=seq(0,1,1/n_bin))))

dag0 <- aggregate(cbind(p_conf_ideal, p_conf, m0_residu, logrt2, rt2, abs1) ~ conf_ideal_bin + id, d, mean)
dag <- aggregate(cbind(p_conf_ideal, p_conf, m0_residu, logrt2, rt2, abs1) ~ conf_ideal_bin, dag0, mean)
dag$p_conf_se <- aggregate(p_conf ~ conf_ideal_bin, dag0, bootMeanSE)$p_conf
dag$p_conf_ideal_se <- aggregate(p_conf_ideal ~ conf_ideal_bin, dag0, bootMeanSE)$p_conf_ideal
dag$rt2_se <- aggregate(rt2 ~ conf_ideal_bin, dag0, bootMeanSE)$rt2
dag$m0_residu_se <- aggregate(m0_residu ~ conf_ideal_bin, dag0, bootMeanSE)$m0_residu

dag$IR1 <- sqrt(2)*erf.inv(1-2*dag$p_conf_ideal)
dag_ideal <- dag

# add median criterion ?
w1 <- dfit$w1[dfit$model=="sub2"]
w1 <- w1[-which(w1==0)]
ci_w1 <- -(erf(-w1/sqrt(2)) -1 )/2
TH_w1 <- mean(ci_w1)


pdf("D:/xiangmu/figures/behavior_for_behaivor/RT_resid.pdf",width=3.36,height=2.4)
ggplot(dag_ideal, aes(x=p_conf_ideal,y=m0_residu, color=rt2))+geom_hline(yintercept=0,lty=2,size=0.3,col="black")+geom_smooth(data=d,method="gam",col="dark grey",size=0.4,se=T,alpha=0.1)+stat_smooth(data=d,method="gam", geom = "ribbon", fill = NA, linetype = "dotted", colour = "dark grey",size=0.2)+geom_vline(xintercept=TH_w1, col=sub_2_col, size=0.4)+geom_point(size=1.8)+nice_theme+labs(x="expected confidence (1st decision)", y="residual, log RT")+scale_color_viridis(name="response time/n(2nd decision)/n[sec]",option="A",direction=-1,begin=0,end=0.9,limits=range(dag$rt2))+geom_errorbar(aes(ymin=m0_residu-m0_residu_se, ymax=m0_residu+m0_residu_se),width=0,size=0.3)+geom_errorbarh(aes(xmin=p_conf_ideal-p_conf_ideal_se, xmax=p_conf_ideal+p_conf_ideal_se),height=0,size=0.2)+theme(legend.title=element_text(size=7),legend.text=element_text(size=7))+theme(legend.key.size =  unit(0.15, "in"))+coord_cartesian(ylim=c(-0.10,0.13))
dev.off()
