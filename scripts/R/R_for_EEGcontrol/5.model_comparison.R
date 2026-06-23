# ------------------------------------------------------------------ #
# This script does the following (not necessarily in this order)
# 1) calculate group-level estimates of goodness of fit (AIC)
# 2) perform Bayesian model comparison (calculate exceedance probabilities)
# 3) plot the results
# 4) plot the average confidence functions, based on estiamted parameters
# Matteo Lisi, 2020
# ------------------------------------------------------------------ #

rm(list=ls())
setwd("D:/xiangmu/scripts/R")
source("./R/code_all_models.R")

# ------------------------------------------------------------------ #
# some plotting libraries and settings
sub_2_col <- rgb(80,80,255,maxColorValue=255)
bayes_col <- rgb(255,80,80,maxColorValue=255)
opti_col <- "black"
library(ggplot2)
library(viridis)
library(mlisi)
nice_theme <- theme_bw()+theme(text=element_text(family="Helvetica",size=9),panel.border=element_blank(),strip.background = element_rect(fill="white",color="white",size=0),strip.text=element_text(size=rel(0.8)),panel.grid.major.x=element_blank(),panel.grid.major.y=element_blank(),panel.grid.minor=element_blank(),axis.line.x=element_line(size=.4),axis.line.y=element_line(size=.4),axis.text.x=element_text(size=7,color="black"),axis.text.y=element_text(size=7,color="black"),axis.line=element_line(size=.4), axis.ticks=element_line(color="black"))

# ------------------------------------------------------------------ #
# load 
dfit <- read.table("D:/xiangmu/data/behavior_for_behavior/fit_par.txt",header=T)#D:/reysy/EEG/data/fit_par.txt

# ------------------------------------------------------------------ #

dfit_ag <- aggregate(AIC ~ model, dfit, sum)
dfit_ag$AIC_diff <- dfit_ag$AIC - min(dfit_ag$AIC)
dfit_ag


# individual participants
dfit_ag <- aggregate(AIC ~ model+id, dfit,sum)
tapply(dfit_ag$AIC, list(dfit_ag$model,dfit_ag$id),mean) -matrix(rep(tapply(dfit_ag$AIC, list(dfit_ag$model,dfit_ag$id),mean)[1,],4),4,length(unique(dfit_ag$id)),byrow=T)

# # ------------------------------------------------------#
# # plot confidence functions (as in Fig 3)
par_s2 <- c(median(dfit$w1[dfit$model=="sub2"],na.rm=T),median(dfit$theta2[dfit$model=="sub2"],na.rm=T))
par_m <- median(dfit$m[dfit$model=="metanoise"],na.rm=T)
par_fixb <- median(dfit$fixed_shift,na.rm=T)

evidence <- seq(0,2,length.out=500)
conf_opti <- pnorm(evidence)
conf_m <- pnorm(evidence, sd=par_m)
conf_s2 <- ifelse(evidence<par_s2[1],0.5, (1-erf(par_s2[2]/sqrt(2)))/2)

conf_fixb <- (1-erf(par_fixb/sqrt(2)))/2
d_cf <- data.frame(evidence, conf_opti, conf_s2, conf_fixb, conf_m)

bayes_col <- rgb(255,80,80,maxColorValue=255)
meta_col <- rgb(255,150,0,maxColorValue=255)

pdf("D:/xiangmu/figures/behavior_for_behavior/conf_fixb.pdf",width=1.6, height=1.6)
ggplot(d_cf)+geom_line(aes(x=evidence,y=conf_opti),size=1,color=bayes_col)+nice_theme+labs(x=expression(paste("evidence [",sigma,"]")),y="confidence")+geom_line(aes(x=evidence,y=conf_fixb),size=1,color="dark grey")+annotate("text",x=1,y=pnorm(2.5),label="ideal Bayesian",color=bayes_col,size=2.5)+annotate("text",x=1.5,y=conf_fixb[1]-0.03,label="fixed-bias",color="dark grey",size=2.5)
dev.off()

pdf("D:/xiangmu/figures/behavior_for_behavior/conf_bayes.pdf",width=1.6, height=1.6)
ggplot(d_cf)+geom_line(aes(x=evidence,y=conf_opti),size=1,color=bayes_col)+nice_theme+labs(x=expression(paste("evidence [",sigma,"]")),y="confidence")+geom_line(aes(x=evidence,y=conf_m),size=1,color=meta_col)+annotate("text",x=1,y=pnorm(2.5),label="ideal Bayesian",color=bayes_col,size=2.5)+annotate("text",x=1.35,y=0.545,label="biased-Bayesian",color=meta_col,size=2.5)
dev.off()

pdf("D:/xiangmu/figures/behavior_for_behavior/conf_discrete.pdf",width=1.6, height=1.6)
ggplot(d_cf)+geom_line(aes(x=evidence,y=conf_opti),size=1,color=bayes_col)+nice_theme+labs(x=expression(paste("evidence [",sigma,"]")),y="confidence")+geom_line(aes(x=evidence,y=conf_s2),size=1,color="blue")+annotate("text",x=1.58,y=0.87,label="ideal",color=bayes_col,size=2.5)+annotate("text",x=1.58,y=0.82,label="Bayesian",color=bayes_col,size=2.5)+annotate("text",x=1.2,y=0.55,label="discrete",color="blue",size=2.5)
dev.off()

# ------------------------------------------------------#
# average noise over-estimation bias
M <- dfit$m[!is.na(dfit$m)]
mean(M)
bootMeanCI(M)

# ------------------------------------------------------#
# statistical tests on AIC values (fixed-effects model comparison)

m_aic <- aov(AIC~model+Error(id),dfit)

library(sjstats)
effectsize::eta_squared(m_aic, partial=T)

summary(m_aic)
t.test(dfit$AIC[dfit$model=="metanoise"], dfit$AIC[dfit$model=="sub2"],paired=T)
t.test(dfit$AIC[dfit$model=="bayes"], dfit$AIC[dfit$model=="sub2"],paired=T)
t.test(dfit$AIC[dfit$model=="bayes"], dfit$AIC[dfit$model=="sub1"],paired=T)
t.test(dfit$AIC[dfit$model=="sub1"], dfit$AIC[dfit$model=="sub2"],paired=T)
t.test(dfit$AIC[dfit$model=="sub1"], dfit$AIC[dfit$model=="metanoise"],paired=T)

# ------------------------------------------------------#
# plot pooled AIC values (for plot)
d_plot_i <- {}
for(i in unique(dfit$id)){
  AIC_i <- with(dfit[dfit$id==i,], tapply(AIC, model, mean))#[1:4]
  AIC_diff_i <- AIC_i - AIC_i[which(names(AIC_i)=="sub2")]
  d_i <- data.frame(AIC=AIC_i,AIC_diff=AIC_diff_i,model=c("Bayesian","biased-Bayesian","fixed bias","discrete"),id=i,cl= factor(c(1, 2, 3, 4)))
  d_plot_i <- rbind(d_plot_i, d_i)
}


# bootstrap SE for sum
bootSumSE <- function(v,nsim=1000,...){
  bootfoo <- function(v,i) sum(v[i],na.rm=T,...)
  bootRes <- boot::boot(v,bootfoo,nsim)
  return(sd(bootRes$t,na.rm=T))
}

d_plot <- aggregate(AIC_diff ~ model, d_plot_i, sum)
d_plot$cl <- factor(c(1, 3, 2, 4))
d_plot$se <- aggregate(AIC_diff ~ model, d_plot_i, bootSumSE)$AIC_diff

# these would be the confidence intervals
aggregate(AIC_diff ~ model, d_plot_i, function(x){bootFooCI(x, foo="sum",nsim=10000)})

d_plot$model <- reorder(d_plot$model, d_plot$AIC_diff)

d_plot_i$se <- NA

bayes_col <- rgb(255,80,80,maxColorValue=255)
meta_col <- rgb(255,150,0,maxColorValue=255)

pdf("D:/xiangmu/figures/behavior_for_behavior/model_comp.pdf",height=4.8,width=1.3)
ggplot(d_plot,aes(x=model,y=AIC_diff, ymin=AIC_diff-se,ymax=AIC_diff+se,fill=model))+geom_col(width=0.7)+geom_hline(yintercept=0,lty=2)+geom_errorbar(width=0)+nice_theme+scale_fill_manual(values=c(sub_2_col,meta_col,"dark grey",bayes_col ),guide=F)+labs(x="",y="summed AIC difference")+ theme(axis.text.x = element_text(angle = 45, hjust = 1))+scale_y_continuous(breaks=seq(0,10000,100))
dev.off()

# ------------------------------------------------------#
# Bayesian (random-effects) model selection
library(bmsR)
m <- tapply(dfit$relModLik, list(dfit$id, dfit$model), mean)[,1:4]
m <- m/repmat(t(t(apply(m,1,sum))),1,4)
m <- log(m)
if (any(is.na(m))) {
  # 方法一：删除包含缺失值的行
  m <- na.omit(m)
  
  # 方法二：用均值填充缺失值
  # m[is.na(m)] <- mean(m, na.rm = TRUE)
}
bms0 <- VB_bms(m, n_samples = 1e+06)

colnames(m) # check the order
d_xp <- data.frame(xp=bms0$pxp, model=c("Bayesian","biased-Bayesian","fixed-bias","discrete"), cl=factor(c(4, 3, 2, 1)),freq=bms0$r)
d_xp$model <- reorder(d_xp$model, c(4,3,2, 1))

pdf("D:/xiangmu/figures/behavior_for_behavior/model_freq.pdf",height=2,width=1.3)
ggplot(d_xp,aes(x=model,y=freq, fill=cl))+geom_hline(yintercept=0,lty=2,size=0.2)+geom_bar(stat="identity",width=0.7)+nice_theme+scale_fill_manual(values=c(sub_2_col,"dark grey", meta_col,bayes_col),guide=F)+labs(x="",y="p(model)")+ theme(axis.text.x = element_text(angle = 45, hjust = 1))+annotate("text",x=2.4,y=0.99,label=expression(paste(italic("protected p")["exc"], " > 0.99")),size=1.8,color="blue")+annotate("text",x=1.99,y=0.89,label=expression(paste("BOR",phantom()%~~%phantom(), 1.7%*%10^{-4})),size=1.8)+coord_cartesian(ylim=c(0,1))
dev.off()

d_plot_i <- {}
for(i in unique(dfit$id)){
  AIC_i <- with(dfit[dfit$id==i,], tapply(AIC, model, mean))[1:4]
  AIC_diff_i <- AIC_i - AIC_i[which(names(AIC_i)=="sub2")]
  d_i <- data.frame(AIC=AIC_i,AIC_diff=AIC_diff_i,model=c("Bayesian","biased-Bayesian","fixed bias","discrete"),id=i,cl= factor(c(1,2, 3, 4)))
  d_plot_i <- rbind(d_plot_i, d_i)
}
str(d_plot_i)

library(gridExtra)

d2 <- d_plot_i[which(d_plot_i$model=="Bayesian"),]
d2$col_d <- ifelse(d2$AIC_diff>=0,sub_2_col,bayes_col)
# 按AIC_diff排序并创建排序后的id因子
d2$id_ordered <- reorder(d2$id, d2$AIC_diff)
p0 <- ggplot(d2,aes(x=id_ordered,y=AIC_diff,fill=col_d))+
  geom_bar(stat="identity")+
  geom_text(aes(label=id, y=ifelse(AIC_diff>=0, AIC_diff+5, AIC_diff-5)), 
            size=2, angle=90, hjust=ifelse(d2$AIC_diff>=0, 0, 1))+
  scale_fill_manual(values=c(sub_2_col,bayes_col),guide=F)+
  nice_theme+
  scale_x_discrete(labels=NULL,name="Participant")+
  labs(y="AIC difference")+
  coord_cartesian(ylim=c(-ceiling(max(d2$AIC_diff))/2.5,ceiling(max(d2$AIC_diff))*1.2))+
  annotate("text",x=3,hjust="left",y=max(d2$AIC_diff)*0.9,label="discrete better",color=sub_2_col,size=2.3)+
  annotate("text",x=3,hjust="left",y=-max(d2$AIC_diff)/5,label="Bayesian better",color=bayes_col,size=2.3)


d2 <- d_plot_i[which(d_plot_i$model=="biased-Bayesian"),]
d2$col_d <- ifelse(d2$AIC_diff>=0,sub_2_col,meta_col)
d2$id_ordered <- reorder(d2$id, d2$AIC_diff)
p1 <- ggplot(d2,aes(x=id_ordered,y=AIC_diff,fill=col_d))+
  geom_bar(stat="identity")+
  geom_text(aes(label=id, y=ifelse(AIC_diff>=0, AIC_diff+5, AIC_diff-5)), 
            size=2, angle=90, hjust=ifelse(d2$AIC_diff>=0, 0, 1))+
  scale_fill_manual(values=c(sub_2_col,meta_col),guide=F)+
  nice_theme+
  scale_x_discrete(labels=NULL,name=" ")+
  labs(y=" ")+
  coord_cartesian(ylim=c(-ceiling(max(d2$AIC_diff))/2.5,ceiling(max(d2$AIC_diff))*1.2))+
  annotate("text",x=3,hjust="left",y=max(d2$AIC_diff)*0.9,label="discrete better",color=sub_2_col,size=2.3)+
  annotate("text",x=3,hjust="left",y=-max(d2$AIC_diff)/5,label="biased-Bayesian better",color=meta_col,size=2.3)


d2 <- d_plot_i[which(d_plot_i$model=="fixed bias"),]
d2$col_d <- ifelse(d2$AIC_diff>=0,sub_2_col,meta_col)
d2$id_ordered <- reorder(d2$id, d2$AIC_diff)
p2 <- ggplot(d2,aes(x=id_ordered,y=AIC_diff,fill=col_d))+
  geom_bar(stat="identity")+
  geom_text(aes(label=id, y=ifelse(AIC_diff>=0, AIC_diff+5, AIC_diff-5)), 
            size=2, angle=90, hjust=ifelse(d2$AIC_diff>=0, 0, 1))+
  scale_fill_manual(values=c(sub_2_col,"dark grey"),guide=F)+
  nice_theme+
  scale_x_discrete(labels=NULL,name=" ")+
  labs(y=" ")+
  coord_cartesian(ylim=c(-ceiling(max(d2$AIC_diff))/2.5,ceiling(max(d2$AIC_diff))*1.2))+
  annotate("text",x=3,hjust="left",y=max(d2$AIC_diff)*0.9,label="discrete better",color=sub_2_col,size=2.3)+
  annotate("text",x=3,hjust="left",y=-max(d2$AIC_diff)/5,label="fixed bias better",color="dark grey",size=2.3)

pdf("D:/xiangmu/figures/behavior_for_behavior/AIC_iindividual.pdf",height=2.5,width=6.67)
grid.arrange(p0,p1,p2,ncol=3)
dev.off()
