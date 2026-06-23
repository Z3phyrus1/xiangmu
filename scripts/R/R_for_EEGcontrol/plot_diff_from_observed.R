# ------------------------------------------------------------------ #
# This script plot the difference from observed behavior in the 
# proportion of choices 'right' at the second decision, for the 
# discrete and biased-Bayesian observer, then make a plot of individual 
# differences
#
# Matteo Lisi, 2020
# ------------------------------------------------------------------ #

# clear
rm(list=ls())
setwd("D:/Lisi")

# ------------------------------------------------------#
# library(ggpubr)
# plotting libraries and color settings
sub_2_col <- rgb(80,80,255,maxColorValue=255)
bayes_col <- rgb(255,150,0,maxColorValue=255)
opti_col <- "black"
library(ggplot2)
library(viridis)
library(mlisi)
nice_theme <- theme_bw()+theme(text=element_text(family="Helvetica",size=9),panel.border=element_blank(),strip.background = element_rect(fill="white",color="white",size=0),strip.text=element_text(size=rel(0.8)),panel.grid.major.x=element_blank(),panel.grid.major.y=element_blank(),panel.grid.minor=element_blank(),axis.line.x=element_line(size=.4),axis.line.y=element_line(size=.4),axis.text.x=element_text(size=7,color="black"),axis.text.y=element_text(size=7,color="black"),axis.line=element_line(size=.4), axis.ticks=element_line(color="black"))

# ------------------------------------------------------#
# load data
d <- read.table("D:/reysy/dual-decision-task-master/dual-decision-task-master/data/data_wide_wmPred.txt",header=T, sep=";")

# ------------------------------------------------------#
# plot 

# calculate differences from model predictions
d$d_r2_optimal <- d$r2 - d$p_r2_optimal
d$d_r2_discrete <- d$r2 - d$p_r2_discrete
d$d_r2_bayesian <- d$r2 - d$p_r2_bayesian

# calculate average and standard errors
dag0 <- aggregate(cbind(d_r2_optimal,d_r2_discrete ,d_r2_bayesian) ~ acc1 +id, d, mean)
dag0$opti_se <- aggregate(d_r2_optimal ~ acc1 + id, d, bootMeanSE)$d_r2_optimal
dag0$discr_se <- aggregate(d_r2_discrete ~ acc1 +  id, d, bootMeanSE)$d_r2_discrete
dag0$bayes_se <- aggregate(d_r2_bayesian ~ acc1 +  id, d, bootMeanSE)$d_r2_bayesian

dag0$discr_lb <- dag0$d_r2_discrete - dag0$discr_se
dag0$discr_ub <- dag0$d_r2_discrete + dag0$discr_se

dag0$bayes_lb <- dag0$d_r2_bayesian - dag0$bayes_se
dag0$bayes_ub <- dag0$d_r2_bayesian + dag0$bayes_se

dag0$discrete_better <- ifelse(abs(dag0$d_r2_discrete)<=abs(dag0$d_r2_bayesian),1,0)
dag0$discrete_better_pl <- ifelse(dag0$discrete_better, -0.2,NA)

dag0$ordering <- dag0$d_r2_discrete

pl1 <- (ggplot(dag0[dag0$acc1==0,], aes(x=reorder(id, ordering)))
        +geom_hline(yintercept=0,lty=2)
        +geom_errorbar(aes(ymin=bayes_lb,ymax=bayes_ub),width=0,size=3,col=bayes_col,alpha=0.3)
        +geom_point(aes(y=d_r2_bayesian),color=bayes_col,pch="-",size=6)
        +geom_errorbar(aes(ymin=discr_lb,ymax=discr_ub),width=0,size=3,col=sub_2_col,alpha=0.3)
        +geom_point(aes(y=d_r2_discrete),color=sub_2_col,pch="-",size=6)
        +geom_point(aes(y=discrete_better_pl),color=sub_2_col,pch=19,size=1.8)
        +scale_x_discrete(labels=NULL)
        + scale_y_continuous(limits=c(-0.2,0.29))
        +nice_theme+labs(x ="participant", y="p('right'), difference from observed behaviour")
        +ggtitle(label="Wrong first decision"))

pl2 <- (ggplot(dag0[dag0$acc1==1,], aes(x=reorder(id, ordering)))
        +geom_hline(yintercept=0,lty=2)
        +geom_errorbar(aes(ymin=bayes_lb,ymax=bayes_ub),width=0,size=3,col=bayes_col,alpha=0.3)
        +geom_point(aes(y=d_r2_bayesian),color=bayes_col,pch="-",size=6)
        +geom_errorbar(aes(ymin=discr_lb,ymax=discr_ub),width=0,size=3,col=sub_2_col,alpha=0.3)
        +geom_point(aes(y=d_r2_discrete),color=sub_2_col,pch="-",size=6)
        +geom_point(aes(y=discrete_better_pl),color=sub_2_col,pch=19,size=1.8)
        +scale_x_discrete(labels=NULL)
        + scale_y_continuous(limits=c(-0.2,0.29))
        +nice_theme+labs(x ="participant", y="p('right'), difference from observed behaviour")
        +ggtitle(label="Correct first decision"))


# make big plot
library(ggpubr)
pdf("D:/reysy/dual-decision-task-master/dual-decision-task-master/fig/pred_diff_individual.pdf",width=6.5,height=5)
ggarrange(pl1, pl2,
          ncol = 2, nrow = 1,
          labels=c("A","B"))
dev.off()

