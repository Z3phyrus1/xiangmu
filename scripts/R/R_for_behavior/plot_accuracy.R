# ------------------------------------------------------------------ #
# This script plot the raw data (prop. correct)
# Matteo Lisi, 2019
# ------------------------------------------------------------------ #

rm(list=ls())
setwd("D:/xiangmu/scripts/R")
source("./R/code_all_models.R")

# load data
d <- read.table("D:/xiangmu/data/behavior/idsess/data_nu.txt",header=T, sep=";")

d$condition <- ifelse(d$dual==1,"dual-decision","control")
nrow(d[d$condition=="control",])
nrow(d[d$condition=="dual-decision",])

# plotting libraries and settings
library(ggplot2)
library(mlisi)
nice_theme <- theme_bw()+theme(text=element_text(family="Helvetica",size=9),panel.border=element_blank(),strip.background = element_rect(fill="white",color="white",size=0),strip.text=element_text(size=rel(0.8)),panel.grid.major.x=element_blank(),panel.grid.major.y=element_blank(),panel.grid.minor=element_blank(),axis.line.x=element_line(size=.4),axis.line.y=element_line(size=.4),axis.text.x=element_text(size=7,color="black"),axis.text.y=element_text(size=7,color="black"),axis.line=element_line(size=.4), axis.ticks=element_line(color="black"))

# ------------------------------------------------------#
# plot
d$abs_s <- abs(d$s)
d$decision <- ifelse(d$decision==1, "1st", "2nd")

# binning
cut_points <- c(0, 0.5, 1, 1.5, 2, 20)
d$bin_s2 <- cut(d$abs_s, breaks=cut_points)

x_breaks <- c(0, 0.5, 1, 1.5, 2)
x_labels <- c("0", "0.5", "1", "1.5", "2") # just to avoid unnecessary double-digits in axis labels

dag1 <- aggregate(cbind(acc,abs_s) ~ condition + decision + bin_s2 + task + id, d[d$abs_s<2.5,], mean)
dag2 <- aggregate(cbind(acc,abs_s) ~ condition + decision + bin_s2 + task, dag1, mean)
dag2$acc_se <- aggregate(acc ~ condition + decision + bin_s2 + task, dag1, bootMeanSE)$acc

dag2$decision <- factor(dag2$decision)
dag2$condition <- as.factor(dag2$condition)

# fit probit regression on pooled data
d$decision <- factor(d$decision)
m0 <- glm(acc ~ abs_s*condition*decision*task, d, family=binomial(probit))
nd <- expand.grid(abs_s=seq(0,2.99,length.out=100),condition=unique(d$condition),task=unique(d$task),decision=unique(d$decision), KEEP.OUT.ATTRS = F)
nd$decision <- factor(nd$decision)
nd$acc <- predict(m0, newdata=nd, type="response")
nd$acc_se <- NA

pdf("D:/xiangmu/figures/behavior_idsess/accuracy_pooled.pdf",width=2.9,height=2.8)
ggplot(dag2, aes(x=abs_s, y=acc, ymin=acc-acc_se, ymax=acc+acc_se, color=decision,group=decision))+geom_line(data=nd,aes(x=abs_s, y=acc),size=0.8)+geom_errorbar(width=0)+geom_point(pch=21,size=1.3)+facet_grid(condition~task,scales="free")+coord_cartesian(xlim=c(0,2.23),ylim=c(0.5,1))+nice_theme+labs(y="p(correct)",x=expression(paste("discriminability [",sigma,"]")))+theme(legend.justification = c(1, 0), legend.position = c(1, 0),legend.background = element_rect(fill="transparent"))+scale_color_manual(values=c("dark grey","black"),guide=guide_legend(title=NULL,keyheight=0.8,label.theme = element_text(size=7,angle=0)))+scale_x_continuous(breaks=x_breaks, labels=x_labels)
dev.off()

