# ------------------------------------------------------------------ #
# This script plots the data together with model predictions
# (split by correctness of the first response)
# Matteo Lisi, 2020
# ------------------------------------------------------------------ #

rm(list=ls())
setwd("D:/xiangmu/scripts/R")
source("./R/code_all_models.R")

# ------------------------------------------------------#
# some plotting libraries and settings
sub_2_col <- rgb(80,80,255,maxColorValue=255)
bayes_col <- rgb(255,80,80,maxColorValue=255)
opti_col <- "black"
library(ggplot2)
library(viridis)
library(mlisi)
nice_theme <- theme_bw()+theme(text=element_text(family="Helvetica",size=9),panel.border=element_blank(),strip.background = element_rect(fill="white",color="white",size=0),strip.text=element_text(size=rel(0.8)),panel.grid.major.x=element_blank(),panel.grid.major.y=element_blank(),panel.grid.minor=element_blank(),axis.line.x=element_line(size=.4),axis.line.y=element_line(size=.4),axis.text.x=element_text(size=7,color="black"),axis.text.y=element_text(size=7,color="black"),axis.line=element_line(size=.4), axis.ticks=element_line(color="black"))

# ------------------------------------------------------#
# do plots starting with correct first decision

# general plot parameters
n_bin <- 9
x_coord_max <- 2.45
y_coord_max_c1 <- 0.25#0.18
fig_height <- 1.8
fig_width <- 1.5
sizedataline <- NA
typedatapoint <- 21

x_breaks <- c(0, 0.5, 1, 1.5, 2)
x_labels <- c("0", "0.5", "1", "1.5", "2") # just to avoid unnecessary double-digits and reduce cluttering

# select correct first decision only
d <- read.table(file="D:/xiangmu/data/behavior_for_behavior/data_wide_wmPred.txt", sep=";", header=T)
d <- d[d$acc1==1,]

# transform as differences from baseline
d$p_r2_discrete <- d$p_r2_discrete - d$p_r2_naive
d$p_r2_bayesian <- d$p_r2_bayesian - d$p_r2_naive
d$p_r2_optimal <- d$p_r2_optimal - d$p_r2_naive
d$p_r2_fixed <- d$p_r2_fixed - d$p_r2_naive
d$r2 <- d$r2 - d$p_r2_naive

# choose what to plot in error bars/bands
m_ci <- function(v,alpha=0.05){
  # ci <- se(v) # SEM
  ci <- bootMeanSE(v) # bootstrapped SEM
  return(ci)
}

# Bin data
cut_points <- c(0, 0.5, 1, 1.5, 2, 20)
d$bin_s1 <- cut(d$abs1, breaks=cut_points) # this is for observed data
d$x_s1 <- cut(d$abs1, breaks=cut_points) # model

# aggregate for plotting
d_m_1 <- aggregate(cbind(p_r2_discrete, p_r2_bayesian,  p_r2_fixed, abs1, p_r2_naive,p_r2_optimal) ~ x_s1 + id, d, mean)
d_m <- aggregate(cbind(p_r2_discrete, p_r2_bayesian,  p_r2_fixed, abs1, p_r2_naive,p_r2_optimal) ~ x_s1, d_m_1, mean)

d_m$p_r2_discrete_se <- aggregate(p_r2_discrete ~ x_s1, d_m_1, m_ci)$p_r2_discrete
d_m$p_r2_bayesian_se <- aggregate(p_r2_bayesian ~ x_s1, d_m_1, m_ci)$p_r2_bayesian
d_m$p_r2_fixed_se <- aggregate(p_r2_fixed ~ x_s1, d_m_1, m_ci)$p_r2_fixed
d_m$p_r2_naive_se <- aggregate(p_r2_naive ~ x_s1, d_m_1, m_ci)$p_r2_naive
d_m$p_r2_optimal_se <- aggregate(p_r2_optimal ~ x_s1, d_m_1, m_ci)$p_r2_optimal

dag <- aggregate(cbind(abs1, r2, p_r2_naive) ~ bin_s1 + id, d, mean)
# dag$r2 <- dag$r2 - dag$p_r2_naive
dag2 <- aggregate(cbind(abs1,  r2) ~ bin_s1 , dag, mean)
dag2$r2_se <- aggregate(r2 ~  bin_s1, dag, bootMeanSE)$r2
 
dag2$model <-"observed\ndata"
colnames(dag2) <- c("bin_s1", "s1", "r2", "r2_se","model" )
dag2$r2_naive <- NA
dag2$r2_naive_se <- NA

color_data <- "black"

model_plot <- "After correct\nfirst decision"
d_plot <- with(d_m, data.frame(bin_s1=NA, s1=abs1, r2=p_r2_optimal, r2_se=p_r2_optimal_se, r2_naive=p_r2_naive,r2_naive_se=p_r2_naive_se, r2_optimal = p_r2_optimal))
d_plot$model <- model_plot
plo_opti <- ggplot(d_plot,aes(x=s1, y=r2, ymin=r2-r2_se,ymax=r2+r2_se))+nice_theme+geom_hline(yintercept=0,color="dark grey")+geom_ribbon(alpha=0.3,size=0,fill=bayes_col)+geom_line(size=0.6,color=bayes_col)+geom_point(data=dag2,pch=typedatapoint,color=color_data)+geom_line(data=dag2,size=sizedataline,color=color_data)+geom_errorbar(data=dag2,width=0,color=color_data,size=0.4)+coord_cartesian(xlim=c(0,x_coord_max),ylim=c(0,y_coord_max_c1))+ggtitle(model_plot)+labs(x =expression(paste("|",S[1],"| [",sigma,"]") ), y="p(choose 'right' option)\n difference from baseline")+theme(plot.title = element_text(size = 8))+scale_x_continuous(breaks=x_breaks, labels=x_labels)
pdf("D:/xiangmu/figures/behavior_for_behavior/optimal_correctC1.pdf",width=fig_width,height=fig_height)
plo_opti
dev.off()
#+geom_line(data=dag2,size=0.4)

model_plot <- "After correct\nfirst decision"
d_plot <- with(d_m, data.frame(bin_s1=NA, s1=abs1, r2=p_r2_bayesian, r2_se=p_r2_bayesian_se, r2_naive=p_r2_naive,r2_naive_se=p_r2_naive_se, r2_optimal = p_r2_optimal))
d_plot$model <- model_plot
plo1 <- ggplot(d_plot,aes(x=s1, y=r2, ymin=r2-r2_se,ymax=r2+r2_se))+nice_theme+geom_hline(yintercept=0,color="dark grey")+geom_ribbon(alpha=0.3,size=0,fill=bayes_col)+geom_line(size=0.6,color=bayes_col)+geom_point(data=dag2,pch=typedatapoint,color=color_data)+geom_line(data=dag2,size=sizedataline,color=color_data)+geom_errorbar(data=dag2,width=0,color=color_data,size=0.4)+coord_cartesian(xlim=c(0,x_coord_max),ylim=c(0,y_coord_max_c1))+ggtitle(model_plot)+theme(plot.title = element_text(size = 8))+labs(x =expression(paste("|",S[1],"| [",sigma,"]") ), y="p(choose 'right' option)\n difference from baseline")+theme(plot.title = element_text(size = 8))+geom_line(aes(y=r2_optimal),size=0.4,color="red",lty=2)+scale_x_continuous(breaks=x_breaks, labels=x_labels)
# pdf(paste("D:/11.13/nhb_data_code/fig/",model_plot,"_r2_ACC11.pdf",sep=""),width=fig_width,height=fig_height)
# plo1
# dev.off()

model_plot <- " \n "
d_plot <- with(d_m, data.frame(bin_s1=NA, s1=abs1, r2=p_r2_discrete, r2_se=p_r2_discrete_se, r2_optimal = p_r2_optimal))
d_plot$model <- model_plot
plo2 <- ggplot(d_plot,aes(x=s1, y=r2,ymin=r2-r2_se,ymax=r2+r2_se))+nice_theme+geom_hline(yintercept=0,color="dark grey")+geom_ribbon(alpha=0.3,size=0,fill=sub_2_col)+geom_line(size=0.6,color=sub_2_col)+geom_point(data=dag2,pch=typedatapoint,color=color_data)+geom_line(data=dag2,size=sizedataline,color=color_data)+geom_errorbar(data=dag2,width=0,color=color_data,size=0.4)+coord_cartesian(xlim=c(0,x_coord_max),ylim=c(0,y_coord_max_c1))+ggtitle(model_plot)+theme(plot.title = element_text(size = 8))+labs(x =expression(paste("|",S[1],"| [",sigma,"]") ), y="p(choose 'right' option)\n difference from baseline")+geom_line(aes(y=r2_optimal),size=0.4,color="red",lty=2)+scale_x_continuous(breaks=x_breaks, labels=x_labels)
# pdf(paste("D:/11.13/nhb_data_code/fig/",model_plot,"_r2_ACC11.pdf",sep=""),width=fig_width,height=fig_height)
# plo2
# dev.off()

model_plot <- " \n "
d_plot <- with(d_m, data.frame(bin_s1=NA, s1=abs1, r2=p_r2_fixed, r2_se=p_r2_fixed_se, r2_optimal = p_r2_optimal))
d_plot$model <- model_plot
plo3 <- ggplot(d_plot,aes(x=s1, y=r2, ymin=r2-r2_se,ymax=r2+r2_se))+nice_theme+geom_hline(yintercept=0,color="dark grey")+geom_ribbon(alpha=0.3,size=0,fill="dark grey")+geom_line(size=0.6,color="dark grey")+geom_point(data=dag2,pch=typedatapoint,color=color_data)+geom_line(data=dag2,size=sizedataline,color=color_data)+geom_errorbar(data=dag2,width=0,color=color_data,size=0.4)+coord_cartesian(xlim=c(0,x_coord_max),ylim=c(0,y_coord_max_c1))+ggtitle(model_plot)+theme(plot.title = element_text(size = 8))+labs(x =expression(paste("|",S[1],"| [",sigma,"]") ), y="p(choose 'right' option)\n difference from baseline")+geom_line(aes(y=r2_optimal),size=0.4,color="red",lty=2)+scale_x_continuous(breaks=x_breaks, labels=x_labels)
# pdf(paste("D:/11.13/nhb_data_code/fig/",model_plot,"_r2_ACC11.pdf",sep=""),width=fig_width,height=fig_height)
# plo3 
# dev.off()


# ------------------------------------------------------ #
# ------------------------------------------------------ #
# Now repeat the same but conditioned on wrong first decision

d <- read.table(file="D:/xiangmu/data/behavior_for_behavior/data_wide_wmPred.txt", sep=";", header=T)
d <- d[d$acc1==0,]

# plot settings
#x_coord_max <- 1.7
y_coord_max_c0 <- 0.3#0.25
# bin width is 0.5 sigmas
cut_points <- c(0, 0.5, 1, 1.5, 20)
d$bin_s1 <- cut(d$abs1, breaks=cut_points) # this is for observed data
d$x_s1 <- cut(d$abs1, breaks=cut_points) # model


# transform as differences
d$p_r2_discrete <- d$p_r2_discrete - d$p_r2_naive
d$p_r2_bayesian <- d$p_r2_bayesian - d$p_r2_naive
d$p_r2_optimal <- d$p_r2_optimal - d$p_r2_naive
d$p_r2_fixed <- d$p_r2_fixed - d$p_r2_naive
d$r2 <- d$r2 - d$p_r2_naive

d_m_1 <- aggregate(cbind(p_r2_discrete, p_r2_bayesian,  p_r2_fixed, abs1, p_r2_naive, p_r2_optimal) ~ x_s1 + id, d, mean)
d_m <- aggregate(cbind(p_r2_discrete, p_r2_bayesian,  p_r2_fixed, abs1, p_r2_naive, p_r2_optimal) ~ x_s1, d_m_1, mean)

d_m$p_r2_discrete_se <- aggregate(p_r2_discrete ~ x_s1, d_m_1, m_ci)$p_r2_discrete
d_m$p_r2_bayesian_se <- aggregate(p_r2_bayesian ~ x_s1, d_m_1, m_ci)$p_r2_bayesian
d_m$p_r2_fixed_se <- aggregate(p_r2_fixed ~ x_s1, d_m_1, m_ci)$p_r2_fixed
d_m$p_r2_naive_se <- aggregate(p_r2_naive ~ x_s1, d_m_1, m_ci)$p_r2_naive
d_m$p_r2_optimal_se <- aggregate(p_r2_optimal ~ x_s1, d_m_1, m_ci)$p_r2_optimal

dag <- aggregate(cbind(abs1, r2) ~ bin_s1 + id, d, mean)
dag2 <- aggregate(cbind(abs1,  r2) ~ bin_s1 , dag, mean)
dag2$r2_se <- aggregate(r2 ~  bin_s1, dag, bootMeanSE)$r2
dag2$model <-"observed\ndata"
colnames(dag2) <- c("bin_s1", "s1", "r2", "r2_se","model" )
dag2$r2_naive <- NA
dag2$r2_naive_se <- NA

color_wrong <- "black" # rgb(0.8,0,0)


model_plot <- "After wrong\nfirst decision"
d_plot <- with(d_m, data.frame(bin_s1=NA, s1=abs1, r2=p_r2_optimal, r2_se=p_r2_optimal_se, r2_naive=p_r2_naive,r2_naive_se=p_r2_naive_se, r2_optimal = p_r2_optimal))
d_plot$model <- model_plot
plo_opti <- ggplot(d_plot,aes(x=s1, y=r2, ymin=r2-r2_se,ymax=r2+r2_se))+nice_theme+geom_hline(yintercept=0,color="dark grey")+geom_ribbon(alpha=0.3,size=0,fill=bayes_col)+geom_line(size=0.6,color=bayes_col)+geom_point(data=dag2,pch=typedatapoint,color=color_data)+geom_line(data=dag2,size=sizedataline,color=color_data)+geom_errorbar(data=dag2,width=0,color=color_data,size=0.4)+coord_cartesian(xlim=c(0,x_coord_max),ylim=c(0,y_coord_max_c0))+ggtitle(model_plot)+labs(x =expression(paste("|",S[1],"| [",sigma,"]") ), y="p(choose 'right' option)\n difference from baseline")+theme(plot.title = element_text(size = 8))+scale_x_continuous(breaks=x_breaks, labels=x_labels)
pdf("D:/xiangmu/figures/behavior_for_behavior/optimal_wrongD1.pdf",width=fig_width,height=fig_height)
plo_opti
dev.off()

model_plot <- " \n "
d_plot <- with(d_m, data.frame(bin_s1=NA, s1=abs1, r2=p_r2_discrete, r2_se=p_r2_discrete_se, r2_optimal = p_r2_optimal))
d_plot$model <- model_plot
plo4 <- ggplot(d_plot,aes(x=s1, y=r2,ymin=r2-r2_se,ymax=r2+r2_se))+nice_theme+geom_hline(yintercept=0,color="dark grey")+geom_ribbon(alpha=0.3,size=0,fill=sub_2_col)+geom_line(size=0.6,color=sub_2_col)+geom_point(data=dag2,pch=typedatapoint,color=color_wrong)+geom_line(data=dag2,size=sizedataline,color=color_wrong)+geom_errorbar(data=dag2,width=0,color=color_wrong,size=0.4)+coord_cartesian(xlim=c(0,x_coord_max),ylim=c(0,y_coord_max_c0))+ggtitle(model_plot)+theme(plot.title = element_text(size = 8))+labs(x =expression(paste("|",S[1],"| [",sigma,"]") ), y=" \n ")+geom_line(aes(y=r2_optimal),size=0.4,color="red",lty=2)+scale_x_continuous(breaks=x_breaks, labels=x_labels)
# pdf(paste("D:/11.13/nhb_data_code/R/fig/",model_plot,"_r2_ACC10.pdf",sep=""),width=fig_width,height=fig_height)
# plo4
# dev.off()

model_plot <- "After wrong\nfirst decision"
d_plot <- with(d_m, data.frame(bin_s1=NA, s1=abs1, r2=p_r2_bayesian, r2_se=p_r2_bayesian_se, r2_naive=p_r2_naive,r2_naive_se=p_r2_naive_se, r2_optimal = p_r2_optimal))
d_plot$model <- model_plot
plo5 <- ggplot(d_plot,aes(x=s1, y=r2, ymin=r2-r2_se,ymax=r2+r2_se))+nice_theme+geom_hline(yintercept=0,color="dark grey")+geom_ribbon(alpha=0.3,size=0,fill=bayes_col)+geom_line(size=0.6,color=bayes_col)+geom_point(data=dag2,pch=typedatapoint,color=color_wrong)+geom_line(data=dag2,size=sizedataline,color=color_wrong)+geom_errorbar(data=dag2,width=0,color=color_wrong,size=0.4)+coord_cartesian(xlim=c(0,x_coord_max),ylim=c(0,y_coord_max_c0))+ggtitle(model_plot)+theme(plot.title = element_text(size = 8))+labs(x =expression(paste("|",S[1],"| [",sigma,"]") ), y=" \n ")+geom_line(aes(y=r2_optimal),size=0.4,color="red",lty=2)+scale_x_continuous(breaks=x_breaks, labels=x_labels)
# pdf(paste("D:/11.13/nhb_data_code/R/fig/",model_plot,"_r2_ACC10.pdf",sep=""),width=fig_width,height=fig_height)
# plo5
# dev.off()

model_plot <- " \n "
d_plot <- with(d_m, data.frame(bin_s1=NA, s1=abs1, r2=p_r2_fixed, r2_se=p_r2_fixed_se, r2_optimal = p_r2_optimal))
d_plot$model <- model_plot
plo6 <- ggplot(d_plot,aes(x=s1, y=r2, ymin=r2-r2_se,ymax=r2+r2_se))+nice_theme+geom_hline(yintercept=0,color="dark grey")+geom_ribbon(alpha=0.3,size=0,fill="dark grey")+geom_line(size=0.6,color="dark grey")+geom_point(data=dag2,pch=typedatapoint,color=color_wrong)+geom_line(data=dag2,size=sizedataline,color=color_wrong)+geom_errorbar(data=dag2,width=0,color=color_wrong,size=0.4)+coord_cartesian(xlim=c(0,x_coord_max),ylim=c(0,y_coord_max_c0))+ggtitle(model_plot)+theme(plot.title = element_text(size = 8))+labs(x =expression(paste("|",S[1],"| [",sigma,"]") ), y=" \n ")+geom_line(aes(y=r2_optimal),size=0.4,color="red",lty=2)+scale_x_continuous(breaks=x_breaks, labels=x_labels)
# pdf(paste("./fig/",model_plot,"_r2_ACC10.pdf",sep=""),width=fig_width,height=fig_height)
# plo6
# dev.off()

# make big plot
library(ggpubr)
pdf("D:/xiangmu/figures/behavior_for_behavior/pred_all.pdf",width=fig_width*2,height=fig_height*3)
#ggarrange(plo1, plo5, plo2, plo4, plo3, plo6,
#          ncol = 2, nrow = 3)
ggarrange(plo5, plo1, plo4, plo2, plo6, plo3,
          ncol = 2, nrow = 3)
dev.off()

