# ------------------------------------------------------------------ #
# This script plots optimal model predictions together with the data
# Matteo Lisi, 2020
# ------------------------------------------------------------------ #

rm(list=ls())
setwd("D:/Lisi")
source("D:/11.13/nhb_data_code/R/code_all_models.R")

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

# general plot parameters (keep those from plot split by D1 accuracy)
n_bin <- 9
x_coord_max <- 2.45
y_coord_max_c1 <- 0.18
fig_height <- 1.8
fig_width <- 1.5
sizedataline <- 0.4
typedatapoint <- 21

x_breaks <- c(0, 0.5, 1, 1.5, 2)
x_labels <- c("0", "0.5", "1", "1.5", "2") # just to avoid unnecessary double-digits and reduce cluttering

# select correct first decision only
d <- read.table(file="D:/reysy/Experiment/test/data_wide_wmPred.txt", sep=";", header=T)

# select what to plot in error bars/bands
m_ci <- function(v,alpha=0.05){
  ci <- bootMeanSE(v) # plot bootstrapped SEM
  return(ci)
}

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

# do plot
model_plot <- "Optimal"
d_plot <- with(d_m, data.frame(bin_s1=NA, s1=abs1, r2=p_r2_optimal, r2_se=p_r2_optimal_se, r2_naive=p_r2_naive,r2_naive_se=p_r2_naive_se, r2_optimal = p_r2_optimal))
d_plot$model <- model_plot
plo_opti <- ggplot(d_plot,aes(x=s1, y=r2, ymin=r2-r2_se,ymax=r2+r2_se))+nice_theme+geom_ribbon(alpha=0.3,size=0,fill=bayes_col)+geom_line(size=0.6,color=bayes_col)+geom_point(data=dag2,pch=typedatapoint,color="black")+geom_line(data=dag2,size=sizedataline,color="black",lty=1)+geom_errorbar(data=dag2,width=0,color="black",size=0.4)+coord_cartesian(xlim=c(0,x_coord_max),ylim=c(0.5,1))+ggtitle(model_plot)+labs(x =expression(paste("|",S[1],"| [",sigma,"]") ), y="p(choose 'right' option)")+theme(plot.title = element_text(size = 8))+scale_x_continuous(breaks=x_breaks, labels=x_labels)+geom_line(aes(x=s1,y=r2_naive),size=0.4,color="dark grey")+geom_line(aes(x=s1,y=r2_naive-r2_naive_se),size=0.3,color="dark grey",lty=2)+geom_line(aes(x=s1,y=r2_naive+r2_naive_se),size=0.3,color="dark grey",lty=2)
pdf("D:/reysy/Experiment/fig/optimal_430.pdf",width=2,height=2)
plo_opti
dev.off()

