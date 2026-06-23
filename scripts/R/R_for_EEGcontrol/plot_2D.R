# ------------------------------------------------------------------ #
# This script plot predicted and observed performance as a 2D surface
# Matteo Lisi, 2020
# ------------------------------------------------------------------ #

# check frequencies of high confidence errors conditione on the all and second decisions.
rm(list=ls())
setwd("D:/11.13")
library(mlisi)
source("D:/11.13/nhb_data_code/R/code_all_models.R")
source("D:/11.13/nhb_data_code/R/additional_optimal_policy_d2.R")
source("D:/11.13/nhb_data_code/simulate_sampling_model.R")
library(viridis)

library(directlabels)

d <- read.table(file="D:/xiangmu/data/behavior_for_behavior/data_wide_wmPred.txt", sep=";", header=T)
dfit <- read.table("D:/xiangmu/data/behavior_for_behavior/fit_par.txt",header=T)

# bound for plot surface
s_bound <-  2.5

# get the median estimated parameters
mnoise <- median(dfit$m[dfit$model=="metanoise"])
sub1_1 <- median(dfit$fixed_shift[dfit$model=="sub1"])
sub2_1 <- median(dfit$w1[dfit$model=="sub2"])
sub2_2 <- median(dfit$theta2[dfit$model=="sub2"])

# # optimal sub 2 parameters ?
#opti_sub2 <- optimal_policy_d2(s_bound)（疑似没用）

# ------------------------------------------------------------------ #
# make new data frame for computing model predictions
nd <- expand.grid(s1=seq(0,s_bound, length.out=100), s2=seq(0,s_bound, length.out=100), KEEP.OUT.ATTRS = F)

# ------------------------------------------------------------------ #
# function to compute the expected proportion of correct responses for all models
p_c2_opti <-function(s1,s2){
  p_c1 <- pnorm(s1)
  p_c2_c1 <- predict_r2_metanoise(data.frame(s1=s1, r1=1, s2=s2), 1)
  p_c2_w1 <- 1 - predict_r2_metanoise(data.frame(s1=s1, r1=0, s2=-s2), 1)
  p_c2 <- p_c1 * p_c2_c1 + (1-p_c1) * p_c2_w1
  return(p_c2)
}

p_c2_meta <-function(s1,s2, m){
  p_c1 <- pnorm(s1)
  p_c2_c1 <- predict_r2_metanoise(data.frame(s1=s1, r1=1, s2=s2), m)
  p_c2_w1 <- 1 - predict_r2_metanoise(data.frame(s1=s1, r1=0, s2=-s2), m)
  p_c2 <- p_c1 * p_c2_c1 + (1-p_c1) * p_c2_w1
  return(p_c2)
}

p_c2_sub2 <-function(s1,s2, par){
  p_c1 <- pnorm(s1)
  p_c2_c1 <- p_r2_sub2(data.frame(s1=s1, r1=1, s2=s2), par)
  p_c2_w1 <- 1 - p_r2_sub2(data.frame(s1=s1, r1=0, s2=-s2), par)
  p_c2 <- p_c1 * p_c2_c1 + (1-p_c1) * p_c2_w1
  return(p_c2)
}

p_c2_sub1 <-function(s1,s2, par){
  p_c1 <- pnorm(s1)
  p_c2_c1 <- pnorm(s2, mean=par, sd=1)
  p_c2_w1 <- 1 - pnorm(-s2, mean=par, sd=1)
  p_c2 <- p_c1 * p_c2_c1 + (1-p_c1) * p_c2_w1
  return(p_c2)
}

# ------------------------------------------------------------------ #
# now all the functions are ready to compute model predictions
nd$p_opti <- p_c2_opti(s1=nd$s1, s2=nd$s2)
nd$p_sub2 <- p_c2_sub2(s1=nd$s1, s2=nd$s2, c(sub2_1,sub2_2))
nd$p_fixd <- p_c2_sub1(s1=nd$s1, s2=nd$s2, sub1_1)
nd$p_meta <- p_c2_meta(s1=nd$s1, s2=nd$s2, mnoise)

# ------------------------------------------------------------------ #
# plot empirical data
n_bins <- 16
breaks_s_data <- seq(0,s_bound, length.out=n_bins)
d$bin_s1 <- cut(d$abs1, breaks=breaks_s_data)
d$bin_s2 <- cut(d$abs2, breaks=breaks_s_data)

bin_centers <- (breaks_s_data[1:(n_bins-1)]+breaks_s_data[2:n_bins])/2
d$bs1 <- bin_centers[as.numeric(d$bin_s1)]
d$bs2 <- bin_centers[as.numeric(d$bin_s2)]

dag <- aggregate(cbind(acc2, bs1, bs2, abs1, abs2)~bin_s1+ bin_s2, d, mean)

pdf(file="D:/xiangmu/fig/behavior_for_behavior/obsdata_raw_p.pdf",width=2.3,height=1.6)
ggplot(dag, aes(x=bs1, y=bs2, fill=acc2,z=acc2))+geom_raster(interpolate=F)+scale_fill_viridis(limits=c(0.5,1))+nice_theme+coord_equal(xlim=c(0.09,2.40),ylim=c(0.09,2.40))+labs(x=expression(paste("|",s[1],"| [",sigma,"]")), y=expression(paste("|",s[2],"| [",sigma,"]")),fill="p(correct)")+ggtitle("Observed")+theme(plot.title = element_text(size = 8, face = "plain"))+theme(legend.title=element_text(size=7),legend.text=element_text(size=7))+theme(legend.key.size =  unit(0.15, "in"))#+ geom_contour(color="white",bins=2)
dev.off()


# ------------------------------------------------------------------ #
# replot data with Gaussian smoothing
library(fields)
library(reshape2)
out<- as.image( d$acc2, x= data.frame(x=d$abs1,y=d$abs2), nrow=100, ncol=100) 
dx<- out$x[2]- out$x[1] 
dy<-  out$y[2] - out$y[1] 
look<- image.smooth( out$z, dx=dx, dy=dy, theta= .2) 
smooth_mat <- look$z
rownames(smooth_mat) <- look$x
colnames(smooth_mat) <- look$y
longData <- melt(smooth_mat)

contour_breaks <- c(seq(0.5,1,0.1),0.95)
library(metR)

color_map <- "D"
scale_factor <- 1.2

slice_locations <- c(0.75, 1.75)

pdf(file="D:/xiangmu/fig/behavior_for_behavior/obsdata_smooth_2.pdf",width=2.3*scale_factor,height=1.6*scale_factor)
ggplot(longData, aes(x=Var1, y=Var2, fill=value,z=value))+geom_raster(interpolate=F)+scale_fill_viridis(limits=c(0.5,1),option=color_map)+geom_vline(xintercept=slice_locations,lty=2,size=0.4,color="black")+nice_theme+coord_equal(xlim=c(0.09,2.40),ylim=c(0.09,2.40))+labs(x=expression(paste("|",s[1],"| [",sigma,"]")), y=expression(paste("|",s[2],"| [",sigma,"]")),fill="p(correct)")+ggtitle("Observed")+theme(plot.title = element_text(size = 8, face = "plain"))+ geom_contour(color="white",breaks=contour_breaks ,size=0.2)+theme(legend.title=element_text(size=7),legend.text=element_text(size=7))+theme(legend.key.size =  unit(0.15, "in"))+geom_text_contour(breaks=contour_breaks,rotate=T,color="white",size=1.7,nudge_y = 0.07,nudge_x = 0.07,stroke.color = rgb(0.4,0.4,0.4),stroke=0.1)
dev.off()

# ------------------------------------------------------------------ #
## now plot the Bayesian model
bayes_col <- rgb(255,80,80,maxColorValue=255)
pdf(file="D:D:/xiangmu/fig/behavior_for_behavior/opti_p.pdf",width=2.3*scale_factor,height=1.6*scale_factor)
ggplot(nd, aes(x=s1, y=s2, z = p_opti, fill=p_opti))+geom_raster(interpolate=T)+scale_fill_viridis(limits=c(0.5,1),option=color_map)+geom_vline(xintercept=slice_locations,lty=2,size=0.4,color=bayes_col)+nice_theme+coord_equal(xlim=c(0.09,2.40),ylim=c(0.09,2.40))+labs(x=expression(paste("|",s[1],"| [",sigma,"]")), y=expression(paste("|",s[2],"| [",sigma,"]")),fill="p(correct)")+ggtitle("Optimal") + theme(plot.title = element_text(size = 8, face = "plain")) +geom_contour(color="white",breaks = contour_breaks,size=0.2)+geom_text_contour(breaks=contour_breaks,rotate=T,color="white",size=1.7,nudge_y = 0.07,nudge_x = 0.07,stroke.color = rgb(0.4,0.4,0.4),stroke=0.1)+theme(legend.title=element_text(size=7),legend.text=element_text(size=7))+theme(legend.key.size =  unit(0.15, "in"))
dev.off()

# ------------------------------------------------------------------ #
## here plot the 2 cross-sections of the 2D surface
str(longData)
diff_slice1 <- abs(longData$Var1 - slice_locations[1])
diff_slice2 <- abs(longData$Var1 - slice_locations[2])
index_1_obs <- which(diff_slice1==min(diff_slice1))
index_2_obs <- which(diff_slice2==min(diff_slice2))
d_obs <- data.frame(x=c(longData$Var2[index_1_obs],longData$Var2[index_2_obs]),y=c(longData$value[index_1_obs],longData$value[index_2_obs]),s1=c(rep(slice_locations[1],length(index_1_obs)),rep(slice_locations[2],length(index_1_obs))),group_i="Observed")

diff_slice1 <- abs(nd$s1 - slice_locations[1])
diff_slice2 <- abs(nd$s1 - slice_locations[2])
index_1_obs <- which(diff_slice1==min(diff_slice1))
index_2_obs <- which(diff_slice2==min(diff_slice2))
d_opti <- data.frame(x=c(nd$s2[index_1_obs],nd$s2[index_2_obs]),y=c(nd$p_opti[index_1_obs],nd$p_opti[index_2_obs]),s1=c(rep(slice_locations[1],length(index_1_obs)),rep(slice_locations[2],length(index_1_obs))),group_i="Optimal")

d_slice <- rbind(d_obs,d_opti)
d_slice$plot_group <- factor(paste(d_slice$group_i,d_slice$s1,sep="_"))
str(d_slice)

bayes_col <- rgb(255,80,80,maxColorValue=255)


pdf(file="D:/dual-decision-task-master/dual-decision-task-master/fig/sliced_2D_p.pdf",width=1.1*scale_factor,height=1.7*scale_factor)
#+geom_line(aes(color=y),size=1)
ggplot(d_slice,aes(x=x,y=y,group=plot_group,color=group_i))+geom_line(size=1)+nice_theme+facet_grid(s1~.)+scale_color_manual(values=c("black",bayes_col),name="")+coord_cartesian(xlim=c(0,2.4))+theme(legend.justification = c(1, 0), legend.position = c(1, 0))+labs(x=expression(paste("|",s[2],"| [",sigma,"]")),y="p(correct)")+theme(legend.background = element_rect(fill = "transparent"))+scale_x_continuous(breaks=c(0,1,2))
dev.off()

pdf(file="D:/reysy/dual-decision-task-master/dual-decision-task-master/fig/sliced_2D_p_h.pdf",width=1.8*scale_factor,height=1.3*scale_factor)
#+geom_line(aes(color=y),size=1)
ggplot(d_slice,aes(x=x,y=y,group=plot_group,color=group_i))+geom_line(size=1)+nice_theme+facet_grid(.~s1)+scale_color_manual(values=c("black",bayes_col),name="")+coord_cartesian(xlim=c(0,2.4))+theme(legend.justification = c(1, 0), legend.position = c(1, 0))+labs(x=expression(paste("|",s[2],"| [",sigma,"]")),y="p(correct)")+theme(legend.background = element_rect(fill = "transparent"))+scale_x_continuous(breaks=c(0,1,2))
dev.off()

# ------------------------------------------------------------------ #
## plot the remaining models?
# 
# pdf(file="./fig/sub2_median.pdf",width=2.3,height=1.6)
# ggplot(nd, aes(x=s1, y=s2, z = p_sub2, fill=p_sub2))+geom_raster(interpolate=T)+scale_fill_viridis(limits=c(0.5,1))+nice_theme+coord_equal(xlim=c(0.09,1.91),ylim=c(0.09,1.91))+labs(x=expression(paste("|",s[1],"| [",sigma,"]")), y=expression(paste("|",s[2],"| [",sigma,"]")),fill="p(correct)")+ggtitle("Discrete 2 lvl") + theme(plot.title = element_text(size = 8, face = "plain")) +geom_contour(color="white",breaks = contour_breaks)
# dev.off()
# 
# pdf(file="./fig/sub1_median.pdf",width=2.3,height=1.6)
# ggplot(nd, aes(x=s1, y=s2, z = p_fixd, fill=p_fixd))+geom_raster(interpolate=T)+scale_fill_viridis(limits=c(0.5,1))+nice_theme+coord_equal(xlim=c(0.09,1.91),ylim=c(0.09,1.91))+labs(x=expression(paste("|",s[1],"| [",sigma,"]")), y=expression(paste("|",s[2],"| [",sigma,"]")),fill="p(correct)")+ggtitle("Fixed-bias") + theme(plot.title = element_text(size = 8, face = "plain")) +geom_contour(color="white",breaks = contour_breaks)
# dev.off()
# 
# pdf(file="./fig/meta_noise_median.pdf",width=2.3,height=1.6)
# ggplot(nd, aes(x=s1, y=s2, z = p_meta, fill=p_meta))+geom_raster(interpolate=T)+scale_fill_viridis(limits=c(0.5,1))+nice_theme+coord_equal(xlim=c(0.09,1.91),ylim=c(0.09,1.91))+labs(x=expression(paste("|",s[1],"| [",sigma,"]")), y=expression(paste("|",s[2],"| [",sigma,"]")),fill="p(correct)")+ggtitle("Biased-Bayesian") + theme(plot.title = element_text(size = 8, face = "plain")) +geom_contour(color="white",breaks = contour_breaks)
# dev.off()

