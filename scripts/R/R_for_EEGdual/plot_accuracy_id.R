# ------------------------------------------------------------------ #
# This script plots the data together with model predictions
# (split by correctness of the first response)
# Matteo Lisi, 2020
# 新增功能：绘制每个被试对应的"After wrong first decision"图表
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
library(ggpubr)  # 用于图表组合
nice_theme <- theme_bw()+theme(text=element_text(family="Helvetica",size=9),panel.border=element_blank(),strip.background = element_rect(fill="white",color="white",size=0),strip.text=element_text(size=rel(0.8)),panel.grid.major.x=element_blank(),panel.grid.major.y=element_blank(),panel.grid.minor=element_blank(),axis.line.x=element_line(size=.4),axis.line.y=element_line(size=.4),axis.text.x=element_text(size=7,color="black"),axis.text.y=element_text(size=7,color="black"),axis.line=element_line(size=.4), axis.ticks=element_line(color="black"))

# ------------------------------------------------------#
# do plots starting with correct first decision

# general plot parameters
n_bin <- 9
x_coord_max <- 2.45
y_coord_max_c1 <- 1#0.18
fig_height <- 1.8
fig_width <- 1.5
sizedataline <- NA
typedatapoint <- 21

x_breaks <- c(0, 0.5, 1, 1.5, 2)
x_labels <- c("0", "0.5", "1", "1.5", "2") # just to avoid unnecessary double-digits and reduce cluttering

# select correct first decision only
d <- read.table(file="E:/raweeg/data/data_wide_wmPred.txt", sep=";", header=T)
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
plo_opti <- ggplot(d_plot,aes(x=s1, y=r2, ymin=r2-r2_se,ymax=r2+r2_se))+nice_theme+geom_hline(yintercept=0,color="dark grey")+geom_ribbon(alpha=0.3,size=0,fill=bayes_col)+geom_line(size=0.6,color=bayes_col)+geom_point(data=dag2,pch=typedatapoint,color=color_data)+geom_line(data=dag2,size=sizedataline,color=color_data)+geom_errorbar(data=dag2,width=0,color=color_data,size=0.4)+coord_cartesian(xlim=c(0,x_coord_max),ylim=c(-1,y_coord_max_c1))+ggtitle(model_plot)+labs(x =expression(paste("|",S[1],"| [",sigma,"]") ), y="p(choose 'right' option)\n difference from baseline")+theme(plot.title = element_text(size = 8))+scale_x_continuous(breaks=x_breaks, labels=x_labels)
pdf("E:/raweeg/fig/optimal_correctC1.pdf",width=fig_width,height=fig_height)
plo_opti
dev.off()
#+geom_line(data=dag2,size=0.4)

model_plot <- "After correct\nfirst decision"
d_plot <- with(d_m, data.frame(bin_s1=NA, s1=abs1, r2=p_r2_bayesian, r2_se=p_r2_bayesian_se, r2_naive=p_r2_naive,r2_naive_se=p_r2_naive_se, r2_optimal = p_r2_optimal))
d_plot$model <- model_plot
plo1 <- ggplot(d_plot,aes(x=s1, y=r2, ymin=r2-r2_se,ymax=r2+r2_se))+nice_theme+geom_hline(yintercept=0,color="dark grey")+geom_ribbon(alpha=0.3,size=0,fill=bayes_col)+geom_line(size=0.6,color=bayes_col)+geom_point(data=dag2,pch=typedatapoint,color=color_data)+geom_line(data=dag2,size=sizedataline,color=color_data)+geom_errorbar(data=dag2,width=0,color=color_data,size=0.4)+coord_cartesian(xlim=c(0,x_coord_max),ylim=c(-1,y_coord_max_c1))+ggtitle(model_plot)+theme(plot.title = element_text(size = 8))+labs(x =expression(paste("|",S[1],"| [",sigma,"]") ), y="p(choose 'right' option)\n difference from baseline")+theme(plot.title = element_text(size = 8))+geom_line(aes(y=r2_optimal),size=0.4,color="red",lty=2)+scale_x_continuous(breaks=x_breaks, labels=x_labels)
# pdf(paste("D:/11.13/nhb_data_code/fig/",model_plot,"_r2_ACC11.pdf",sep=""),width=fig_width,height=fig_height)
# plo1
# dev.off()

model_plot <- " \n "
d_plot <- with(d_m, data.frame(bin_s1=NA, s1=abs1, r2=p_r2_discrete, r2_se=p_r2_discrete_se, r2_optimal = p_r2_optimal))
d_plot$model <- model_plot
plo2 <- ggplot(d_plot,aes(x=s1, y=r2,ymin=r2-r2_se,ymax=r2+r2_se))+nice_theme+geom_hline(yintercept=0,color="dark grey")+geom_ribbon(alpha=0.3,size=0,fill=sub_2_col)+geom_line(size=0.6,color=sub_2_col)+geom_point(data=dag2,pch=typedatapoint,color=color_data)+geom_line(data=dag2,size=sizedataline,color=color_data)+geom_errorbar(data=dag2,width=0,color=color_data,size=0.4)+coord_cartesian(xlim=c(0,x_coord_max),ylim=c(-1,y_coord_max_c1))+ggtitle(model_plot)+theme(plot.title = element_text(size = 8))+labs(x =expression(paste("|",S[1],"| [",sigma,"]") ), y="p(choose 'right' option)\n difference from baseline")+geom_line(aes(y=r2_optimal),size=0.4,color="red",lty=2)+scale_x_continuous(breaks=x_breaks, labels=x_labels)
# pdf(paste("D:/11.13/nhb_data_code/fig/",model_plot,"_r2_ACC11.pdf",sep=""),width=fig_width,height=fig_height)
# plo2
# dev.off()

model_plot <- " \n "
d_plot <- with(d_m, data.frame(bin_s1=NA, s1=abs1, r2=p_r2_fixed, r2_se=p_r2_fixed_se, r2_optimal = p_r2_optimal))
d_plot$model <- model_plot
plo3 <- ggplot(d_plot,aes(x=s1, y=r2, ymin=r2-r2_se,ymax=r2+r2_se))+nice_theme+geom_hline(yintercept=0,color="dark grey")+geom_ribbon(alpha=0.3,size=0,fill="dark grey")+geom_line(size=0.6,color="dark grey")+geom_point(data=dag2,pch=typedatapoint,color=color_data)+geom_line(data=dag2,size=sizedataline,color=color_data)+geom_errorbar(data=dag2,width=0,color=color_data,size=0.4)+coord_cartesian(xlim=c(0,x_coord_max),ylim=c(-1,y_coord_max_c1))+ggtitle(model_plot)+theme(plot.title = element_text(size = 8))+labs(x =expression(paste("|",S[1],"| [",sigma,"]") ), y="p(choose 'right' option)\n difference from baseline")+geom_line(aes(y=r2_optimal),size=0.4,color="red",lty=2)+scale_x_continuous(breaks=x_breaks, labels=x_labels)
# pdf(paste("D:/11.13/nhb_data_code/fig/",model_plot,"_r2_ACC11.pdf",sep=""),width=fig_width,height=fig_height)
# plo3 
# dev.off()


# ------------------------------------------------------ #
# ------------------------------------------------------ #
# Now repeat the same but conditioned on wrong first decision

d <- read.table(file="E:/raweeg/data/data_wide_wmPred.txt", sep=";", header=T)
d_wrong <- d[d$acc1==0,]  # 保存原始错误数据

# plot settings
y_coord_max_c0 <- 0.25#0.25
# bin width is 0.5 sigmas
cut_points <- c(0, 0.5, 1, 1.5, 20)
d_wrong$bin_s1 <- cut(d_wrong$abs1, breaks=cut_points) # 数据分箱
d_wrong$x_s1 <- cut(d_wrong$abs1, breaks=cut_points) # 模型分箱

# 创建个体被试图表保存目录
if(!dir.exists("E:/raweeg/fig/individual_wrong")){
  dir.create("E:/raweeg/fig/individual_wrong", recursive = TRUE)
}

# 获取所有被试ID
unique_subjects <- unique(d_wrong$id)
cat("发现", length(unique_subjects), "个被试将被绘制\n")

# 定义绘制个体被试图表的函数
plot_individual_subject <- function(subject_id) {
  cat("正在绘制被试 ID:", subject_id, "\n")
  
  # 筛选当前被试的数据
  sub_data <- d_wrong[d_wrong$id == subject_id, ]
  
  # 数据转换
  sub_data$p_r2_discrete <- sub_data$p_r2_discrete - sub_data$p_r2_naive
  sub_data$p_r2_bayesian <- sub_data$p_r2_bayesian - sub_data$p_r2_naive
  sub_data$p_r2_optimal <- sub_data$p_r2_optimal - sub_data$p_r2_naive
  sub_data$p_r2_fixed <- sub_data$p_r2_fixed - sub_data$p_r2_naive
  sub_data$r2 <- sub_data$r2 - sub_data$p_r2_naive
  
  # 数据聚合（按分箱计算均值）
  sub_agg <- aggregate(cbind(p_r2_discrete, p_r2_bayesian, p_r2_fixed, abs1, p_r2_naive, p_r2_optimal) ~ x_s1, sub_data, mean)
  
  # 计算标准误差（使用引导法）
  sub_agg$p_r2_discrete_se <- bootMeanSE(sub_data$p_r2_discrete)
  sub_agg$p_r2_bayesian_se <- bootMeanSE(sub_data$p_r2_bayesian)
  sub_agg$p_r2_fixed_se <- bootMeanSE(sub_data$p_r2_fixed)
  sub_agg$p_r2_optimal_se <- bootMeanSE(sub_data$p_r2_optimal)
  
  # 准备观测数据
  sub_obs_agg <- aggregate(cbind(abs1, r2) ~ bin_s1, sub_data, mean)
  sub_obs_agg$r2_se <- bootMeanSE(sub_data$r2)
  sub_obs_agg$model <- "observed\ndata"
  colnames(sub_obs_agg) <- c("bin_s1", "s1", "r2", "r2_se", "model")
  
  # 绘制最优模型图
  model_plot <- paste("After wrong\nfirst decision\n(subject ID:", subject_id, ")")
  d_plot <- data.frame(
    bin_s1 = NA,
    s1 = sub_agg$abs1,
    r2 = sub_agg$p_r2_optimal,
    r2_se = sub_agg$p_r2_optimal_se,
    r2_optimal = sub_agg$p_r2_optimal
  )
  
  p_optimal <- ggplot(d_plot, aes(x = s1, y = r2, ymin = r2 - r2_se, ymax = r2 + r2_se)) +
    nice_theme +
    geom_hline(yintercept = 0, color = "dark grey") +
    geom_ribbon(alpha = 0.3, size = 0, fill = bayes_col) +
    geom_line(size = 0.6, color = bayes_col) +
    geom_point(data = sub_obs_agg, pch = typedatapoint, color = "black") +
    geom_line(data = sub_obs_agg, size = 0.4, color = "black") +
    geom_errorbar(data = sub_obs_agg, width = 0, color = "black", size = 0.4) +
    coord_cartesian(xlim = c(0, x_coord_max), ylim = c(-0.5, y_coord_max_c0)) +
    ggtitle(model_plot) +
    labs(
      x = expression(paste("|", S[1], "| [", sigma, "]")),
      y = "p(choose 'right' option)\n difference from baseline"
    ) +
    theme(plot.title = element_text(size = 8)) +
    scale_x_continuous(breaks = x_breaks, labels = x_labels)
  
  # 绘制贝叶斯模型图
  d_plot_bayes <- data.frame(
    bin_s1 = NA,
    s1 = sub_agg$abs1,
    r2 = sub_agg$p_r2_bayesian,
    r2_se = sub_agg$p_r2_bayesian_se,
    r2_optimal = sub_agg$p_r2_optimal
  )
  
  p_bayes <- ggplot(d_plot_bayes, aes(x = s1, y = r2, ymin = r2 - r2_se, ymax = r2 + r2_se)) +
    nice_theme +
    geom_hline(yintercept = 0, color = "dark grey") +
    geom_ribbon(alpha = 0.3, size = 0, fill = bayes_col) +
    geom_line(size = 0.6, color = bayes_col) +
    geom_point(data = sub_obs_agg, pch = typedatapoint, color = "black") +
    geom_line(data = sub_obs_agg, size = 0.4, color = "black") +
    geom_errorbar(data = sub_obs_agg, width = 0, color = "black", size = 0.4) +
    coord_cartesian(xlim = c(0, x_coord_max), ylim = c(5, y_coord_max_c0)) +
    ggtitle(model_plot) +
    theme(plot.title = element_text(size = 8)) +
    labs(
      x = expression(paste("|", S[1], "| [", sigma, "]")),
      y = "p(choose 'right' option)\n difference from baseline"
    ) +
    geom_line(aes(y = r2_optimal), size = 0.4, color = "red", lty = 2) +
    scale_x_continuous(breaks = x_breaks, labels = x_labels)
  
  # 绘制离散模型图
  d_plot_disc <- data.frame(
    bin_s1 = NA,
    s1 = sub_agg$abs1,
    r2 = sub_agg$p_r2_discrete,
    r2_se = sub_agg$p_r2_discrete_se,
    r2_optimal = sub_agg$p_r2_optimal
  )
  
  p_disc <- ggplot(d_plot_disc, aes(x = s1, y = r2, ymin = r2 - r2_se, ymax = r2 + r2_se)) +
    nice_theme +
    geom_hline(yintercept = 0, color = "dark grey") +
    geom_ribbon(alpha = 0.3, size = 0, fill = sub_2_col) +
    geom_line(size = 0.6, color = sub_2_col) +
    geom_point(data = sub_obs_agg, pch = typedatapoint, color = "black") +
    geom_line(data = sub_obs_agg, size = 0.4, color = "black") +
    geom_errorbar(data = sub_obs_agg, width = 0, color = "black", size = 0.4) +
    coord_cartesian(xlim = c(0, x_coord_max), ylim = c(5, y_coord_max_c0)) +
    ggtitle(model_plot) +
    theme(plot.title = element_text(size = 8)) +
    labs(
      x = expression(paste("|", S[1], "| [", sigma, "]")),
      y = "p(choose 'right' option)\n difference from baseline"
    ) +
    geom_line(aes(y = r2_optimal), size = 0.4, color = "red", lty = 2) +
    scale_x_continuous(breaks = x_breaks, labels = x_labels)
  
  # 绘制固定模型图
  d_plot_fixed <- data.frame(
    bin_s1 = NA,
    s1 = sub_agg$abs1,
    r2 = sub_agg$p_r2_fixed,
    r2_se = sub_agg$p_r2_fixed_se,
    r2_optimal = sub_agg$p_r2_optimal
  )
  
  p_fixed <- ggplot(d_plot_fixed, aes(x = s1, y = r2, ymin = r2 - r2_se, ymax = r2 + r2_se)) +
    nice_theme +
    geom_hline(yintercept = 0, color = "dark grey") +
    geom_ribbon(alpha = 0.3, size = 0, fill = "dark grey") +
    geom_line(size = 0.6, color = "dark grey") +
    geom_point(data = sub_obs_agg, pch = typedatapoint, color = "black") +
    geom_line(data = sub_obs_agg, size = 0.4, color = "black") +
    geom_errorbar(data = sub_obs_agg, width = 0, color = "black", size = 0.4) +
    coord_cartesian(xlim = c(0, x_coord_max), ylim = c(0, y_coord_max_c0)) +
    ggtitle(model_plot) +
    theme(plot.title = element_text(size = 8)) +
    labs(
      x = expression(paste("|", S[1], "| [", sigma, "]")),
      y = "p(choose 'right' option)\n difference from baseline"
    ) +
    geom_line(aes(y = r2_optimal), size = 0.4, color = "red", lty = 2) +
    scale_x_continuous(breaks = x_breaks, labels = x_labels)
  
  # 组合四个模型的图表
  p_combined <- ggarrange(
    p_optimal, p_bayes,
    p_disc, p_fixed,
    ncol = 2, nrow = 2,
    common.legend = TRUE,
    legend = "bottom"
  )
  
  # 保存为PDF文件
  pdf_file <- paste("E:/raweeg/fig/individual_wrong/subject_", subject_id, ".pdf", sep = "")
  pdf(pdf_file, width = fig_width * 2, height = fig_height * 2)
  print(p_combined)
  dev.off()
  
  cat("已保存图表到", pdf_file, "\n")
}

# 循环绘制每个被试的图表
for (subj in unique_subjects) {
  plot_individual_subject(subj)
}

# 继续原有脚本的群体分析部分
d <- d_wrong  # 恢复为原始错误数据

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
plo_opti <- ggplot(d_plot,aes(x=s1, y=r2, ymin=r2-r2_se,ymax=r2+r2_se))+nice_theme+geom_hline(yintercept=0,color="dark grey")+geom_ribbon(alpha=0.3,size=0,fill=bayes_col)+geom_line(size=0.6,color=bayes_col)+geom_point(data=dag2,pch=typedatapoint,color=color_data)+geom_line(data=dag2,size=sizedataline,color=color_data)+geom_errorbar(data=dag2,width=0,color=color_data,size=0.4)+coord_cartesian(xlim=c(0,x_coord_max),ylim=c(-1,y_coord_max_c0))+ggtitle(model_plot)+labs(x =expression(paste("|",S[1],"| [",sigma,"]") ), y="p(choose 'right' option)\n difference from baseline")+theme(plot.title = element_text(size = 8))+scale_x_continuous(breaks=x_breaks, labels=x_labels)
pdf("E:/raweeg/fig/optimal_wrongD1.pdf",width=fig_width,height=fig_height)
plo_opti
dev.off()

model_plot <- " \n "
d_plot <- with(d_m, data.frame(bin_s1=NA, s1=abs1, r2=p_r2_discrete, r2_se=p_r2_discrete_se, r2_optimal = p_r2_optimal))
d_plot$model <- model_plot
plo4 <- ggplot(d_plot,aes(x=s1, y=r2,ymin=r2-r2_se,ymax=r2+r2_se))+nice_theme+geom_hline(yintercept=0,color="dark grey")+geom_ribbon(alpha=0.3,size=0,fill=sub_2_col)+geom_line(size=0.6,color=sub_2_col)+geom_point(data=dag2,pch=typedatapoint,color=color_wrong)+geom_line(data=dag2,size=sizedataline,color=color_wrong)+geom_errorbar(data=dag2,width=0,color=color_wrong,size=0.4)+coord_cartesian(xlim=c(0,x_coord_max),ylim=c(-1,y_coord_max_c0))+ggtitle(model_plot)+theme(plot.title = element_text(size = 8))+labs(x =expression(paste("|",S[1],"| [",sigma,"]") ), y=" \n ")+geom_line(aes(y=r2_optimal),size=0.4,color="red",lty=2)+scale_x_continuous(breaks=x_breaks, labels=x_labels)
# pdf(paste("D:/11.13/nhb_data_code/R/fig/",model_plot,"_r2_ACC10.pdf",sep=""),width=fig_width,height=fig_height)
# plo4
# dev.off()

model_plot <- "After wrong\nfirst decision"
d_plot <- with(d_m, data.frame(bin_s1=NA, s1=abs1, r2=p_r2_bayesian, r2_se=p_r2_bayesian_se, r2_naive=p_r2_naive,r2_naive_se=p_r2_naive_se, r2_optimal = p_r2_optimal))
d_plot$model <- model_plot
plo5 <- ggplot(d_plot,aes(x=s1, y=r2, ymin=r2-r2_se,ymax=r2+r2_se))+nice_theme+geom_hline(yintercept=0,color="dark grey")+geom_ribbon(alpha=0.3,size=0,fill=bayes_col)+geom_line(size=0.6,color=bayes_col)+geom_point(data=dag2,pch=typedatapoint,color=color_wrong)+geom_line(data=dag2,size=sizedataline,color=color_wrong)+geom_errorbar(data=dag2,width=0,color=color_wrong,size=0.4)+coord_cartesian(xlim=c(0,x_coord_max),ylim=c(-1,y_coord_max_c0))+ggtitle(model_plot)+theme(plot.title = element_text(size = 8))+labs(x =expression(paste("|",S[1],"| [",sigma,"]") ), y=" \n ")+geom_line(aes(y=r2_optimal),size=0.4,color="red",lty=2)+scale_x_continuous(breaks=x_breaks, labels=x_labels)
# pdf(paste("D:/11.13/nhb_data_code/R/fig/",model_plot,"_r2_ACC10.pdf",sep=""),width=fig_width,height=fig_height)
# plo5
# dev.off()

model_plot <- " \n "
d_plot <- with(d_m, data.frame(bin_s1=NA, s1=abs1, r2=p_r2_fixed, r2_se=p_r2_fixed_se, r2_optimal = p_r2_optimal))
d_plot$model <- model_plot
plo6 <- ggplot(d_plot,aes(x=s1, y=r2, ymin=r2-r2_se,ymax=r2+r2_se))+nice_theme+geom_hline(yintercept=0,color="dark grey")+geom_ribbon(alpha=0.3,size=0,fill="dark grey")+geom_line(size=0.6,color="dark grey")+geom_point(data=dag2,pch=typedatapoint,color=color_wrong)+geom_line(data=dag2,size=sizedataline,color=color_wrong)+geom_errorbar(data=dag2,width=0,color=color_wrong,size=0.4)+coord_cartesian(xlim=c(0,x_coord_max),ylim=c(-1,y_coord_max_c0))+ggtitle(model_plot)+theme(plot.title = element_text(size = 8))+labs(x =expression(paste("|",S[1],"| [",sigma,"]") ), y=" \n ")+geom_line(aes(y=r2_optimal),size=0.4,color="red",lty=2)+scale_x_continuous(breaks=x_breaks, labels=x_labels)
# pdf(paste("./fig/",model_plot,"_r2_ACC10.pdf",sep=""),width=fig_width,height=fig_height)
# plo6
# dev.off()

# make big plot
pdf("E:/raweeg/fig/pred_all.pdf",width=fig_width*2,height=fig_height*3)
ggarrange(plo5, plo1, plo4, plo2, plo6, plo3,
          ncol = 2, nrow = 3)
dev.off()