#
# make lookup table for the integral of optimal mosign_sdel
#

# clear workspace?
rm(list=ls())

binsize <- 0.0125 #0.00625
st <- seq(-10,10,binsize)
tab_plus = matrix(nrow=length(st),ncol=length(st))
tab_minus = matrix(nrow=length(st),ncol=length(st))


# functions

# erf <- function(x)
# {
# 	# from: numerical recipes in C, 2nd ed
# 	t <- 1/(1+0.5*abs(x))
# 	tau <- t*exp(-x^2 -1.26551223 + 1.00002368*t + 0.37409196*t^2 + 0.09678418*t^3 - 0.18628806*t^4 + 0.27886807*t^5 - 1.13520398*t^6 + 1.48851587*t^7 - 0.82215223*t^8 + 0.17087277*t^9)
# 	return(ifelse(x>=0, 1-tau, tau -1))
# }
# alternative formulation in R
erf <- function(x) 2 * pnorm(x * sqrt(2)) - 1

Fplus <- function(x,s1,s2) erf(x + s2) * exp(-(x-s1)^2)
Fminus <- function(x,s1,s2) erf(x - s2) * exp(-(x-s1)^2)

# fill table (s1 on rows)
for(s1 in 1:length(st))
{
	for(s2 in 1:length(st))
	{
		integrand_plus <- function(x) Fplus(x, st[s1], st[s2])
		integrand_minus <- function(x) Fminus(x, st[s1], st[s2])
		tab_plus[s1,s2] <- (1/sqrt(pi)) * integrate(integrand_plus, lower= 0, upper= Inf)$value
		tab_minus[s1,s2] <- (1/sqrt(pi)) * integrate(integrand_minus, lower= -Inf, upper= 0)$value
	}
}

# save the RDS files that are necessary to estimate likelihood of optimal model
saveRDS(tab_plus,file="tabplus.rds",compress=F)
saveRDS(tab_minus,file="tabminus.rds",compress=F)
saveRDS(st,file="tabindex.rds",compress=F)

# lookup function (also perform bilinear interpolation)
# intF <- function(s1,s2,tab,ind)
# {
	# ti <- findInterval(c(s1,s2), ind)
	# a <- matrix(c(rep(1,4),rep(ind[ti[1]],2),rep(ind[ti[1]+1],2),ind[ti[2]],ind[ti[2]+1],ind[ti[2]],ind[ti[2]+1]),4,4)
	# a[,4] <- a[,2]*a[,3]
	# b <- c(tab[ti[1],ti[2]], tab[ti[1],ti[2]+1], tab[ti[1]+1,ti[2]], tab[ti[1],ti[2]+1])
	# beta <- solve(a,b)
	# return(beta[1] + beta[2]*s1 + beta[3]*s2 + beta[4]*s1*s2)
# }

# faster version 
# (don't use solve, which is an interface to fortran package LAPACK, routines DGESV and ZGESV)
intF <- function(x,y,tab,ind)
{
	ti <- findInterval(c(x,y), ind)
	x1 <- ind[ti[1]]; x2 <- ind[ti[1]+1]
	y1 <- ind[ti[2]]; y2 <- ind[ti[2]+1]
	(1/((x2-x1)*(y2-y1))) * (tab[ti[1],ti[2]]*(x2-x)*(y2-y) + tab[ti[1]+1,ti[2]]*(x-x1)*(y2-y) + tab[ti[1],ti[2]+1]*(x2-x)*(y-y1) + tab[ti[1]+1,ti[2]+1]*(x-x1)*(y-y1))
}


#### SANITY CHECK #####
# 
# compare values from table+linear interpolation to that 
# obtained from numerical integration
# 

# numFplus <- function(s1,s2)
# {
# 	integrand <- function(x) Fplus(x, s1, s2)
# 	return((1/sqrt(pi)) * integrate(integrand, lower= 0, upper= Inf)$value)
# }
# 
# # test 1
# s1 <- runif(1,min=-1,max=1)
# s2 <- runif(1,min=-1,max=1)
# system.time(intF(s1,s2,tab_plus,st), gcFirst=T) # faster
# system.time(numFplus(s1,s2), gcFirst=T)
# 
# 
# # test  2
# n <- 5000
# s1 <- runif(n,min=-1,max=1)
# s2 <- runif(n,min=-1,max=1)
# tab_val <- vector(length=n, mode="numeric")
# num_val <- vector(length=n, mode="numeric")
# 
# system.time(for(i in 1:n){tab_val[i] <- intF(s1[i],s2[i],tab_plus,st)}, gcFirst=T)
# system.time(for(i in 1:n){num_val[i] <- numFplus(s1[i],s2[i])}, gcFirst=T)
# 
# # compare output
# 
# quartz(width=12,height=4)
# par(mfrow=c(1,3),mar=(c(4, 3.5, 3.5, 2) + 0.1),mgp=c(2.5,0.4,0),cex=0.7)
# # plot(tab_val, num_val,pch=".",col="blue",xlab="table lookup",ylab="numerical integration")
# hist(abs(tab_val - num_val),xlab="absolute error",breaks=30,col="light blue",main="")
# plot(num_val,tab_val - num_val,pch=21,col="blue",xlab="numerical integration",ylab="error (table - numerical)",main=paste("bin = ",binsize),cex=0.5,ylim=c(-0.0001,0.0001))
# abline(h=0,lty =2)
# plot(num_val,(tab_val - num_val)/(num_val),pch=21,cex=0.5,col="blue",xlab="numerical integration",ylab="relative error [(table - numerical)/numerical]",ylim=c(-0.012,0.012))
# abline(h=c(0.01, -0.01),lty =2)





