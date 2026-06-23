# ------------------------------------------------------------------ #
# 1) fit models in individual data
# 2) save parameters and goodness-of-fit indicators
# ------------------------------------------------------------------ #

rm(list=ls())
setwd("D:/xiangmu/scripts/R")
source("./R/code_all_models.R")
source("./R/additional_optimal_policy_d2.R")      

# load everything
d <- read.table("D:/xiangmu/data/behavior/idsess/data_nu.txt", header=TRUE, sep=";")
str(d)

## exclude control sessions
d <- d[d$dual==1,]

# convert decision order to factor/character
d$decision <- as.character(d$decision)

fit_models <- function(d){
  d_s <- data.frame(
    s1 = d$s[d$decision=="1"],
    s2 = d$s[d$decision=="2"],
    r1 = d$rr[d$decision=="1"],
    r2 = d$rr[d$decision=="2"]
  )
  fctrl <- list(maxit=100000000)
  logLikOpti <- l_bayes_nf(d_s)
  sub1 <- optim(par=-.5, l_sub1_nf, d=d_s, hessian=TRUE,
                method="Brent", lower=-10, upper=0, control=fctrl)
  sub2 <- fit_sub_model(d_s, nlevels=2, hess=TRUE, maxb=10)
  meta_fit <- optimx::optimx(par=1, l_bayes_metanoise, d=d_s,
                             method=c("bobyqa"), lower=0.1, upper=10,
                             control=list(maxfun=10000000))
  model <- c("bayes","sub1","sub2","metanoise")
  logLik <- -c(logLikOpti, sub1$value, sub2$value, meta_fit$value)
  n_par <- c(0,1,2,1)
  AIC <- 2*n_par - 2*logLik
  relModLik <- exp(-(AIC - min(AIC))/2)
  w <- relModLik / sum(relModLik)
  fixed_shift <- c(NA, sub1$par, NA, NA)
  w1 <- c(NA, NA, sub2$par[1], NA)
  theta2 <- c(NA, NA, sub2$par[2], NA)
  m <- c(NA, NA, NA, unlist(meta_fit[1,1]))
  data.frame(model, logLik, n_par, AIC, relModLik, fixed_shift, w1, theta2, m, w)
}

# 实际存在的 id（排序后）
ids <- sort(unique(as.character(d$id)))

# 预检查（可选）
bad_ids <- c()
for (i in ids) {
  cat("checking id:", i, "... ")
  res <- try(fit_models(d[d$id == i, ]), silent = TRUE)
  if (inherits(res, "try-error")) {
    cat("FAILED\n"); bad_ids <- c(bad_ids, i)
  } else {
    cat("OK\n")
  }
}
cat("\n出错的 id 列表：\n")
print(bad_ids)

# do fitting
d_fit <- NULL
for(i in ids){
  d_id <- fit_models(d[d$id==i,])
  d_id$id <- i

  # 取该 id 的 seq（如无则 NA；多于1个则警告并取第一个）
  seq_i <- unique(d$seq[d$id==i])
  if (length(seq_i) > 1) warning(sprintf("id %s 对应多个 seq：%s，取第一个。", i, paste(seq_i, collapse=",")))
  d_id$seq <- if (length(seq_i) == 0) NA else seq_i[1]

  # test overall difference in prop. correct d1 vs d2
  d_s <- d[d$id==i,]
  n_ <- nrow(d_s)/2
  macc1 <- mean(d_s$acc[d_s$decision==1])
  macc2 <- mean(d_s$acc[d_s$decision==2])
  crosstab_acc <- matrix(c(macc1*n_, macc2*n_, n_-macc1*n_, n_-macc2*n_),
                         2, 2, dimnames=list(c("d1","d2"), c(1,0)))
  bin_test <- prop.test(crosstab_acc, alternative="less", correct=FALSE)
  d_id$acc.stat <- bin_test$statistic
  d_id$acc.p.value <- bin_test$p.value

  d_fit <- rbind(d_fit, d_id)
  cat(i, " done! moving on...\n\t")
}
cat("\ncompleted!")

write.table(d_fit, "D:/xiangmu/data/behavior_for_behavior/fit_par.txt", row.names=FALSE)