

library(Rcpp)
library(RcppArmadillo)
library(globpso)

sourceCpp(file.path('C:/Users/USER/Desktop/PYCProfessor/Lader GP/ladderGP-main/src/cppFunc.cpp'))
# 
# 
# ################################################################################
# ### Additive Ladder GP
# ################################################################################
# # y: vector
# # x: matrix
# # z: integer vector
# 
# #更改範圍 原本contiParRange = 10^c(-3, .5), 
# #varParRange = 10^c(-3, .5)
# #nugget = 0.
# #nSwarm = 64, maxIter = 200

# aLadderFit <- function(yList, xList,
#                        contiParRange = 10^c(-3, 3),
#                        varParRange = 10^c(-3, 2),
#                        nSwarm = 150, maxIter = 400, nugget = 1e-6, optVerbose = TRUE) {
# 
#   cputime <- system.time({
#     xDims <- sapply(1:length(xList), function(k) ncol(xList[[k]]))
#     xzDim <- min(xDims)
#     zs <- xDims/xzDim
#     x <- matrix(0, nrow = 0, ncol = max(xDims))
#     y <- z <- dimCheck <- c()
#     for (i in 1:length(yList)) {
#       n <- length(yList[[i]])
#       x <- rbind(x, cbind(xList[[i]], matrix(-1, n, ncol(x) - ncol(xList[[i]]))))
#       y <- c(y, yList[[i]])
#       z <- c(z, rep(xDims[i]/xzDim, n))
#       dimCheck[i] <- xDims[i] %% xzDim
#     }
#     stopifnot(all(dimCheck == 0))
# 
#     nContiPar <- ncol(x)
#     nVarPar <- max(z) + (0.5*max(z)*(max(z) - 1))
#     low_bound <- c(rep(min(contiParRange), nContiPar),
#                    rep(min(varParRange), nVarPar))
#     upp_bound <- c(rep(max(contiParRange), nContiPar),
#                    rep(max(varParRange), nVarPar))
# 
#     alg_setting <- getPSOInfo(nSwarm = nSwarm, maxIter = maxIter, psoType = "quantum")
# 
#     res <- globpso(objFunc = aIntObjCpp, lower = low_bound, upper = upp_bound,
#                    PSO_INFO = alg_setting, verbose = optVerbose,
#                    y = y, x = x, z = z, xzDim = xzDim, nugget = nugget)
#     #res$val
#     mdl <- aIntModel(param = res$par, y = y, x = x, z = z, xzDim = xzDim, nugget = nugget)
#     mdl$data <- list(y = y, x = x, z = z, xzDim = xzDim)
#   })[3]
#   mdl$cputime <- cputime
#   cat(sprintf("aLadderGP FIT CPU time: %.2f seconds.\n", cputime))
#   return(mdl)
# }

#--------------------------------------------------------------
#更改參數範圍
aLadderFit <- function(yList, xList, 
                       contiParRange = 10^c(-3, 3), 
                       varParRange = 10^c(-3, 2),
                       nSwarm = 150, maxIter = 400, nugget = 1e-6, optVerbose = TRUE) {
  
  cputime <- system.time({
    # --- Part 1: 數據準備 --
    #教授的版本如果 xList 裡面有一個元素是 NULL 或不是矩陣，ncol() 會報錯，程式停。
    #我的是先檢查這個東西有沒有 ncol，如果沒有，就當作它的維度是 0
    #增加了 valid_xDims 這個變數，並且只從有效的維度中找最小值
    xDims <- sapply(1:length(xList), function(k) if(!is.null(ncol(xList[[k]]))) ncol(xList[[k]]) else 0)
    valid_xDims <- xDims[sapply(xList, function(m) !is.null(m) && NROW(m) > 0)]
    if (length(valid_xDims) == 0) stop("xList 中所有元素都为空或没有行。")
    xzDim <- min(valid_xDims[valid_xDims > 0])
    
    max_xDims <- if(length(xDims) > 0) max(xDims) else 0
    
    x <- matrix(0, nrow = 0, ncol = max_xDims)
    y <- z <- c()
    dimCheck <- logical(length(yList))
    
    for (i in 1:length(yList)) {
      n <- length(yList[[i]])
      if(n == 0) next
      
      current_x <- xList[[i]]
      if(!is.matrix(current_x)) current_x <- as.matrix(current_x)
      
      padding_cols <- max_xDims - ncol(current_x)
      if (padding_cols > 0) {
        padding_matrix <- matrix(-1, n, padding_cols)
        current_x_padded <- cbind(current_x, padding_matrix)
      } else {
        current_x_padded <- current_x
      }
      
      x <- rbind(x, current_x_padded)
      y <- c(y, yList[[i]])
      z <- c(z, rep(xDims[i]/xzDim, n))
      dimCheck[i] <- (xDims[i] %% xzDim) == 0
    }
    stopifnot(all(dimCheck, na.rm = TRUE))
    
    # --- Part 2: 超參數邊界設定 (修改) ---
    zMax <- max(z)
    n_thetaZ <- zMax * xzDim  
    n_sigmaF <- zMax
    n_sigmaInt <- 0.5 * zMax * (zMax - 1)
    
    total_params_needed <- n_thetaZ + n_sigmaF + n_sigmaInt
    
    cat(sprintf("--- R aLadderFit DEBUG ---\n"))
    cat(sprintf("R is preparing boundaries for %d parameters.\n", total_params_needed))
    cat(sprintf("Breakdown: n_thetaZ=%d, n_sigmaF=%d, n_sigmaInt=%d\n", n_thetaZ, n_sigmaF, n_sigmaInt))
    cat(sprintf("---------------------------\n"))
    
    # 2a. 定義原始尺度的邊界
    low_bound_orig <- c(rep(min(contiParRange), n_thetaZ), # 使用 n_thetaZ
                        rep(min(varParRange), n_sigmaF + n_sigmaInt)) # 把兩種 variance 參數合在一起
    upp_bound_orig <- c(rep(max(contiParRange), n_thetaZ), # 使用 n_thetaZ
                        rep(max(varParRange), n_sigmaF + n_sigmaInt))
    
    # 2b. 將邊界轉換到對數尺度
    low_bound_log <- log(low_bound_orig)
    upp_bound_log <- log(upp_bound_orig)
    
    # 2c. 創建一個包裝函式(wrapper)，用於在優化過程中轉換尺度
    objective_wrapper_log <- function(params_log) {
      params_orig <- exp(params_log)
      return(aIntObjCpp(params_orig, y, x, z, xzDim, nugget))
    }
    
    # --- Part 3: 執行優化 (修改) ---
    
    alg_setting <- getPSOInfo(nSwarm = nSwarm, maxIter = maxIter, psoType = "quantum")
    
    # 讓 globpso 在對數尺度上進行優化
    res_log <- globpso(objFunc = objective_wrapper_log, 
                       lower = low_bound_log, 
                       upper = upp_bound_log,
                       PSO_INFO = alg_setting, 
                       verbose = optVerbose)
    
    # 將優化器找到的最佳解，從對數尺度轉回原始尺度
    best_params_orig <- exp(res_log$par)
    
    # --- Part 4: 建立最終模型 (修改) ---
    # 使用原始尺度的最佳參數來建立最終的模型物件
    mdl <- aIntModel(param = best_params_orig, y = y, x = x, z = z, xzDim = xzDim, nugget = nugget)
    mdl$data <- list(y = y, x = x, z = z, xzDim = xzDim)
    
  })[3]
  mdl$cputime <- cputime
  cat(sprintf("aLadderGP FIT CPU time: %.2f seconds.\n", cputime))
  return(mdl)
}

aLadderPred <- function(gpMdl, x0List, y0listTrue = NULL, ei_alpha = 0.5, min_y = NULL) {

  cputime <- system.time({

    xDims <- sapply(1:length(x0List), function(k) ncol(x0List[[k]]))
    xzDim <- min(xDims)
    zs <- xDims/xzDim
    x0 <- matrix(0, nrow = 0, ncol = max(xDims))
    z0 <- dimCheck <- c()
    for (i in 1:length(x0List)) {
      n <- nrow(x0List[[i]])
      x0 <- rbind(x0, cbind(x0List[[i]], matrix(-1, n, ncol(x0) - ncol(x0List[[i]]))))
      z0 <- c(z0, rep(xDims[i]/xzDim, n))
      dimCheck[i] <- xDims[i] %% xzDim
    }
    stopifnot(all(dimCheck == 0))

    if (!is.null(y0listTrue)) {
      stopifnot(length(x0List) == length(y0listTrue))
      yTrue <- c()
      for (i in 1:length(yList)) {
        yTrue <- c(yTrue, y0listTrue[[i]])
      }
    } else {
      yTrue <- "Empty true value in the input arguments"
    }

    if (is.null(min_y)) { min_y <- min(gpMdl$data$y) }

    pred <- aIntPred(x0, z0, gpMdl$data$y, gpMdl$data$x, gpMdl$data$z, gpMdl$data$xzDim,
                     gpMdl$vecParams, gpMdl$invPsi, gpMdl$mu, ei_alpha, min_y)
  })[3]

  pred$y_true <- yTrue
  return(pred)
}