# =========================================================================
# File: R/aladderGP.R
# Version: 2025-12-04 (with Y-scaling fix)
# Author: Yu-Long Wang & Ping-Yang Chen (with modifications)
# Description: Implements the Additive Ladder Gaussian Process model.
#              This version includes Y-scaling for numerical stability.
# =========================================================================

# 確保 C++ 函數已經被載入
# library(Rcpp)
# library(RcppArmadillo)
# library(globpso)
# sourceCpp('src/cppFunc.cpp')


################################################################################
### Additive Ladder GP Fit (with Y-Scaling)
################################################################################
aLadderFit <- function(yList, xList, 
                       contiParLogRange = c(-6.5, 1.5), 
                       varParLogRange = c(-6.5, 1.5),
                       nSwarm = 64, maxIter = 200, psoType = "basic", nugget = 1e-6, optVerbose = TRUE) {
  
  cputime <- system.time({
    # --- 1. Prepare X and Z data for C++ ---
    xDims <- sapply(1:length(xList), function(k) ncol(xList[[k]]))
    xzDim <- min(xDims)
    x <- matrix(0, nrow = 0, ncol = max(xDims))
    y_raw <- z <- dimCheck <- c()
    for (i in 1:length(yList)) {
      n <- length(yList[[i]])
      x <- rbind(x, cbind(xList[[i]], matrix(-1, n, ncol(x) - ncol(xList[[i]]))))
      y_raw <- c(y_raw, yList[[i]])
      z <- c(z, rep(xDims[i]/xzDim, n))
      dimCheck[i] <- xDims[i] %% xzDim
    }
    stopifnot(all(dimCheck == 0))
    
    # --- 2. [CRITICAL FIX] Standardize the target variable Y ---
    y_mean <- mean(y_raw)
    y_sd <- sd(y_raw)
    
    # Prevent division by zero if all y values are the same
    if (is.na(y_sd) || y_sd < 1e-9) { 
      y_sd <- 1 
    }
    
    y_scaled <- (y_raw - y_mean) / y_sd
    
    # --- 3. Set up PSO optimization ---
    nContiPar <- ncol(x)
    nVarPar <- max(z) + (0.5*max(z)*(max(z) - 1))
    
    low_bound <- c(rep(min(contiParLogRange), nContiPar),
                   rep(min(varParLogRange), nVarPar))
    upp_bound <- c(rep(max(contiParLogRange), nContiPar),
                   rep(max(varParLogRange), nVarPar))
    
    alg_setting <- getPSOInfo(nSwarm = nSwarm, maxIter = maxIter, psoType = psoType)
    
    # --- 4. Run PSO using the scaled Y ---
    res <- globpso(objFunc = aIntObjCpp, lower = low_bound, upper = upp_bound,
                   PSO_INFO = alg_setting, verbose = optVerbose,
                   y = y_scaled, x = x, z = z, xzDim = xzDim, nugget = nugget, logParam = TRUE)
    
    # --- 5. Build the final model using optimized parameters and scaled Y ---
    mdl <- aIntModel(param = res$par, y = y_scaled, x = x, z = z, xzDim = xzDim, nugget = nugget, logParam = TRUE)
    
    # --- 6. [DIAGNOSTIC] Check for matrix inversion failure ---
    if (is.null(mdl$inv_success) || !mdl$inv_success) {
      debug_dir <- "debug_matrices"
      if (!dir.exists(debug_dir)) {
        dir.create(debug_dir)
      }
      timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
      filename <- file.path(debug_dir, paste0("failed_psi_aLadder_", timestamp, ".rds"))
      
      cat(sprintf("\n  [Debug] Matrix inversion failed. Saving problematic psi matrix to '%s'\n", filename))
      saveRDS(mdl$psi, file = filename)
      
      stop("Matrix inversion failed in C++ (aIntModel). Problematic psi matrix has been saved.")
    }
    
    # --- 7. Store necessary data and scaling parameters in the model object ---
    mdl$data <- list(y_scaled = y_scaled, x = x, z = z, xzDim = xzDim) # Store scaled y
    mdl$y_scaling_params <- list(mean = y_mean, sd = y_sd) # Store scaling params
    
  })[3]
  
  mdl$cputime <- cputime
  return(mdl)
}


################################################################################
### Additive Ladder GP Prediction (with Y-Unscaling)
################################################################################
aLadderPred <- function(gpMdl, x0List, y0listTrue = NULL, ei_alpha = 0.5, min_y = NULL) {
  
  # --- 1. Retrieve training and scaling parameters from the model object ---
  trainMaxDim <- ncol(gpMdl$data$x)
  y_mean <- gpMdl$y_scaling_params$mean
  y_sd <- gpMdl$y_scaling_params$sd
  
  cputime <- system.time({
    # --- 2. Prepare new input data x0 ---
    xDims <- sapply(1:length(x0List), function(k) ncol(x0List[[k]]))
    xzDim <- min(xDims)
    x0 <- matrix(0, nrow = 0, ncol = trainMaxDim)
    z0 <- c()
    for (i in 1:length(x0List)) {
      n <- nrow(x0List[[i]])
      current_cols <- ncol(x0List[[i]])
      
      if (current_cols < trainMaxDim) {
        padding <- matrix(-1, nrow = n, ncol = trainMaxDim - current_cols)
        x_padded <- cbind(x0List[[i]], padding)
      } else if (current_cols > trainMaxDim) {
        x_padded <- x0List[[i]][, 1:trainMaxDim, drop = FALSE]
      } else {
        x_padded <- x0List[[i]]
      }
      x0 <- rbind(x0, x_padded)
      z0 <- c(z0, rep(ncol(x_padded)/xzDim, n))
    }
    
    # --- 3. [CRITICAL] Adjust min_y for EI calculation to the scaled space ---
    # If min_y is provided in original scale, we need to scale it down.
    min_y_scaled <- if (!is.null(min_y)) (min_y - y_mean) / y_sd else min(gpMdl$data$y_scaled)
    
    # --- 4. Call C++ for prediction in the scaled space ---
    # Note: gpMdl$data$y is now gpMdl$data$y_scaled
    pred_scaled <- aIntPred(x0, z0, gpMdl$data$y_scaled, gpMdl$data$x, gpMdl$data$z, gpMdl$data$xzDim,
                            gpMdl$vecParams, gpMdl$invPsi, gpMdl$mu, ei_alpha, min_y_scaled, logParam = TRUE)
    
    # --- 5. [CRITICAL FIX] Unscale the prediction results back to original scale ---
    pred_unscaled <- list()
    pred_unscaled$pred <- pred_scaled$pred * y_sd + y_mean
    pred_unscaled$mse <- pred_scaled$mse * (y_sd^2) # Variance unscales with sd^2
    
    # Also unscale EI-related components for correct interpretation
    pred_unscaled$ei <- pred_scaled$ei * y_sd # EI has the same unit as Y
    pred_unscaled$improvement <- pred_scaled$improvement * y_sd
    pred_unscaled$uncertainty <- pred_scaled$uncertainty * y_sd
    
  })[3]
  
  # --- 6. Return the unscaled results ---
  pred_unscaled$y_true <- if(!is.null(y0listTrue)) unlist(y0listTrue) else "Empty"
  return(pred_unscaled)
}