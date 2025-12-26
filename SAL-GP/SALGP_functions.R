# -----------------------------------------------------------------------------
# R Functions for the SAL-GP (Softmax Attention Ladder Gaussian Process) Model
# -----------------------------------------------------------------------------

################################################################################
### SAL-GP Fit (with Y-Scaling)
################################################################################
SALGPFit <- function(yList, xList, 
                     contiParLogRange = c(-6.5, 1.5), # Range for theta (θ 的範圍)
                     attnParLogRange = c(-5.0, 5.0),  # Range for logits (o 的範圍)
                     nSwarm = 64, maxIter = 200, psoType = "basic", nugget = 1e-6, optVerbose = TRUE) {
  # 對應 SAL 說明文件中 Step 2: Attention Weights 裡的 logits (也就是 oz 和 ouv)。這些是注意力權重 w 的原始數值，在通過 Softmax 函數之前。
  # optVerbose: 一個布林值 (TRUE/FALSE)，決定是否在執行最佳化時印出詳細過程。
  
  cputime <- system.time({
    # --- 1. Prepare X and Z data (consistent with other models) ---
    xDims <- sapply(1:length(xList), function(k) ncol(xList[[k]]))
    #xDims <- sapply(...)：計算 xList 中每個資料集（矩陣）的欄位數（維度）。
    #function(k) 給一個數字 k，就回傳 xList 列表中第 k 個元素的欄位數
    #Simplified Apply (簡化應用) 的縮寫。智慧型的迴圈。接收一個列表或向量 (第一個參數)。 接收一個函式 (第二個參數)。 依序將列表/向量中的每一個元素，丟進那個函式中去運算。 將所有運算結果收集起來，並盡可能地簡化成最方便的格式 (例如，如果結果都是單一數字，就簡化成一個向量)。
    
    xzDim <- min(xDims)
    #xzDim <- min(xDims)：找到所有資料集中最小的維度，作為一個階段的基本維度。

    x <- matrix(0, nrow = 0, ncol = max(xDims))
    #x <- matrix(...)：初始化一個空的矩陣 x，它的欄位數等於所有資料集中最大的維度
    y_raw <- z <- dimCheck <- c()
    #y_raw <- z <- dimCheck <- c()：初始化三個空的向量，y_raw 用來存放所有 y 值，z 用來記錄每個樣本屬於第幾個階段，dimCheck 用來做檢查。
    for (i in 1:length(yList)) {
      n <- length(yList[[i]]) #取得第 i 個資料集的樣本數。
      x <- rbind(x, cbind(xList[[i]], matrix(-1, n, ncol(x) - ncol(xList[[i]]))))
      #處理階梯狀資料。將維度較小的資料集用 -1 進行填充，使其維度與最寬的資料集一致，然後將所有資料集合併成一個大的矩陣 x。
      y_raw <- c(y_raw, yList[[i]]) #將所有 yList 中的 y 值合併成一個單一的向量 y_raw。
      z <- c(z, rep(xDims[i]/xzDim, n)) #計算第 i 個資料集的維度是基本維度的幾倍，並將這個階段數記錄在 z 向量中。例如，如果基本維度是 2，某個資料集有 6 個欄位，那麼它的階段數就是 3。
      dimCheck[i] <- xDims[i] %% xzDim
    }
    stopifnot(all(dimCheck == 0))
    #檢查確保所有資料集的維度都是基本維度 xzDim 的整數倍。如果不是，程式會報錯停止
    
    # --- 2. Standardize the target variable Y for numerical stability ---
    y_mean <- mean(y_raw)
    y_sd <- sd(y_raw)
    if (is.na(y_sd) || y_sd < 1e-9) { y_sd <- 1 }
    #如果所有 y 值都一樣，標準差會是 0 或非常小，這在除法中會產生問題。這種情況下，將標準差設為 1。
    y_scaled <- (y_raw - y_mean) / y_sd
    #高斯過程模型的一個標準步驟。標準化 y 值可以讓模型的最佳化過程更加穩定和快速。會在最後預測時，再將結果「反標準化」回原來的尺度。

    # --- 3. Set up PSO optimization parameters for SAL-GP ---
    nContiPar <- ncol(x) # Number of theta parameters
    max_stages <- max(z)
    nMainLogits <- max_stages
    nInterLogits <- 0.5 * max_stages * (max_stages - 1) #交互作用 logits (ouv) 的數量。這是一個組合計算 C(Z, 2)，也就是從 Z 個階段中任選兩個的組合數。
    nAttnPar <- nMainLogits + nInterLogits # Total number of attention logits
    
    cat("--- [SAL-GP Model] Parameter Count Breakdown ---\n")
    cat(sprintf("Number of theta parameters (nContiPar): %d\n", nContiPar))
    cat(sprintf("Number of attention logits (nAttnPar): %d (%d main + %d inter)\n", nAttnPar, nMainLogits, nInterLogits))
    cat(sprintf("Total parameters to optimize: %d\n", nContiPar + nAttnPar))
    cat("-------------------------------------------------\n")
    
    low_bound <- c(rep(min(contiParLogRange), nContiPar),
                   rep(min(attnParLogRange), nAttnPar))
    upp_bound <- c(rep(max(contiParLogRange), nContiPar),
                   rep(max(attnParLogRange), nAttnPar))
    
    alg_setting <- getPSOInfo(nSwarm = nSwarm, maxIter = maxIter, psoType = psoType)
    #輔助函式，它會把 nSwarm 等設定打包成 PSO 演算法看得懂的格式。
    
    # --- 4. Run PSO using the new SALGP C++ objective function ---
    res <- globpso(objFunc = SALGPObjCpp, lower = low_bound, upper = upp_bound,
                   PSO_INFO = alg_setting, verbose = optVerbose,
                   y = y_scaled, x = x, z = z, xzDim = xzDim, nugget = nugget, logParam = TRUE)
    #objFunc = SALGPObjCpp：是目標函式 (Objective Function)。 
    #這是 PSO 演算法試圖去最小化的函式。它被設定為 SALGPObjCpp，實際的數學計算（計算高斯過程的負對數概似值）是在一個 C++ 函式中完成的。
    
    # --- 5. Build the final model using optimized parameters ---
    mdl <- SALGPModel(param = res$par, y = y_scaled, x = x, z = z, xzDim = xzDim, nugget = nugget, logParam = TRUE)
    #呼叫另一個輔助函式 SALGPModel。這個函式會使用 PSO 找到的最佳參數 res$par 來建立最終的模型物件。
    #在 SALGPModel 內部，它會用最佳參數再次計算相關矩陣 Ψ，並計算 Ψ 的反矩陣 (invPsi)。利用反矩陣在後續做預測
    
    # --- 6. Store necessary data and scaling parameters in the model object ---
    mdl$data <- list(y_scaled = y_scaled, x = x, z = z, xzDim = xzDim)
    mdl$y_scaling_params <- list(mean = y_mean, sd = y_sd)
    
  })[3]
  
  mdl$cputime <- cputime # 將執行時間存入模型
  return(mdl) # 回傳完整的模型物件
  #這個物件打包了所有關於這個訓練好的模型的資訊：最佳參數、處理過的訓練資料、標準化資訊、計算時間等
}

################################################################################
### SAL-GP Prediction (with Y-Unscaling)
################################################################################
SALGPPred <- function(gpMdl, x0List, y0listTrue = NULL) {
  #gpMdl`: 上一個函式 `SALGPFit` 訓練完回傳的模型物件。
  #`y0listTrue`: 這是選填的。如果有新資料的真實 y 值，可以傳入用來比較預測的準確度。

  # --- 1. Retrieve training and scaling parameters from the model object ---
  trainMaxDim <- ncol(gpMdl$data$x) #取得模型在訓練時所使用的標準資料寬度 (最大維度)。
  y_mean <- gpMdl$y_scaling_params$mean #取得原始訓練資料 y 值的平均數
  y_sd <- gpMdl$y_scaling_params$sd
  
  cputime <- system.time({
    # --- 2. Prepare new input data x0 ---
    xzDim <- gpMdl$data$xzDim # Use xzDim from the trained model
    x0 <- matrix(0, nrow = 0, ncol = trainMaxDim) 
    #建立一個空的矩陣 x0，目前沒有任何橫列 (nrow = 0)，但欄位寬度已經設定為從模型中讀取到的標準寬度 trainMaxDim。
    z0 <- c()
    
    for (i in 1:length(x0List)) {
      n <- nrow(x0List[[i]])
      # i會是我的階段數， n會是各階段的筆數
      current_cols <- ncol(x0List[[i]])
      
      # Pad or truncate x0 to match training dimensions
      if (current_cols < trainMaxDim) {
        padding <- matrix(-1, nrow = n, ncol = trainMaxDim - current_cols)
        x_padded <- cbind(x0List[[i]], padding)
      } else if (current_cols > trainMaxDim) {
        x_padded <- x0List[[i]][, 1:trainMaxDim, drop = FALSE]
        # 1:trainMaxDim: 橫列部分留空表示「選擇所有橫列」，欄位部分 1:trainMaxDim 表示「只選擇從第 1 欄到第 trainMaxDim 欄」
      } else {
        x_padded <- x0List[[i]]
      }
      x0 <- rbind(x0, x_padded)
      z0 <- c(z0, rep(ncol(x0List[[i]])/xzDim, n)) # 根據新資料的原始寬度計算其階段數，並將這些階段數記錄到 z0 向量中。
      #為這 10 筆來自同一個矩陣的樣本，全部貼上它們共同的標籤——「階段 X」
    }
    
    # --- 3. Call C++ for prediction in the scaled space ---
    pred_scaled <- SALGPPredCpp(x0, z0, gpMdl$data$y_scaled, gpMdl$data$x, gpMdl$data$z, gpMdl$data$xzDim,
                                gpMdl$vecParams, gpMdl$invPsi, gpMdl$mu, logParam = TRUE)
    #結果包含預測的平均值和均方誤差，但它們都是在標準化後的空間中的數值。
    
    # --- 4. Unscale the prediction results back to original scale ---
    pred_unscaled <- list()
    pred_unscaled$pred <- pred_scaled$pred * y_sd + y_mean
    pred_unscaled$mse <- pred_scaled$mse * (y_sd^2)
    
  })[3]
  
  pred_unscaled$y_true <- if(!is.null(y0listTrue)) unlist(y0listTrue) else "Empty"
  #如果有提供真實的 y 值，就整理好放進結果列表中；如果沒有，就記錄為空
  return(pred_unscaled)
  #pred_unscaled 這個列表。這個列表裡包含了最終預測值、均方誤差，以及可能有的真實 y 值
}