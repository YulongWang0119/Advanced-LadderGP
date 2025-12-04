
rm(list = ls())

# --- 1. 初始設定與載入套件 ---
library(SFDesign)
library(ggplot2)
library(dplyr)
library(reshape2) 

# 引入所有需要的模型函數 
sourceCpp('src/cppFunc.cpp')
source('C:/Users/USER/Desktop/PYCProfessor/Ladder GP New2025/ladderGP-main/R/testfunction.R')
source('C:/Users/USER/Desktop/PYCProfessor/Ladder GP New2025/ladderGP-main/R/aladderGP_NEW.R')
source('C:/Users/USER/Desktop/PYCProfessor/Ladder GP New2025/ladderGP-main/R/mladderGP.R')
# aLadderFit <- function(yList, xList, 
#                        contiParLogRange = c(-6.5, 1.5), 
#                        varParLogRange = c(-6.5, 1.5),
#                        nSwarm = 64, maxIter = 200, psoType = "basic", nugget = 1e-6, optVerbose = TRUE) {
#   
#   cputime <- system.time({
#     xDims <- sapply(1:length(xList), function(k) ncol(xList[[k]]))
#     xzDim <- min(xDims)
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
#     
#     low_bound <- c(rep(min(contiParLogRange), nContiPar),
#                    rep(min(varParLogRange), nVarPar))
#     upp_bound <- c(rep(max(contiParLogRange), nContiPar),
#                    rep(max(varParLogRange), nVarPar))
#     
#     alg_setting <- getPSOInfo(nSwarm = nSwarm, maxIter = maxIter, psoType = psoType)
#     
#     res <- globpso(objFunc = aIntObjCpp, lower = low_bound, upper = upp_bound,
#                    PSO_INFO = alg_setting, verbose = optVerbose,
#                    y = y, x = x, z = z, xzDim = xzDim, nugget = nugget, logParam = TRUE)
#     
#     mdl <- aIntModel(param = res$par, y = y, x = x, z = z, xzDim = xzDim, nugget = nugget, logParam = TRUE)
#     
#  
#     # 檢查 C++ 回傳的求逆成功標記
#     if (is.null(mdl$inv_success) || !mdl$inv_success) {
#       debug_dir <- "debug_matrices"
#       if (!dir.exists(debug_dir)) {
#         dir.create(debug_dir)
#       }
#       timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
#       filename <- file.path(debug_dir, paste0("failed_psi_aLadder_", timestamp, ".rds"))
#       
#       cat(sprintf("\n  [Debug] Matrix inversion failed. Saving problematic psi matrix to '%s'\n", filename))
#       saveRDS(mdl$psi, file = filename)
#       
#       # 拋出一個明確的 R 錯誤，讓實驗主迴圈的 tryCatch 接住
#       stop("Matrix inversion failed in C++. Problematic psi matrix has been saved.")
#     }
#     
#     mdl$data <- list(y = y, x = x, z = z, xzDim = xzDim)
#   })[3]
#   
#   mdl$cputime <- cputime
#   return(mdl)
# }
# 
# aLadderPred <- function(gpMdl, x0List, y0listTrue = NULL, ei_alpha = 0.5, min_y = NULL) {
#   
#   trainMaxDim <- ncol(gpMdl$data$x) # 取得訓練時的最大維度 (例如 12)
#   
#   cputime <- system.time({
#     xDims <- sapply(1:length(x0List), function(k) ncol(x0List[[k]]))
#     xzDim <- min(xDims)
#     
#     # 初始化 x0
#     x0 <- matrix(0, nrow = 0, ncol = trainMaxDim)
#     z0 <- c()
#     
#     for (i in 1:length(x0List)) {
#       n <- nrow(x0List[[i]])
#       current_cols <- ncol(x0List[[i]])
#       
#       # 修正：強制裁切或補齊，確保維度與訓練完全一致
#       if (current_cols < trainMaxDim) {
#         # 太短：補 -1
#         padding <- matrix(-1, nrow = n, ncol = trainMaxDim - current_cols)
#         x_padded <- cbind(x0List[[i]], padding)
#       } else if (current_cols > trainMaxDim) {
#         # 太長：切掉 (這就是 Train 4 vs Test 5 崩潰的原因)
#         x_padded <- x0List[[i]][, 1:trainMaxDim, drop = FALSE]
#       } else {
#         # 剛好
#         x_padded <- x0List[[i]]
#       }
#       
#       x0 <- rbind(x0, x_padded)
#       z0 <- c(z0, rep(ncol(x_padded)/xzDim, n))
#     }
#     
#     if (is.null(min_y)) { min_y <- min(gpMdl$data$y) }
#     
#     # 呼叫 C++
#     pred <- aIntPred(x0, z0, gpMdl$data$y, gpMdl$data$x, gpMdl$data$z, gpMdl$data$xzDim,
#                      gpMdl$vecParams, gpMdl$invPsi, gpMdl$mu, ei_alpha, min_y, logParam = TRUE)
#     
#   })[3]
#   
#   pred$y_true <- if(!is.null(y0listTrue)) unlist(y0listTrue) else "Empty"
#   return(pred)
# }

# --- 2. 實驗參數 ---
num_simulations <- 1 # <--- 正式跑的時候可以設高一點，測試時設 3-5 即可
experiment_name <- "Experiment_3_Stability_Diagnosis" 

output_base_path <- "C:/Users/USER/Desktop/PYCProfessor/Ladder GP New2025/Result"
timestamp <- format(Sys.time(), "%Y-%m-%d_%H-%M-%S")
output_dir <- file.path(output_base_path, paste0(experiment_name, "_", timestamp))
dir.create(output_dir, showWarnings = FALSE)
output_filename <- file.path(output_dir, "results_exp3.csv")

# --- 3. 實驗設定 ---

training_scenarios_exp3 <- list(
  train_3_stages = list(name = "Train with 3 Stages", p_data = c(3, 6, 9), n_train = c(10, 10, 10)),
  train_4_stages = list(name = "Train with 4 Stages", p_data = c(3, 6, 9, 12), n_train = c(10, 10, 10, 10)),
  train_5_stages = list(name = "Train with 5 Stages", p_data = c(3, 6, 9, 12, 15), n_train = c(10, 10, 10, 10, 10))
)

N_TEST_EXP3 <- 20
TARGET_STAGE_IDX <- 3 # 預測目標是第 3 階段
TARGET_STAGE_DIM <- training_scenarios_exp3[[1]]$p_data[TARGET_STAGE_IDX] # 自動抓取第3階段的維度 (9維)

BEST_CONTI_RANGE <- c(-3, 0.9)
# BEST_VAR_RANGE <- c(-3, 0.9)
BEST_VAR_RANGE <- c(-3, 0.9)
BEST_CATEG_RANGE <- c(-3, 0.9)
FIXED_NUGGET <- 1e-6 # <--- 固定 Nugget


model_list <- list(
  aLadder = list(name = "aLadder"),
  mLadder_o = list(name = "mLadder-Ordinal"),
  mLadder_n = list(name = "mLadder-Nominal")
)


results_df <- data.frame(
  Simulation_ID = integer(),
  Scenario = character(),
  Model_Name = character(),
  RMSE = numeric(),
  NRMSE = numeric(),
  R_Squared = numeric(),
  CPU_Time_sec = numeric(),
  Nugget_Used = numeric(),       
  stringsAsFactors = FALSE
)

# --- 4. 主循環 ---
for (i in 1:num_simulations) {
  
  max_dim_needed <- max(sapply(training_scenarios_exp3, function(s) max(s$p_data)))
  master_test_data <- maxproLHD(N_TEST_EXP3, max_dim_needed)$design
  raw_test_y_true <- apply(master_test_data[, 1:TARGET_STAGE_DIM], 1, Rastrigin)
  
  for (scenario in training_scenarios_exp3) {
    cat(sprintf("\n--- Sim #%d, Scenario: %s ---\n", i, scenario$name))
    
    xList <- lapply(1:length(scenario$p_data), function(k) maxproLHD(scenario$n_train[k], scenario$p_data[k])$design)
    yList <- lapply(1:length(xList), function(k) apply(xList[[k]], 1, Rastrigin))
    
    x0List_exp3 <- lapply(scenario$p_data, function(p) master_test_data[, 1:p, drop = FALSE])
    y0List_exp3 <- lapply(x0List_exp3, function(x) rep(0, nrow(x))) # 產生對應長度的假Y
    y0List_exp3[[TARGET_STAGE_IDX]] <- raw_test_y_true
    
    for (model_setting in model_list) {
      model_name <- model_setting$name
      cat(sprintf("  Testing model: %s\n", model_name))
      
      current_rmse <- NA; current_nrmse <- NA; current_rsq <- NA
      current_cputime <- NA; mdl <- NULL
      
      tryCatch({
        # 1. 模型訓練 (強制使用固定的 Nugget)
        if (model_name == "aLadder") {
          mdl <- aLadderFit(yList, xList, contiParLogRange = BEST_CONTI_RANGE, varParLogRange = BEST_VAR_RANGE, nugget = FIXED_NUGGET, optVerbose = FALSE)
        } else if (model_name == "mLadder-Ordinal") {
          mdl <- mLadderFit(yList, xList, zType = "o", contiParLogRange = BEST_CONTI_RANGE, nugget = FIXED_NUGGET, optVerbose = FALSE)
        } else if (model_name == "mLadder-Nominal") {
          mdl <- mLadderFit(yList, xList, zType = "n", contiParLogRange = BEST_CONTI_RANGE, categParLogRange = BEST_CATEG_RANGE, nugget = FIXED_NUGGET, optVerbose = FALSE)
        }
        
        # 2. 模型預測
        pred_obj <- NULL
        if (model_name == "aLadder") {
          pred_obj <- aLadderPred(mdl, x0List_exp3, y0listTrue = y0List_exp3)
        } else {
          pred_obj <- mLadderPred(mdl, x0List_exp3, y0listTrue = y0List_exp3)
        }
        
        # 修正：手動計算索引來提取預測結果 
        if (is.list(pred_obj$pred)) {
          # 如果 mLadderPred 未來可能回傳 list，這段邏輯也能處理
          final_pred_vector <- pred_obj$pred[[TARGET_STAGE_IDX]]
        } else {
          # 處理扁平化向量的情況
          # 計算前幾個階段總共有多少測試點
          points_before_target <- 0
          if (TARGET_STAGE_IDX > 1) {
            points_before_target <- sum(sapply(x0List_exp3[1:(TARGET_STAGE_IDX - 1)], nrow))
          }
          
          # 計算目標階段的測試點數量
          num_points_in_target <- nrow(x0List_exp3[[TARGET_STAGE_IDX]])
          
          # 計算在預測向量中的起始和結束索引
          start_index <- points_before_target + 1
          end_index <- points_before_target + num_points_in_target
          
          # 從大向量中提取出目標階段的預測值
          final_pred_vector <- pred_obj$pred[start_index:end_index]
        }
        # <--- 修正結束 ---
        
        # 4. 計算所有指標
        Error <- final_pred_vector - raw_test_y_true
        current_rmse <- sqrt(mean(Error^2))
        
        y_range <- max(raw_test_y_true) - min(raw_test_y_true)
        current_nrmse <- if (y_range > 0) current_rmse / y_range else 0
        
        SS_tot <- sum((raw_test_y_true - mean(raw_test_y_true))^2)
        current_rsq <- if (SS_tot > 0) 1 - (sum(Error^2) / SS_tot) else 1
        
        current_cputime <- mdl$cputime
        
        cat(sprintf("    -> SUCCESS: RMSE=%.4f, NRMSE=%.2f%%, R2=%.4f, CPU=%.2fs\n", 
                    current_rmse, current_nrmse*100, current_rsq, current_cputime))
        
      }, error = function(e) {
        # 失敗診斷的核心部分 
        cat(sprintf("    -> FAILED: %s\n", e$message))
        
        psi_matrix_name <- if (model_name == "aLadder") "psi" else "psi" # mLadder也叫psi
        
        if (!is.null(mdl) && !is.null(mdl[[psi_matrix_name]])) {
          cat("    -> Capturing diagnostic info for psi matrix...\n")
          psi <- mdl[[psi_matrix_name]]
          failed_case_name <- paste0("FAIL_Sim", i, "_", gsub("[^A-Za-z0-9]", "", scenario$name), "_", model_name)
          
          # 儲存原始 psi 矩陣
          saveRDS(psi, file = file.path(output_dir, paste0(failed_case_name, "_psi.Rds")))
          
          # 產生並儲存熱圖
          p <- ggplot(melt(psi), aes(Var1, Var2, fill = value)) +
            geom_tile() + scale_fill_viridis_c() +
            labs(title = paste("Psi Matrix Heatmap -", failed_case_name)) + theme_minimal()
          ggsave(file.path(output_dir, paste0(failed_case_name, "_heatmap.png")), plot = p)
          
          # 計算並儲存診斷日誌
          condition_number <- tryCatch(kappa(psi), error = function(k_e) "Calculation failed")
          log_content <- c(paste("--- Diagnostics for", failed_case_name, "---"),
                           paste("Error:", e$message),
                           paste("Matrix Dimensions:", paste(dim(psi), collapse = " x ")),
                           paste("Condition Number:", format(condition_number, scientific = TRUE, digits = 4)))
          writeLines(log_content, file.path(output_dir, paste0(failed_case_name, "_log.txt")))
        }
      })
      
      results_df <- rbind(results_df, data.frame(
        Simulation_ID = i, Scenario = scenario$name, Model_Name = model_name,
        RMSE = current_rmse, NRMSE = current_nrmse, R_Squared = current_rsq,
        CPU_Time_sec = current_cputime,
        Nugget_Used = if (!is.null(mdl)) mdl$nugget else FIXED_NUGGET
      ))
    }
  }
}

# --- 5. 儲存與分析結果 ---
write.csv(results_df, output_filename, row.names = FALSE)
cat(sprintf("\nExperiment 3 finished. Results saved to '%s'.\n", output_filename))

summary_df <- aggregate(cbind(RMSE, NRMSE, R_Squared, CPU_Time_sec) ~ Scenario + Model_Name, data = results_df, 
                        FUN = function(x) c(mean = mean(x, na.rm = TRUE), sd = sd(x, na.rm = TRUE)))
summary_filename <- file.path(output_dir, "summary_exp3.csv")
write.csv(summary_df, summary_filename, row.names = FALSE)

failure_rate_df <- results_df %>%
  group_by(Scenario, Model_Name) %>%
  summarise(Failure_Count = sum(is.na(RMSE)), .groups = 'drop') %>%
  mutate(Failure_Rate_Percent = (Failure_Count / num_simulations) * 100)
failure_filename <- file.path(output_dir, "failure_rate_exp3.csv")
write.csv(failure_rate_df, failure_filename, row.names = FALSE)

cat("\n--- Final Summary ---\n")
print(summary_df)
cat("\n--- Failure Rate Summary ---\n")
print(failure_rate_df)

# --- 6. 視覺化結果 ---
cat("\nGenerating plots for NRMSE...\n")
plot_filename <- file.path(output_dir, "boxplot_nrmse_exp3.png")
p <- ggplot(results_df, aes(x = Model_Name, y = NRMSE * 100, fill = Model_Name)) +
  geom_boxplot(na.rm = TRUE) +
  facet_wrap(~ Scenario) +
  labs(title = "Model Performance Comparison (Cross-Stage Prediction)",
       subtitle = paste("Target: Stage 3 | Based on", num_simulations, "simulations"),
       x = "Model Type",
       y = "Normalized RMSE (%)") +
  theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none")
ggsave(plot_filename, plot = p, width = 10, height = 7)
cat("Plot saved to:", plot_filename, "\n")



# 範例：載入 3 階段失敗時的 psi 矩陣
psi_3_stages <- readRDS("C:/Users/USER/Desktop/PYCProfessor/Ladder GP New2025/Result/Experiment_3_Stability_Diagnosis_2025-12-03_17-11-04/FAIL_Sim1_Trainwith3Stages_aLadder_psi.Rds")
find_high_corr_pairs <- function(psi_matrix, threshold = 0.99) {
  # 將對角線設為0，因為自己和自己的相關性永遠是1，沒有意義
  diag(psi_matrix) <- 0
  
  # 找出所有大於閾值的元素的索引
  indices <- which(psi_matrix > threshold, arr.ind = TRUE)
  
  if (nrow(indices) == 0) {
    return("No pairs found above the threshold.")
  }
  
  # 將結果整理成 data frame
  high_corr_pairs <- data.frame(
    Point1 = indices[, 1],
    Point2 = indices[, 2],
    Correlation = psi_matrix[indices]
  )
  
  # 移除重複的配對 (例如 5-10 和 10-5 是一樣的)
  unique_pairs <- high_corr_pairs[!duplicated(t(apply(high_corr_pairs[, 1:2], 1, sort))), ]
  
  return(unique_pairs)
}

# 找出 3 階段時的高度相關點對
problem_pairs_3_stages <- find_high_corr_pairs(psi_3_stages, threshold = 0.99)
print("Problematic pairs for 3-stage case:")
print(problem_pairs_3_stages)
