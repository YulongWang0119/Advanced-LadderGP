# ==============================================================================
# Experiment 1 (Updated): 4-Model Benchmark (aLadder, mLadder-O, mLadder-N, SAL-GP)
# ==============================================================================

# --- 1. 初始設定與載入套件 ---
rm(list = ls())
library(SFDesign)
library(openxlsx) 
library(ggplot2) 
library(dplyr)    
library(magrittr)

# 引入所有需要的模型函數 (確認路徑正確)
source('C:/Users/USER/Desktop/PYCProfessor/Ladder GP New2025/ladderGP-main/R/testfunction.R')
source('C:/Users/USER/Desktop/PYCProfessor/Ladder GP New2025/ladderGP-main/R/aladderGP_NEW.R')
source('C:/Users/USER/Desktop/PYCProfessor/Ladder GP New2025/ladderGP-main/R/mladderGP.R')

# ***** ↓↓↓ 新增：引入 SAL-GP 函數 ↓↓↓ *****
source('C:/Users/USER/Desktop/PYCProfessor/Ladder GP New2025/ladderGP-main/R/SALGP_functions.R') 

# *** 關鍵 ***: 重新編譯 C++ 程式碼
library(Rcpp)
sourceCpp('C:/Users/USER/Desktop/PYCProfessor/Ladder GP New2025/ladderGP-main/src/cppFunc.cpp')


# --- 2. 實驗參數與輸出路徑設定 ---
num_simulations <- 10 # 設定重複次數
experiment_name <- "Experiment_1_4-Model_Benchmark"

# 設定儲存結果的基礎路徑
output_base_path <- "C:/Users/USER/Desktop/PYCProfessor/Ladder GP New2025/Result"

# 建立本次實驗專屬的資料夾
timestamp <- format(Sys.time(), "%Y-%m-%d_%H-%M-%S")
output_dir <- file.path(output_base_path, paste0(experiment_name, "_", timestamp))
dir.create(output_dir, showWarnings = FALSE)

# 完整的輸出檔名路徑
output_filename <- file.path(output_dir, "results_exp1_4-model.csv")


# --- 3. 固定實驗設定 ---
# 實驗一鎖定數據設定為 "二階段"
P_DATA_EXP1 <- c(3, 6)
N_TRAIN_EXP1 <- c(30, 30)
N_TEST_EXP1 <- c(30, 30)

# 實驗一鎖定最佳參數範圍
BEST_CONTI_RANGE <- c(-3, 0.9)
BEST_VAR_RANGE <- c(-3, 0.9)
BEST_CATEG_RANGE <- c(-3, 0.9)
# ***** ↓↓↓ 新增：SAL-GP 的專屬參數範圍 ↓↓↓ *****
BEST_ATTN_RANGE <- c(-5.0, 5.0) # For logits

# ***** ↓↓↓ 修改：實驗一定義要比較的模型列表，加入 SAL-GP ↓↓↓ *****
model_list <- list(
  aLadder = list(name = "aLadder"),
  mLadder_o = list(name = "mLadder-Ordinal"),
  mLadder_n = list(name = "mLadder-Nominal"),
  SAL_GP = list(name = "SAL-GP") # 新增參賽者
)

# 準備一個空的 data frame 來儲存所有結果
results_df <- data.frame(
  Simulation_ID = integer(), Model_Name = character(),
  RMSE = numeric(), NRMSE = numeric(), MAPE = numeric(), R_Squared = numeric(),
  CPU_Time_sec = numeric(), Nugget_Used = numeric(), 
  stringsAsFactors = FALSE
)

# 儲存 Sim 1 詳細預測結果的結構
sim1_predictions_list <- list() 
y_true_for_sim1 <- NULL 
y_true_stats_for_sim1 <- NULL 


# --- 4. 主循環：運行多次模擬 ---
for (i in 1:num_simulations) {
  
  set.seed(1234 + i) 
  cat(sprintf("\n--- Starting Simulation #%d of %d ---\n", i, num_simulations))
  
  # 在每次模擬開始時重新生成固定設定的隨機數據
  xList <- lapply(1:length(P_DATA_EXP1), function(k) maxproLHD(N_TRAIN_EXP1[k], P_DATA_EXP1[k])$design)
  yList <- lapply(1:length(xList), function(k) apply(xList[[k]], 1, Rastrigin))
  x0List <- lapply(1:length(P_DATA_EXP1), function(k) maxproLHD(N_TEST_EXP1[k], P_DATA_EXP1[k])$design)
  y0List <- lapply(1:length(x0List), function(k) apply(x0List[[k]], 1, Rastrigin))
  
  total_y_true_test <- unlist(y0List)
  y_range_test <- max(total_y_true_test) - min(total_y_true_test)
  y_mean_test <- mean(total_y_true_test)
  
  if (i == 1) {
    y_true_for_sim1 <- total_y_true_test 
    y_true_stats_for_sim1 <- c(Y_Max = max(y_true_for_sim1), Y_Min = min(y_true_for_sim1))
  }
  
  # --- 內部循環：測試不同的模型 ---
  for (model_setting in model_list) {
    
    model_name <- model_setting$name
    cat(sprintf("Testing model: %s\n", model_name))
    
    current_rmse <- NA; current_nrmse <- NA; current_mape <- NA; current_rsq <- NA; current_cputime <- NA;
    
    tryCatch({
      mdl <- NULL; pred_obj <- NULL;
      
      # ***** ↓↓↓ 修改：在主循環中加入 SAL-GP 的訓練邏輯 ↓↓↓ *****
      if (model_name == "aLadder") {
        mdl <- aLadderFit(yList, xList, contiParLogRange = BEST_CONTI_RANGE, varParLogRange = BEST_VAR_RANGE, nugget = 1e-6, optVerbose = FALSE)
        pred_obj <- aLadderPred(mdl, x0List, y0listTrue = y0List)
      } else if (model_name == "mLadder-Ordinal") {
        mdl <- mLadderFit(yList, xList, zType = "o", contiParLogRange = BEST_CONTI_RANGE, nugget = 1e-6, optVerbose = FALSE)
        pred_obj <- mLadderPred(mdl, x0List, y0listTrue = y0List)
      } else if (model_name == "mLadder-Nominal") {
        mdl <- mLadderFit(yList, xList, zType = "n", contiParLogRange = BEST_CONTI_RANGE, categParLogRange = BEST_CATEG_RANGE, nugget = 1e-6, optVerbose = FALSE)
        pred_obj <- mLadderPred(mdl, x0List, y0listTrue = y0List)
      } else if (model_name == "SAL-GP") { # 新增的判斷式
        mdl <- SALGPFit(yList, xList, 
                        contiParLogRange = BEST_CONTI_RANGE, 
                        attnParLogRange = BEST_ATTN_RANGE, 
                        nugget = 1e-6, optVerbose = FALSE)
        pred_obj <- SALGPPred(mdl, x0List, y0listTrue = y0List)
      }
      # ***** ↑↑↑ 修改結束 ↑↑↑ *****
      
      y_predicted <- pred_obj$pred
      Error <- y_predicted - total_y_true_test
      
      # 計算所有指標
      current_rmse <- sqrt(mean(Error^2))
      current_nrmse <- current_rmse / y_range_test
      current_mape <- mean(abs(Error / (total_y_true_test + 1e-9))) * 100
      current_rsq <- 1 - (sum(Error^2) / sum((total_y_true_test - y_mean_test)^2))
      current_cputime <- mdl$cputime
      current_nugget_used <- mdl$nugget # 假設所有模型都有 nugget
      
      cat(sprintf("  -> SUCCESS: RMSE=%.4f, NRMSE=%.2f%%, MAPE=%.2f%%, R-Sq=%.4f, CPU=%.2f sec\n", 
                  current_rmse, current_nrmse*100, current_mape, current_rsq, current_cputime))
      
      if (i == 1) {
        sim1_predictions_list[[paste0("Y_Pred_", model_name)]] <- y_predicted
      }
      
    },  error = function(e) {
      current_nugget_used <- 1e-6
      cat(sprintf("  -> FAILED: %s\n", e$message))
    })
    
    results_df <- rbind(results_df, data.frame(
      Simulation_ID = i, Model_Name = model_name,
      RMSE = current_rmse, NRMSE = current_nrmse, MAPE = current_mape, R_Squared = current_rsq,
      CPU_Time_sec = current_cputime, Nugget_Used = current_nugget_used
    ))
  } 
  
  # --- 這部分程式碼不變，它會自動處理新的 SAL-GP 結果 ---
  if (i == 1) {
    detail_df <- data.frame(Y_True = y_true_for_sim1)
    for (model_setting in model_list) {
      model_name <- model_setting$name
      pred_col_name <- paste0("Y_Pred_", model_name)
      if (!is.null(sim1_predictions_list[[pred_col_name]])) {
        detail_df[[pred_col_name]] <- sim1_predictions_list[[pred_col_name]]
        detail_df[[paste0("Error_", model_name)]] <- detail_df$Y_True - detail_df[[pred_col_name]]
      }
    }
    detail_filename_full <- file.path(output_dir, "Exp1_Detailed_Prediction_Sim1.csv")
    write.csv(detail_df, detail_filename_full, row.names = FALSE)
  }
} 

# --- 5. 儲存與分析結果 (這部分程式碼不變，會自動處理) ---
write.csv(results_df, output_filename, row.names = FALSE)

summary_df <- results_df %>%
  group_by(Model_Name) %>%
  summarise(
    across(c(RMSE, NRMSE, MAPE, R_Squared, CPU_Time_sec),
           list(Mean = ~mean(.x, na.rm = TRUE), SD = ~sd(.x, na.rm = TRUE))),
    Total_Successes = sum(!is.na(RMSE))
  )

summary_filename_final <- file.path(output_dir, "summary_exp1_final.csv")
write.csv(summary_df, summary_filename_final, row.names = FALSE)

cat("\n--- Final Summary (Experiment 1: 4-Model Benchmark) ---\n")
print(as.data.frame(summary_df)) # 用 as.data.frame 讓輸出更整齊

# --- 6. 視覺化結果 (這部分程式碼不變，會自動處理) ---
plots_list <- list(
  RMSE = list(title = "Comparison of Model Performance (RMSE)", y_label = "Root Mean Squared Error (RMSE)"),
  NRMSE = list(title = "Comparison of Model Robustness (NRMSE)", y_label = "Normalized RMSE (%)"),
  MAPE = list(title = "Comparison of Model Performance (MAPE)", y_label = "Mean Absolute Percentage Error (%)"),
  R_Squared = list(title = "Comparison of Model Explanatory Power (R-Squared)", y_label = "R-Squared")
)

for (metric in names(plots_list)) {
  plot_filename <- file.path(output_dir, paste0("boxplot_", tolower(metric), "_exp1.png"))
  
  p <- ggplot(results_df, aes_string(x = "Model_Name", y = metric, fill = "Model_Name")) +
    geom_boxplot() +
    labs(title = plots_list[[metric]]$title,
         subtitle = paste("Based on", num_simulations, "simulations"),
         x = "Model Type", y = plots_list[[metric]]$y_label) +
    theme_minimal(base_size = 14) + 
    theme(plot.title = element_text(hjust = 0.5, face = "bold"), 
          plot.subtitle = element_text(hjust = 0.5), legend.position = "none",
          axis.text.x = element_text(angle = 15, hjust = 1)) # 讓 X 軸標籤稍微傾斜
  
  if (metric %in% c("NRMSE", "MAPE")) {
    p <- p + scale_y_continuous(labels = scales::percent_format(scale = if(metric == "NRMSE") 100 else 1))
  }
  
  print(p)
  ggsave(plot_filename, plot = p, width = 8, height = 6, dpi = 300)
  cat(sprintf("\n%s Boxplot saved to '%s'.\n", metric, plot_filename))
}