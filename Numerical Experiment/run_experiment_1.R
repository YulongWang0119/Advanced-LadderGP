# --- 1. 初始設定與載入套件 ---
library(SFDesign)
library(openxlsx) 
library(ggplot2) 
library(dplyr)    
library(magrittr)

# 引入所有需要的模型函數 (確認路徑正確)
source('C:/Users/USER/Desktop/PYCProfessor/Ladder GP New2025/ladderGP-main/R/testfunction.R')
source('C:/Users/USER/Desktop/PYCProfessor/Ladder GP New2025/ladderGP-main/R/aladderGP.R')
source('C:/Users/USER/Desktop/PYCProfessor/Ladder GP New2025/ladderGP-main/R/mladderGP.R')

# --- 2. 實驗參數與輸出路徑設定 ---
num_simulations <- 3 # 【實驗一】設定重複 30 次
experiment_name <- "Experiment_1_Benchmark"

# 設定儲存結果的基礎路徑
output_base_path <- "C:/Users/USER/Desktop/PYCProfessor/Ladder GP New2025/Result"

# 建立本次實驗專屬的資料夾
timestamp <- format(Sys.time(), "%Y-%m-%d_%H-%M-%S")
output_dir <- file.path(output_base_path, paste0(experiment_name, "_", timestamp))
dir.create(output_dir, showWarnings = FALSE)

# 完整的輸出檔名路徑
output_filename <- file.path(output_dir, "results_exp1.csv")

# --- 3. 固定實驗設定 ---
# 實驗一鎖定數據設定為 "二階段"
P_DATA_EXP1 <- c(3, 6)
N_TRAIN_EXP1 <- c(30, 30)
N_TEST_EXP1 <- c(30, 30) # 測試點可以稍微多一點

# 實驗一鎖定最佳參數範圍
BEST_CONTI_RANGE <- c(-3, 0.9)
BEST_VAR_RANGE <- c(-3, 0.9)
BEST_CATEG_RANGE <- c(-3, 0.9) # 這是 mLadder-n 的參數

# 實驗一定義要比較的模型列表
model_list <- list(
  aLadder = list(name = "aLadder"),
  mLadder_o = list(name = "mLadder-Ordinal"),
  mLadder_n = list(name = "mLadder-Nominal")
)

# 準備一個空的 data frame 來儲存所有結果 (整體摘要)
results_df <- data.frame(
  Simulation_ID = integer(),
  Model_Name = character(),
  RMSE = numeric(),
  NRMSE = numeric(),
  R_Squared = numeric(),
  CPU_Time_sec = numeric(),
  Nugget_Used = numeric(), 
  stringsAsFactors = FALSE
)
# 儲存 Sim 1 詳細預測結果的結構
sim1_predictions_list <- list() 
y_true_for_sim1 <- NULL 
y_true_stats_for_sim1 <- NULL 

# --- 4. 主循環：運行多次模擬 ---
for (i in 1:num_simulations) {
  
  # 確保 Paired T/T 隨機性
  set.seed(1234 + i) 
  cat(sprintf("\n--- Starting Simulation #%d of %d ---\n", i, num_simulations))
  
  # 在每次模擬開始時重新生成固定設定的隨機數據
  xList <- lapply(1:length(P_DATA_EXP1), function(k) maxproLHD(N_TRAIN_EXP1[k], P_DATA_EXP1[k])$design)
  yList <- lapply(1:length(xList), function(k) {
    apply(xList[[k]], 1, Rastrigin)
  })
  
  x0List <- lapply(1:length(P_DATA_EXP1), function(k) maxproLHD(N_TEST_EXP1[k], P_DATA_EXP1[k])$design)
  y0List <- lapply(1:length(x0List), function(k) {
    apply(x0List[[k]], 1, Rastrigin)
  })
  
  # 聚合測試集的 Y 值，用於計算 NRMSE 和 R-Squared
  total_y_true_test <- unlist(y0List)
  if (i == 1) {
    cat("\n--- 訓練集維度檢查 (Simulation 1) ---\n")
    
    # 檢查 Stage 1 (P_DATA_EXP1[1] = 3)
    X1_train <- xList[[1]]
    cat(sprintf("Stage 1 (X1) 筆數: %d, 維度: %d\n", nrow(X1_train), ncol(X1_train)))
    
    # 檢查 Stage 2 (P_DATA_EXP1[2] = 6)
    X2_train <- xList[[2]]
    cat(sprintf("Stage 2 (X2) 筆數: %d, 維度: %d\n", nrow(X2_train), ncol(X2_train)))
    
    # 檢查總點數
    cat(sprintf("總訓練點數: %d\n", nrow(X1_train) + nrow(X2_train)))
    
    # 可選：將 X1 和 X2 分別寫入 CSV
    write.csv(X1_train, file.path(output_dir, "X_Train_Stage1_Sim1.csv"), row.names = FALSE)
    write.csv(X2_train, file.path(output_dir, "X_Train_Stage2_Sim1.csv"), row.names = FALSE)
    cat("已將訓練集 X1 和 X2 存入 CSV 供確認。\n")
  }
  # **********************************************************
  
  y_range_test <- max(total_y_true_test) - min(total_y_true_test)
  y_mean_test <- mean(total_y_true_test)
  
  # 準備 Sim 1 的 Y_True 數據 (僅在 Sim 1 儲存)
  if (i == 1) {
    y_true_for_sim1 <- total_y_true_test 
    y_true_stats_for_sim1 <- c(Y_Max = max(y_true_for_sim1), Y_Min = min(y_true_for_sim1))
  }
  
  # --- 內部循環：測試不同的模型 ---
  for (model_setting in model_list) {
    
    model_name <- model_setting$name
    cat(sprintf("Testing model: %s\n", model_name))
    
    current_rmse <- NA; current_nrmse <- NA; current_rsq <- NA; current_cputime <- NA;
    
    # 訓練模型 (使用 tryCatch 處理失敗)
    tryCatch({
      mdl <- NULL; pred_obj <- NULL;
      
      # Model Specific Fit & Predict
      if (model_name == "aLadder") {
        
        #增加 aLadder 參數計數診斷 
        # 由於 P_DATA_EXP1 = c(3, 6)，D_Max = 6
        D_Max <- max(P_DATA_EXP1) 
        
        # 1. Lengthscale (Theta) 數量
        nContiPar_AL <- D_Max # 應為 6
        
        # 2. Variance (Sigma) 數量：mLadder 參數邏輯計算
        # Stage 數量 Z = length(P_DATA_EXP1) = 2
        Z_Stages <- length(P_DATA_EXP1) 
        # Var 數量 = Z + 0.5 * Z * (Z - 1) = 2 + 0.5 * 2 * 1 = 3
        nVarPar_AL <- Z_Stages + (0.5 * Z_Stages * (Z_Stages - 1)) 
        
        cat("--- [aLadder Model] Parameter Count Breakdown ---\n")
        cat(sprintf("Number of theta parameters: %d\n", nContiPar_AL))
        cat(sprintf("Number of sigma parameters: %d\n", nVarPar_AL))
        cat(sprintf("Total parameters to optimize: %d\n", nContiPar_AL + nVarPar_AL))
        cat("-----------------------------------------------\n")
        # ************ 診斷結束 ************
        
        mdl <- aLadderFit(yList, xList, contiParLogRange = BEST_CONTI_RANGE, varParLogRange = BEST_VAR_RANGE, nugget = 1e-6, optVerbose = FALSE)
        pred_obj <- aLadderPred(mdl, x0List, y0listTrue = y0List)
        
      } else if (model_name == "mLadder-Ordinal") {
        mdl <- mLadderFit(yList, xList, zType = "o", contiParLogRange = BEST_CONTI_RANGE, nugget = 1e-6, optVerbose = FALSE)
        pred_obj <- mLadderPred(mdl, x0List, y0listTrue = y0List)
      } else if (model_name == "mLadder-Nominal") {
        mdl <- mLadderFit(yList, xList, zType = "n", contiParLogRange = BEST_CONTI_RANGE, categParLogRange = BEST_CATEG_RANGE, nugget = 1e-6, optVerbose = FALSE)
        pred_obj <- mLadderPred(mdl, x0List, y0listTrue = y0List)
      }
      
      y_predicted <- pred_obj$pred
      Error <- y_predicted - total_y_true_test
      
      #計算所有指標 (包含絕對指標 NRMSE 和 R-Squared) 
      current_rmse <- sqrt(mean(Error^2))
      current_nrmse <- current_rmse / y_range_test
      
      SS_res <- sum(Error^2)
      SS_tot <- sum((total_y_true_test - y_mean_test)^2)
      current_rsq <- 1 - (SS_res / SS_tot)
      
      current_cputime <- mdl$cputime
      current_nugget_used <- mdl$nugget # <-- 提取實際使用的 Nugget
      
      cat(sprintf("  -> SUCCESS: RMSE=%.4f, NRMSE=%.2f%%, R-Sq=%.4f, CPU=%.2f sec, Nugget=%.1e\n", 
                  current_rmse, current_nrmse*100, current_rsq, current_cputime, current_nugget_used)) # <-- 打印 Nugget
      
      # --- 儲存 Sim 1 的詳細預測 ---
      if (i == 1) {
        sim1_predictions_list[[paste0("Y_Pred_", model_name)]] <- y_predicted
      }
      
    },  error = function(e) {

      current_nugget_used <- 1e-6 # 即使失敗，也要記錄目標 Nugget 值
      cat(sprintf("  -> FAILED: %s\n", e$message))
      # 失敗時指標為 NA
    })
    
    
    # 將結果加到 data frame
    results_df <- rbind(results_df, data.frame(
      Simulation_ID = i,
      Model_Name = model_name,
      RMSE = current_rmse,
      NRMSE = current_nrmse,
      R_Squared = current_rsq,
      CPU_Time_sec = current_cputime,
      Nugget_Used = current_nugget_used # <-- 儲存 Nugget
    ))
    
  } # 結束內部循環 (模型)
  
  # -寫入 Sim 1 的詳細預測 CSV 
  if (i == 1) {
    
    # 建立基本 dataframe
    detail_df <- data.frame(Y_True = y_true_for_sim1) # 1. 初始化 detail_df
    
    # 2. 逐一添加各模型的預測值和誤差
    for (model_setting in model_list) {
      model_name <- model_setting$name # 確保使用正確的 model_name
      pred_col_name <- paste0("Y_Pred_", model_name)
      error_col_name <- paste0("Error_", model_name)
      
      # 從 sim1_predictions_list 中提取結果
      if (!is.null(sim1_predictions_list[[pred_col_name]])) {
        detail_df[[pred_col_name]] <- sim1_predictions_list[[pred_col_name]]
        detail_df[[error_col_name]] <- detail_df$Y_True - sim1_predictions_list[[pred_col_name]]
      } else {
        detail_df[[pred_col_name]] <- NA
        detail_df[[error_col_name]] <- NA
      }
    }
    
    # 3. 添加總 Y 的統計資訊 (修正：讓所有行都顯示)
    detail_df$Y_Max_True <- y_true_stats_for_sim1["Y_Max"]
    detail_df$Y_Min_True <- y_true_stats_for_sim1["Y_Min"]
    
    detail_filename_full <- file.path(output_dir, "Exp1_Detailed_Prediction_Sim1.csv")
    write.csv(detail_df, detail_filename_full, row.names = FALSE)
  }
  
} # 結束主循環 (模擬次數)

# --- 5. 儲存與分析結果 ---
# 寫入完整的 CSV 檔案 (包含每次模擬的指標)
write.csv(results_df, output_filename, row.names = FALSE)

# 顯示並儲存摘要 (Mean / SD)
summary_df <- results_df %>%
  group_by(Model_Name) %>%
  summarise(
    Mean_RMSE = mean(RMSE, na.rm = TRUE),
    Std_Dev_RMSE = sd(RMSE, na.rm = TRUE),
    Mean_NRMSE = mean(NRMSE, na.rm = TRUE),
    Std_Dev_NRMSE = sd(NRMSE, na.rm = TRUE),
    Mean_R_Squared = mean(R_Squared, na.rm = TRUE),
    Std_Dev_R_Squared = sd(R_Squared, na.rm = TRUE),
    Mean_CPU = mean(CPU_Time_sec, na.rm = TRUE),
    Total_Successes = sum(!is.na(RMSE))
  )

summary_filename_final <- file.path(output_dir, "summary_exp1_final.csv")
write.csv(summary_df, summary_filename_final, row.names = FALSE)

cat("\n--- Final Summary (Standardized Metrics) ---\n")
print(summary_df)
# --- 6. 視覺化結果 (新增的區塊) ---

# 設定圖片儲存路徑
plot_filename <- file.path(output_dir, "boxplot_rmse_exp1.png")

# 使用 ggplot2 繪製箱型圖
rmse_boxplot <- ggplot(results_df, aes(x = Model_Name, y = RMSE, fill = Model_Name)) +
  geom_boxplot() +
  labs(
    title = "Comparison of Model Performance (Experiment 1: 2-Stage Data)",
    subtitle = paste("Based on", num_simulations, "simulations"),
    x = "Model Type",
    y = "Root Mean Squared Error (RMSE)"
  ) +
  theme_minimal(base_size = 14) + 
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"), 
    plot.subtitle = element_text(hjust = 0.5), 
    legend.position = "none" 
  )

# 顯示圖表
print(rmse_boxplot)

# 將圖表儲存為 PNG 檔案
ggsave(plot_filename, plot = rmse_boxplot, width = 8, height = 6, dpi = 300)

cat(sprintf("\nBoxplot saved to '%s'.\n", plot_filename))

# 6. 視覺化結果 

# -----------------------------------------------------
# A. NRMSE 箱形圖
# -----------------------------------------------------
plot_filename_nrmse <- file.path(output_dir, "boxplot_nrmse_exp1.png")

nrmse_boxplot <- ggplot(results_df, aes(x = Model_Name, y = NRMSE, fill = Model_Name)) +
  geom_boxplot() +
  labs(
    title = "Comparison of Model Robustness (NRMSE)",
    subtitle = paste("Based on", num_simulations, "simulations (Lower is Better)"),
    x = "Model Type",
    y = "Normalized Root Mean Squared Error (NRMSE)"
  ) +
  scale_y_continuous(labels = scales::percent) + # 將 Y 軸格式化為百分比
  theme_minimal(base_size = 14) + 
  theme(plot.title = element_text(hjust = 0.5, face = "bold"), 
        plot.subtitle = element_text(hjust = 0.5), 
        legend.position = "none")

# 顯示並儲存 NRMSE 圖表
print(nrmse_boxplot)
ggsave(plot_filename_nrmse, plot = nrmse_boxplot, width = 8, height = 6, dpi = 300)
cat(sprintf("\nNRMSE Boxplot saved to '%s'.\n", plot_filename_nrmse))


# -----------------------------------------------------
# B. R-Squared 箱形圖
# -----------------------------------------------------
plot_filename_rsq <- file.path(output_dir, "boxplot_rsq_exp1.png")

rsq_boxplot <- ggplot(results_df, aes(x = Model_Name, y = R_Squared, fill = Model_Name)) +
  geom_boxplot() +
  labs(
    title = "Comparison of Model Explanatory Power (R-Squared)",
    subtitle = paste("Based on", num_simulations, "simulations (Higher is Better)"),
    x = "Model Type",
    y = "R-Squared"
  ) +
  scale_y_continuous(limits = c(0, 1)) + # 限制 Y 軸範圍在 0 到 1 之間
  theme_minimal(base_size = 14) + 
  theme(plot.title = element_text(hjust = 0.5, face = "bold"), 
        plot.subtitle = element_text(hjust = 0.5), 
        legend.position = "none")

# 顯示並儲存 R-Squared 圖表
print(rsq_boxplot)
ggsave(plot_filename_rsq, plot = rsq_boxplot, width = 8, height = 6, dpi = 300)
cat(sprintf("\nR-Squared Boxplot saved to '%s'.\n", plot_filename_rsq))