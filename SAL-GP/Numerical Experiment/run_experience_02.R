# ==============================================================================
# Experiment 2 (Updated): 4-Model Imbalance Test (aLadder, mLadder-O/N, SAL-GP)
# ==============================================================================

# --- 1. 初始設定與載入套件 ---
rm(list = ls())
library(SFDesign)
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
num_simulations <- 10
experiment_name <- "Experiment_2_4-Model_Imbalance"

output_base_path <- "C:/Users/USER/Desktop/PYCProfessor/Ladder GP New2025/Result"
timestamp <- format(Sys.time(), "%Y-%m-%d_%H-%M-%S")
output_dir <- file.path(output_base_path, paste0(experiment_name, "_", timestamp))
dir.create(output_dir, showWarnings = FALSE)
output_filename <- file.path(output_dir, "results_exp2_4-model.csv")


# --- 3. 實驗設定 (實驗二 - 不平衡性測試) ---
P_DATA_EXP2 <- c(3, 6, 9)
N_TEST_EXP2 <- c(10, 10, 10)

training_scenarios <- list(
  balanced = list(name = "Balanced (30:30:30)", n_train = c(30, 30, 30)),
  front_heavy = list(name = "Front-Heavy (60:15:15)", n_train = c(60, 15, 15)),
  back_heavy = list(name = "Back-Heavy (15:15:60)", n_train = c(15, 15, 60))
)

# 固定的參數範圍
BEST_CONTI_RANGE <- c(-3, 0.9)
BEST_VAR_RANGE <- c(-3, 0.9)
BEST_CATEG_RANGE <- c(-3, 0.9)
BEST_ATTN_RANGE <- c(-5.0, 5.0) # SAL-GP 的專屬參數

# ***** ↓↓↓ 修改：模型列表，加入 SAL-GP ↓↓↓ *****
model_list <- list(
  aLadder = list(name = "aLadder"),
  mLadder_o = list(name = "mLadder-Ordinal"),
  mLadder_n = list(name = "mLadder-Nominal"),
  SAL_GP = list(name = "SAL-GP") # 新增參賽者
)

# 準備結果儲存的 data frame
results_df <- data.frame(
  Simulation_ID = integer(), Scenario = character(), Model_Name = character(),
  RMSE = numeric(), NRMSE = numeric(), MAPE = numeric(), R_Squared = numeric(),    
  CPU_Time_sec = numeric(), stringsAsFactors = FALSE
)

# 準備詳細結果輸出的資料夾
detail_output_dir <- file.path(output_dir, "detailed_predictions")
dir.create(detail_output_dir, showWarnings = FALSE)


# --- 4. 主循環：運行多次模擬 ---
for (i in 1:num_simulations) {
  
  set.seed(4321 + i) # 為每次模擬設定不同的種子
  
  # 測試集在每次模擬中固定，以確保所有情境和模型都基於相同的測試目標
  x0List <- lapply(1:length(P_DATA_EXP2), function(k) maxproLHD(N_TEST_EXP2[k], P_DATA_EXP2[k])$design)
  y0List <- lapply(1:length(x0List), function(k) apply(x0List[[k]], 1, Rastrigin))
  total_y_true_test <- unlist(y0List)
  
  for (scenario in training_scenarios) {
    
    cat(sprintf("\n--- Sim #%d, Scenario: %s ---\n", i, scenario$name))
    
    # 訓練集根據不同情境生成
    N_TRAIN_CURRENT <- scenario$n_train
    xList <- lapply(1:length(P_DATA_EXP2), function(k) maxproLHD(N_TRAIN_CURRENT[k], P_DATA_EXP2[k])$design)
    yList <- lapply(1:length(xList), function(k) apply(xList[[k]], 1, Rastrigin))
    
    # ***** ↓↓↓ 新增：為第一次模擬的每個情境，建立一個空的 data frame 來收集預測值 ↓↓↓ *****
    if (i == 1) {
      predictions_for_scenario <- data.frame(Y_True = total_y_true_test)
    }
    
    for (model_setting in model_list) {
      model_name <- model_setting$name
      cat(sprintf("  Testing model: %s\n", model_name))
      
      current_rmse <- NA; current_nrmse <- NA; current_mape <- NA; current_rsq <- NA; current_cputime <- NA
      
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
        # ***** ↓↓↓ 新增：將當前模型的預測結果，作為新的一欄加入 data frame ↓↓↓ *****
        if (i == 1) {
          predictions_for_scenario[[paste0("Pred_", model_name)]] <- y_predicted
        }
        
        Error <- y_predicted - total_y_true_test
        
        # 計算所有指標
        current_rmse <- sqrt(mean(Error^2))
        y_range <- max(total_y_true_test) - min(total_y_true_test)
        current_nrmse <- current_rmse / y_range
        current_mape <- mean(abs(Error / (total_y_true_test + 1e-9))) * 100
        current_rsq <- 1 - (sum(Error^2) / sum((total_y_true_test - mean(total_y_true_test))^2))
        current_cputime <- mdl$cputime
        
        cat(sprintf("    -> SUCCESS: RMSE=%.4f, NRMSE=%.2f%%, MAPE=%.2f%%, R2=%.4f, CPU=%.2f sec\n", 
                    current_rmse, current_nrmse*100, current_mape, current_rsq, current_cputime)) 
        
      }, error = function(e) {
        cat(sprintf("    -> FAILED: %s\n", e$message))
      })
      
      results_df <- rbind(results_df, data.frame(
        Simulation_ID = i, Scenario = scenario$name, Model_Name = model_name,
        RMSE = current_rmse, NRMSE = current_nrmse, MAPE = current_mape, R_Squared = current_rsq,     
        CPU_Time_sec = current_cputime
      ))
    }
    # ***** ↓↓↓ 新增：在跑完一個情境的所有模型後，將收集到的詳細預測結果寫入 CSV 檔案 ↓↓↓ *****
    if (i == 1) {
      # 為每個預測欄位新增一個對應的誤差欄位
      for(m_name in model_list) {
        pred_col <- paste0("Pred_", m_name$name)
        error_col <- paste0("Error_", m_name$name)
        if(pred_col %in% names(predictions_for_scenario)) {
          predictions_for_scenario[[error_col]] <- predictions_for_scenario$Y_True - predictions_for_scenario[[pred_col]]
        }
      }
      
      # 產生檔案名稱並寫入
      detail_filename <- paste0("Sim1_Predictions_", gsub("[^A-Za-z0-9]", "_", scenario$name), ".csv")
      detail_filepath <- file.path(detail_output_dir, detail_filename)
      write.csv(predictions_for_scenario, detail_filepath, row.names = FALSE)
      cat(sprintf("  -> Detailed predictions for '%s' saved to CSV.\n", scenario$name))
    }
    
  }
}

# --- 5. 儲存與分析結果 (這部分程式碼不變，會自動處理) ---
write.csv(results_df, output_filename, row.names = FALSE)
cat(sprintf("\nExperiment 2 finished. Results saved to '%s'.\n", output_filename))

summary_df <- results_df %>%
  group_by(Scenario, Model_Name) %>%
  summarise(
    across(c(RMSE, NRMSE, MAPE, R_Squared, CPU_Time_sec),
           list(Mean = ~mean(.x, na.rm = TRUE), SD = ~sd(.x, na.rm = TRUE))),
    Total_Successes = sum(!is.na(RMSE))
  )

summary_filename <- file.path(output_dir, "summary_exp2.csv")
write.csv(summary_df, summary_filename, row.names = FALSE)

cat("\n--- Final Summary (Experiment 2) ---\n")
print(as.data.frame(summary_df))


# --- 6. 視覺化結果 (這部分程式碼不變，會自動處理) ---
plots_list <- list(
  RMSE = list(title = "RMSE under Different Training Scenarios", y_label = "Root Mean Squared Error (RMSE)"),
  NRMSE = list(title = "NRMSE under Different Training Scenarios", y_label = "Normalized RMSE (%)"),
  MAPE = list(title = "MAPE under Different Training Scenarios", y_label = "Mean Absolute Percentage Error (%)"),
  R_Squared = list(title = "R-Squared under Different Training Scenarios", y_label = "R-Squared (R²)")
)

for (metric in names(plots_list)) {
  plot_filename <- file.path(output_dir, paste0("boxplot_", tolower(metric), "_exp2.png"))
  
  p <- ggplot(results_df, aes_string(x = "Model_Name", y = metric, fill = "Model_Name")) +
    geom_boxplot() +
    facet_wrap(~ Scenario, scales = "free_y") + # 使用 free_y 讓 Y 軸各自調整
    labs(title = plots_list[[metric]]$title,
         subtitle = paste("Based on", num_simulations, "simulations"),
         x = "Model Type", y = plots_list[[metric]]$y_label) +
    theme_bw(base_size = 12) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5),
          axis.text.x = element_text(angle = 25, hjust = 1),
          legend.position = "none")
  
  if (metric %in% c("NRMSE", "MAPE")) {
    p <- p + scale_y_continuous(labels = scales::percent_format(scale = if(metric == "NRMSE") 100 else 1))
  }
  
  print(p)
  ggsave(plot_filename, plot = p, width = 12, height = 7, dpi = 300)
  cat(sprintf("\nBoxplot for %s saved to '%s'.\n", metric, plot_filename))
}