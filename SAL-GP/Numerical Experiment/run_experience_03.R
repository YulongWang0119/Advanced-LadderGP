# ==============================================================================
# Experiment 3 (Final Diagnostic Version): 4-Model Stability with Excel Export
# ==============================================================================

# --- 0. 設定正確的工作目錄 ---
# setwd("C:/Users/USER/Desktop/PYCProfessor/Ladder GP New2025/ladderGP-main") # 如果需要的話

# --- 1. 初始設定與載入套件 ---
rm(list = ls())
library(Rcpp)
library(SFDesign)
library(ggplot2)
library(dplyr)
library(reshape2)
library(viridis)
library(openxlsx) # ***** ↓↓↓ 新增：用於寫入 Excel 的套件 ↓↓↓ *****

# 引入所有需要的模型函數 
# 將您提供的路徑貼上，並將反斜線 "\" 改成正斜線 "/"
setwd("C:/Users/USER/Desktop/PYCProfessor/Ladder GP New2025/ladderGP-main")
sourceCpp('src/cppFunc.cpp')
source('C:/Users/USER/Desktop/PYCProfessor/Ladder GP New2025/ladderGP-main/R/testfunction.R')
source('C:/Users/USER/Desktop/PYCProfessor/Ladder GP New2025/ladderGP-main/R/aladderGP_NEW.R')
source('C:/Users/USER/Desktop/PYCProfessor/Ladder GP New2025/ladderGP-main/R/mladderGP.R')
source('C:/Users/USER/Desktop/PYCProfessor/Ladder GP New2025/ladderGP-main/R/SALGP_functions.R') 


# --- 2. 實驗參數 ---
num_simulations <- 10
experiment_name <- "Experiment_3_4-Model_Stability_Diagnosis" 

output_base_path <- "C:/Users/USER/Desktop/PYCProfessor/Ladder GP New2025/Result"
timestamp <- format(Sys.time(), "%Y-%m-%d_%H-%M-%S")
output_dir <- file.path(output_base_path, paste0(experiment_name, "_", timestamp))
dir.create(output_dir, showWarnings = FALSE)
output_filename <- file.path(output_dir, "results_exp3_4-model.csv")

detail_output_dir <- file.path(output_dir, "detailed_predictions")
dir.create(detail_output_dir, showWarnings = FALSE)


# --- 3. 實驗設定 ---
# (這整個區塊的程式碼完全不變)
training_scenarios_exp3 <- list(
  train_3_stages = list(name = "Train with 3 Stages", p_data = c(3, 6, 9), n_train = c(20, 20, 20)),
  train_4_stages = list(name = "Train with 4 Stages", p_data = c(3, 6, 9, 12), n_train = c(20, 20, 20, 20)),
  train_5_stages = list(name = "Train with 5 Stages", p_data = c(3, 6, 9, 12, 15), n_train = c(20, 20, 20, 20, 20))
)
N_TEST_EXP3 <- 30 # 增加測試點數
TARGET_STAGE_IDX <- 3
TARGET_STAGE_DIM <- training_scenarios_exp3[[1]]$p_data[TARGET_STAGE_IDX]
BEST_CONTI_RANGE <- c(-3, 0.9)
BEST_VAR_RANGE <- c(-3, 0.9)
BEST_CATEG_RANGE <- c(-3, 0.9)
BEST_ATTN_RANGE <- c(-5.0, 5.0)
FIXED_NUGGET <- 1e-6
model_list <- list(
  aLadder = list(name = "aLadder"),
  mLadder_o = list(name = "mLadder-Ordinal"),
  mLadder_n = list(name = "mLadder-Nominal"),
  SAL_GP = list(name = "SAL-GP")
)
results_df <- data.frame(
  Simulation_ID = integer(), Scenario = character(), Model_Name = character(),
  RMSE = numeric(), NRMSE = numeric(), MAPE = numeric(), R_Squared = numeric(),
  CPU_Time_sec = numeric(), Nugget_Used = numeric(), stringsAsFactors = FALSE
)

# ***** ↓↓↓ 新增：用來收集所有預測值的資料結構 ↓↓↓ *****
predictions_collector <- list()


# --- 4. 主循環 ---
for (i in 1:num_simulations) {
  
  set.seed(1122 + i)
  
  max_dim_needed <- max(sapply(training_scenarios_exp3, function(s) max(s$p_data)))
  master_test_data <- maxproLHD(N_TEST_EXP3, max_dim_needed)$design
  raw_test_y_true <- apply(master_test_data[, 1:TARGET_STAGE_DIM], 1, Rastrigin)
  
  # ***** ↓↓↓ 新增：將真實 Y 值預先存入收集器 ↓↓↓ *****
  # 我們只儲存第一次模擬的詳細結果
  if (i == 1) {
    predictions_collector[['Y_True']] <- raw_test_y_true
  }
  
  for (scenario in training_scenarios_exp3) {
    cat(sprintf("\n--- Sim #%d, Scenario: %s ---\n", i, scenario$name))
    
    xList <- lapply(1:length(scenario$p_data), function(k) maxproLHD(scenario$n_train[k], scenario$p_data[k])$design)
    yList <- lapply(1:length(xList), function(k) apply(xList[[k]], 1, Rastrigin))
    
    x0List_exp3 <- lapply(scenario$p_data, function(p) master_test_data[, 1:p, drop = FALSE])
    
    for (model_setting in model_list) {
      model_name <- model_setting$name
      cat(sprintf("  Testing model: %s\n", model_name))
      
      current_rmse <- NA; current_nrmse <- NA; current_mape <- NA; current_rsq <- NA
      current_cputime <- NA; mdl <- NULL
      
      tryCatch({
        # 1. 模型訓練 (這部分不變)
        if (model_name == "aLadder") {
          mdl <- aLadderFit(yList, xList, contiParLogRange = BEST_CONTI_RANGE, varParLogRange = BEST_VAR_RANGE, nugget = FIXED_NUGGET, optVerbose = FALSE)
        } else if (model_name == "mLadder-Ordinal") {
          mdl <- mLadderFit(yList, xList, zType = "o", contiParLogRange = BEST_CONTI_RANGE, nugget = FIXED_NUGGET, optVerbose = FALSE)
        } else if (model_name == "mLadder-Nominal") {
          mdl <- mLadderFit(yList, xList, zType = "n", contiParLogRange = BEST_CONTI_RANGE, categParLogRange = BEST_CATEG_RANGE, nugget = FIXED_NUGGET, optVerbose = FALSE)
        } else if (model_name == "SAL-GP") {
          mdl <- SALGPFit(yList, xList, contiParLogRange = BEST_CONTI_RANGE, attnParLogRange = BEST_ATTN_RANGE, nugget = FIXED_NUGGET, optVerbose = FALSE)
        }
        
        # 2. 模型預測 (這部分不變)
        pred_obj <- NULL
        if (model_name == "aLadder") { pred_obj <- aLadderPred(mdl, x0List_exp3)
        } else if (model_name %in% c("mLadder-Ordinal", "mLadder-Nominal")) { pred_obj <- mLadderPred(mdl, x0List_exp3)
        } else if (model_name == "SAL-GP") { pred_obj <- SALGPPred(mdl, x0List_exp3) }
        
        # 3. 提取目標階段的預測值 (這部分不變)
        final_pred_vector <- NULL
        if (is.list(pred_obj$pred) && !is.data.frame(pred_obj$pred)) {
          final_pred_vector <- pred_obj$pred[[TARGET_STAGE_IDX]]
        } else {
          points_before_target <- 0
          if (TARGET_STAGE_IDX > 1) { points_before_target <- sum(sapply(x0List_exp3[1:(TARGET_STAGE_IDX - 1)], nrow)) }
          num_points_in_target <- nrow(x0List_exp3[[TARGET_STAGE_IDX]])
          start_index <- points_before_target + 1; end_index <- points_before_target + num_points_in_target
          final_pred_vector <- pred_obj$pred[start_index:end_index]
        }
        
        # ***** ↓↓↓ 新增：將這次的預測結果收集起來 ↓↓↓ *****
        if (i == 1) {
          col_name <- paste0("Pred_", gsub("[^A-Za-z0-9]", "_", scenario$name), "_", model_name)
          predictions_collector[[col_name]] <- final_pred_vector
        }
        
        # 4. 計算所有指標 (這部分不變)
        Error <- final_pred_vector - raw_test_y_true
        current_rmse <- sqrt(mean(Error^2)); y_range <- max(raw_test_y_true) - min(raw_test_y_true)
        current_nrmse <- current_rmse / y_range; current_mape <- mean(abs(Error / (raw_test_y_true + 1e-9))) * 100
        current_rsq <- 1 - (sum(Error^2) / sum((raw_test_y_true - mean(raw_test_y_true))^2))
        current_cputime <- mdl$cputime
        
        cat(sprintf("    -> SUCCESS: RMSE=%.4f, NRMSE=%.2f%%, MAPE=%.2f%%, R2=%.4f, CPU=%.2fs\n", 
                    current_rmse, current_nrmse*100, current_mape, current_rsq, current_cputime))
        
      }, error = function(e) {
        cat(sprintf("    -> FAILED: %s\n", e$message))
      })
      
      # (這部分不變)
      results_df <- rbind(results_df, data.frame(
        Simulation_ID = i, Scenario = scenario$name, Model_Name = model_name,
        RMSE = current_rmse, NRMSE = current_nrmse, MAPE = current_mape, R_Squared = current_rsq,
        CPU_Time_sec = current_cputime,
        Nugget_Used = if (!is.null(mdl) && !is.null(mdl$nugget)) mdl$nugget else FIXED_NUGGET
      ))
    }
  }
}

# --- 5. 儲存與分析結果 (這部分不變) ---
write.csv(results_df, output_filename, row.names = FALSE)
cat(sprintf("\nExperiment 3 finished. Results saved to '%s'.\n", output_filename))
summary_df <- results_df %>% group_by(Scenario, Model_Name) %>%
  summarise(across(c(RMSE, NRMSE, MAPE, R_Squared, CPU_Time_sec), list(Mean = ~mean(.x, na.rm = TRUE), SD = ~sd(.x, na.rm = TRUE))), Total_Successes = sum(!is.na(RMSE)))
summary_filename <- file.path(output_dir, "summary_exp3.csv")
write.csv(summary_df, summary_filename, row.names = FALSE)
cat("\n--- Final Summary (Experiment 3) ---\n")
print(as.data.frame(summary_df))


# --- 6. 視覺化結果 (這部分程式碼不變) ---
plots_list <- list(
  RMSE = list(title = "Model Performance Comparison (RMSE)", y_label = "RMSE"),
  MAPE = list(title = "Model Performance Comparison (MAPE)", y_label = "Mean Absolute Percentage Error (%)")
)

for (metric in names(plots_list)) {
  plot_filename <- file.path(output_dir, paste0("boxplot_", tolower(metric), "_exp3.png"))
  
  p <- ggplot(results_df, aes_string(x = "Model_Name", y = metric, fill = "Model_Name")) +
    geom_boxplot() +
    facet_wrap(~ Scenario, scales = "free_y") +
    labs(title = plots_list[[metric]]$title,
         subtitle = paste("Target: Stage 3 Prediction | Based on", num_simulations, "simulations"),
         x = "Model Type", y = plots_list[[metric]]$y_label) +
    theme_bw(base_size = 12) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5),
          axis.text.x = element_text(angle = 25, hjust = 1),
          legend.position = "none")
  
  if (metric == "MAPE") {
    p <- p + scale_y_continuous(labels = scales::percent_format(scale = 1))
  }
  
  print(p)
  ggsave(plot_filename, plot = p, width = 12, height = 7, dpi = 300)
  cat(sprintf("\nBoxplot for %s saved to '%s'.\n", metric, plot_filename))
}


# ***** ↓↓↓ 新增：將收集到的預測結果寫入 Excel 檔案的完整區塊 ↓↓↓ *****

cat("\n--- Writing detailed predictions from Simulation 1 to Excel file ---\n")

# 準備 Excel 檔案
wb <- createWorkbook()
sheet_name <- "Sim_1_Predictions"
addWorksheet(wb, sheet_name)

# 將 Sim 1 的預測結果 list 轉換為一個寬格式的 data frame
if (length(predictions_collector) > 0) {
  # 確保所有向量長度一致，以防萬一
  max_len <- max(sapply(predictions_collector, length))
  df_for_excel <- as.data.frame(lapply(predictions_collector, function(v) {
    length(v) <- max_len
    return(v)
  }))
  
  # 寫入資料到工作表
  writeData(wb, sheet = sheet_name, x = df_for_excel, startCol = 1, startRow = 1)
  
  # 增加一些樣式讓表格更好看
  header_style <- createStyle(textDecoration = "bold", fgFill = "#DCE6F1", border = "TopBottomLeftRight")
  addStyle(wb, sheet = sheet_name, style = header_style, rows = 1, cols = 1:ncol(df_for_excel))
  setColWidths(wb, sheet = sheet_name, cols = 1:ncol(df_for_excel), widths = "auto")
  
  # 儲存 Excel 檔案
  excel_filename <- file.path(output_dir, "Exp3_Detailed_Predictions_Sim1.xlsx")
  saveWorkbook(wb, excel_filename, overwrite = TRUE)
  
  cat(sprintf("Detailed prediction Excel file for Simulation 1 saved to '%s'.\n", excel_filename))
} else {
  cat("No prediction data collected for Simulation 1. Excel file not created.\n")
}
# ***** ↑↑↑ 新增結束 ↑↑↑ *****