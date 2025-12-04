# --- 1. 初始設定與載入套件 ---
library(SFDesign)
library(ggplot2)
library(dplyr)
# 引入所有需要的模型函數 (確認路徑正確)
source('C:/Users/USER/Desktop/PYCProfessor/Ladder GP New2025/ladderGP-main/R/testfunction.R')
source('C:/Users/USER/Desktop/PYCProfessor/Ladder GP New2025/ladderGP-main/R/aladderGP.R')
source('C:/Users/USER/Desktop/PYCProfessor/Ladder GP New2025/ladderGP-main/R/mladderGP.R')

# --- 2. 實驗參數與輸出路徑設定 ---
num_simulations <- 10
experiment_name <- "Experiment_2_Imbalance"

output_base_path <- "C:/Users/USER/Desktop/PYCProfessor/Ladder GP New2025/Result"
timestamp <- format(Sys.time(), "%Y-%m-%d_%H-%M-%S")
output_dir <- file.path(output_base_path, paste0(experiment_name, "_", timestamp))
dir.create(output_dir, showWarnings = FALSE)
output_filename <- file.path(output_dir, "results_exp2.csv")

# --- 3. 實驗設定 (實驗二 - 不平衡性測試) ---
P_DATA_EXP2 <- c(3, 6, 9)
N_TEST_EXP2 <- c(10, 10, 10)

training_scenarios <- list(
  balanced = list(name = "Balanced (30:30:30)", n_train = c(30, 30, 30)),
  front_heavy = list(name = "Front-Heavy (60:15:15)", n_train = c(60, 15, 15)),
  back_heavy = list(name = "Back-Heavy (15:15:60)", n_train = c(15, 15, 60))
)

BEST_CONTI_RANGE <- c(-3, 0.9)
BEST_VAR_RANGE <- c(-3, 0.9)
BEST_CATEG_RANGE <- c(-3, 0.9)

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
  stringsAsFactors = FALSE
)


# 建立一個子資料夾來存放詳細的預測結果
detail_output_dir <- file.path(output_dir, "detailed_predictions")
dir.create(detail_output_dir, showWarnings = FALSE)


# --- 4. 主循環：運行多次模擬 ---
for (i in 1:num_simulations) {
  
  x0List <- lapply(1:length(P_DATA_EXP2), function(k) maxproLHD(N_TEST_EXP2[k], P_DATA_EXP2[k])$design)
  y0List <- lapply(1:length(x0List), function(k) apply(x0List[[k]], 1, Rastrigin))
  
  for (scenario in training_scenarios) {
    
    cat(sprintf("\n--- Sim #%d, Scenario: %s ---\n", i, scenario$name))
    
    N_TRAIN_CURRENT <- scenario$n_train
    xList <- lapply(1:length(P_DATA_EXP2), function(k) maxproLHD(N_TRAIN_CURRENT[k], P_DATA_EXP2[k])$design)
    yList <- lapply(1:length(xList), function(k) apply(xList[[k]], 1, Rastrigin))
    
    # --- 內部循環：測試不同的模型 ---
    for (model_setting in model_list) {
      model_name <- model_setting$name
      cat(sprintf("  Testing model: %s\n", model_name))

      current_rmse <- NA
      current_nrmse <- NA     
      current_rsq <- NA        
      current_cputime <- NA
      
      tryCatch({
        # 2. 把所有可能出錯的訓練和預測程式碼都放進來
        mdl <- NULL
        if (model_name == "aLadder") {
          mdl <- aLadderFit(yList, xList, contiParLogRange = BEST_CONTI_RANGE, varParLogRange = BEST_VAR_RANGE, nugget = 1e-6, optVerbose = FALSE)
        } else if (model_name == "mLadder-Ordinal") {
          mdl <- mLadderFit(yList, xList, zType = "o", contiParLogRange = BEST_CONTI_RANGE, nugget = 1e-6, optVerbose = FALSE)
        } else if (model_name == "mLadder-Nominal") {
          mdl <- mLadderFit(yList, xList, zType = "n", contiParLogRange = BEST_CONTI_RANGE, categParLogRange = BEST_CATEG_RANGE, nugget = 1e-6, optVerbose = FALSE)
        }
        
        pred_obj <- NULL
        if (model_name == "aLadder") {
          pred_obj <- aLadderPred(mdl, x0List, y0listTrue = y0List)
        } else {
          pred_obj <- mLadderPred(mdl, x0List, y0listTrue = y0List)
        }
        
        # 預測成功後，pred_obj 會有 pred (預測值) 和 y_true (真實值)
        y_predicted <- pred_obj$pred
        y_true <- pred_obj$y_true
        
        # 計算誤差
        Error <- y_predicted - y_true
        
        # 只有成功運行到這裡，才更新 rmse 和 cputime 的值
        current_rmse <- sqrt(mean((pred_obj$pred - pred_obj$y_true)^2))
        # 2. 計算 NRMSE (使用 y 真實值的範圍進行標準化)
        y_range <- max(y_true) - min(y_true)
        if (y_range > 0) {
          current_nrmse <- current_rmse / y_range
        } else {
          current_nrmse <- 0 # 如果 y 值都一樣，則 NRMSE 為 0
        }
        
        # 3. 計算 R-Squared (R²)
        SS_res <- sum(Error^2)
        SS_tot <- sum((y_true - mean(y_true))^2)
        if (SS_tot > 0) {
          current_rsq <- 1 - (SS_res / SS_tot)
        } else {
          current_rsq <- 1 
        }
        
        current_cputime <- mdl$cputime
        
        cat(sprintf("    -> SUCCESS: RMSE=%.4f, NRMSE=%.2f%%, R2=%.4f, CPU=%.2f sec\n", 
                    current_rmse, current_nrmse*100, current_rsq, current_cputime)) 
        
        # 1. 建立一個包含詳細預測結果的 data frame
        detail_df <- data.frame(
          Y_True = y_true,
          Y_Predicted = y_predicted,
          Error = Error
        )
        
        # 2. 建立一個包含摘要統計的 data frame
        summary_stats_df <- data.frame(
          Statistic = c("Y_True_Min", "Y_True_Max", "Y_Predicted_Min", "Y_Predicted_Max"),
          Value = c(min(y_true), max(y_true), min(y_predicted), max(y_predicted))
        )
        
        # 3. 定義這次模擬的詳細結果CSV檔名
        # 檔名會包含模擬ID、情境和模型名稱，確保唯一性
        detail_filename <- sprintf("sim_%02d_%s_%s.csv", 
                                   i, 
                                   gsub("[^A-Za-z0-9]", "_", scenario$name), 
                                   model_name)
        
        # 4. 組合完整的檔案路徑
        detail_filepath <- file.path(detail_output_dir, detail_filename)
        
        # 5. 將摘要統計和詳細結果寫入同一個CSV檔案
        # write.csv 比較不方便追加用 write.table 來做
        # 首先寫入摘要
        write.table(summary_stats_df, file = detail_filepath, sep = ",", row.names = FALSE, col.names = TRUE)
        
        # 然後用 append = TRUE 追加一個空行和詳細數據
        write.table(data.frame(V1=""), file = detail_filepath, sep = ",", row.names = FALSE, col.names = FALSE, append = TRUE) # 寫入空行分隔
        write.table(detail_df, file = detail_filepath, sep = ",", row.names = FALSE, col.names = TRUE, append = TRUE)
        
        # cat(sprintf("      -> Detailed prediction saved to: %s\n", detail_filename))
        
      }, error = function(e) {
        cat(sprintf("    -> FAILED: %s\n", e$message))
      })
      

      results_df <- rbind(results_df, data.frame(
        Simulation_ID = i,
        Scenario = scenario$name,
        Model_Name = model_name,
        RMSE = current_rmse,
        NRMSE = current_nrmse,        
        R_Squared = current_rsq,     
        CPU_Time_sec = current_cputime
      ))
      
    }
  }
}

# --- 5. 儲存與分析結果 ---
write.csv(results_df, output_filename, row.names = FALSE)
cat(sprintf("\nExperiment 2 finished. Results saved to '%s'.\n", output_filename))

# 根據 Scenario 和 Model_Name 進行分組摘要
# na.rm = TRUE 可以在計算平均值和標準差時，自動忽略失敗的 NA 值
summary_df <- aggregate(cbind(RMSE, NRMSE, R_Squared, CPU_Time_sec) ~ Scenario + Model_Name, 
                        data = results_df, 
                        FUN = function(x) c(mean = mean(x, na.rm = TRUE), sd = sd(x, na.rm = TRUE)))

summary_filename <- file.path(output_dir, "summary_exp2.csv")
write.csv(summary_df, summary_filename, row.names = FALSE)

# 計算失敗率
failure_counts <- aggregate(RMSE ~ Scenario + Model_Name, data = results_df, 
                            FUN = function(x) sum(is.na(x)))
# 使用 group_by 和 summarise 來計算失敗次數和失敗率
failure_rate_df <- results_df %>%
  group_by(Scenario, Model_Name) %>%
  summarise(
    Failure_Count = sum(is.na(RMSE)),
    .groups = 'drop' # 計算完後取消分組
  ) %>%
  mutate(
    Failure_Rate_Percent = (Failure_Count / num_simulations) * 100
  )

failure_filename <- file.path(output_dir, "failure_rate_exp2.csv")
write.csv(failure_rate_df, failure_filename, row.names = FALSE)

cat("\n--- Final Summary ---\n")
print(summary_df)
cat("\n--- Failure Rate Summary (Corrected) ---\n") # 加上 (Corrected) 標記
print(failure_rate_df)

# --- 6. 視覺化結果 ---

# --- 步驟 6.1: 建立一個通用的繪圖函數 ---
create_boxplot <- function(data, y_metric, y_label, file_suffix, is_percent = FALSE) {
  
  # 移除NA值，避免繪圖警告
  plot_data <- data[!is.na(data[[y_metric]]), ]
  
  # 如果是百分比，將y值乘以100
  if (is_percent) {
    plot_data[[y_metric]] <- plot_data[[y_metric]] * 100
  }
  
  # 組合圖檔名
  plot_filename <- file.path(output_dir, paste0("boxplot_", file_suffix, "_exp2_ALL.png"))
  
  # 繪圖
  p <- ggplot(plot_data, aes(x = Model_Name, y = .data[[y_metric]], fill = Model_Name)) +
    geom_boxplot() +
    facet_wrap(~ Scenario) + 
    labs(
      title = paste("Comparison of", y_label, "under Different Training Scenarios"),
      subtitle = paste("Experiment 2: 3-Stage Data, Based on", num_simulations, "simulations"),
      x = "Model Type",
      y = y_label
    ) +
    theme_bw(base_size = 12) + 
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5),
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "none"
    )
  
  # R-Squared 的 Y 軸通常從 0 到 1 (或 100%)
  if (y_metric == "R_Squared") {
    p <- p + coord_cartesian(ylim = c(min(0, min(plot_data$R_Squared, na.rm=TRUE)), 1))
  }
  
  print(p)
  ggsave(plot_filename, plot = p, width = 12, height = 7, dpi = 300)
  
  cat(sprintf("\nCombined boxplot for %s saved to '%s'.\n", y_label, plot_filename))
}


# --- 步驟 6.2: 呼叫函數來產生三張不同的圖表 ---
# 1. 繪製 RMSE 圖 
create_boxplot(data = results_df, 
               y_metric = "RMSE", 
               y_label = "Root Mean Squared Error (RMSE)", 
               file_suffix = "rmse")

# 2. 繪製 NRMSE 圖 
create_boxplot(data = results_df, 
               y_metric = "NRMSE", 
               y_label = "Normalized RMSE (%)", 
               file_suffix = "nrmse",
               is_percent = TRUE) # 標記為百分比，Y軸會自動乘以100

# 3. 繪製 R-Squared 圖 
create_boxplot(data = results_df, 
               y_metric = "R_Squared", 
               y_label = "R-Squared (R²)", 
               file_suffix = "r_squared")


# --- 步驟 6.3:建立並呼叫一個通用的「獨立圖檔」繪圖函數 ---

create_individual_plots <- function(data, y_metric, y_label, file_prefix, is_percent = FALSE) {
  
  cat(sprintf("\nGenerating individual plots for %s...\n", y_label))
  unique_scenarios <- unique(data$Scenario)
  
  for (current_scenario in unique_scenarios) {
    
    # 篩選出只屬於當前 scenario 的資料
    subset_df <- data[data$Scenario == current_scenario, ]
    
    # 移除 NA 值
    subset_df <- subset_df[!is.na(subset_df[[y_metric]]), ]
    
    # 如果是百分比，將y值乘以100
    if (is_percent) {
      subset_df[[y_metric]] <- subset_df[[y_metric]] * 100
    }
    
    # 設定獨立的檔名
    plot_filename <- file.path(output_dir, paste0(file_prefix, "_", gsub("[^A-Za-z0-9]", "_", current_scenario), ".png"))
    
    # 繪製單張圖
    p <- ggplot(subset_df, aes(x = Model_Name, y = .data[[y_metric]], fill = Model_Name)) +
      geom_boxplot() +
      labs(
        title = paste(y_label, "for Scenario:", current_scenario),
        subtitle = paste("Based on", num_simulations, "simulations"),
        x = "Model Type",
        y = y_label
      ) +
      theme_bw(base_size = 14) + 
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none"
      )
    
    # R-Squared 的 Y 軸從 0 到 1 (或 100%)
    if (y_metric == "R_Squared") {
      p <- p + coord_cartesian(ylim = c(min(0, min(subset_df$R_Squared, na.rm=TRUE)), 1))
    }
    
    # 儲存單張圖
    ggsave(plot_filename, plot = p, width = 8, height = 6, dpi = 300)
    
    cat(sprintf("-> Individual plot for '%s' saved.\n", current_scenario))
  }
}

# 呼叫函數三次，為三個指標分別產生獨立圖檔
create_individual_plots(data = results_df, 
                        y_metric = "RMSE", 
                        y_label = "RMSE", 
                        file_prefix = "boxplot_rmse")

create_individual_plots(data = results_df, 
                        y_metric = "NRMSE", 
                        y_label = "NRMSE (%)", 
                        file_prefix = "boxplot_nrmse",
                        is_percent = TRUE)

create_individual_plots(data = results_df, 
                        y_metric = "R_Squared", 
                        y_label = "R-Squared (R²)", 
                        file_prefix = "boxplot_r_squared")

