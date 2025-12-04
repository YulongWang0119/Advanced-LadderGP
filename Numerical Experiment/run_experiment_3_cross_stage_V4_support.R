# --- 1. 初始設定 ---
library(ggplot2)
library(reshape2) # 用於 melt() 函數
library(viridis)  # 提供更好的顏色方案

# --- 2. 載入失敗的矩陣 ---
# *** 把這裡的路徑換成你實際保存的 rds 檔案路徑 ***
# 例如，分析 4 階段失敗的矩陣
file_path <- "debug_matrices/failed_psi_aLadder_20251204_122952.rds" 
failed_psi <- readRDS(file_path)

# 取得檔案的基本資訊
file_info <- gsub(".rds", "", basename(file_path))


# --- 3. 核心診斷分析 ---

# (1) 檢查矩陣維度
cat("Matrix Dimensions:", dim(failed_psi)[1], "x", dim(failed_psi)[2], "\n")

# (2) 查看數值範圍
summary_stats <- summary(as.vector(failed_psi))
cat("\nSummary of matrix values:\n")
print(summary_stats)

# (3) 計算條件數 (Condition Number)
# 條件數衡量矩陣的穩定性，數字越大越不穩定 (接近無限大就是奇異)
# 因為矩陣可能不是正定的用 tryCatch 包起來
condition_number <- tryCatch({
  kappa(failed_psi, exact = TRUE)
}, error = function(e) {
  return("Calculation failed: Matrix is likely not positive definite.")
})
cat("\nCondition Number:", format(condition_number, scientific = TRUE, digits = 4), "\n")

# (4) 計算特徵值 (Eigenvalues)
# 對於一個有效的協方差矩陣，所有特徵值都必須是正數。
eigen_values <- eigen(failed_psi, only.values = TRUE)$values
cat("\nSummary of Eigenvalues:\n")
print(summary(eigen_values))

# --- 4. 視覺化診斷 ---
# (1) 熱圖 (Heatmap) - 觀察數值分佈
# melt 函數將矩陣轉換成長格式的 data frame，方便 ggplot 使用
melted_psi <- melt(failed_psi) 

heatmap_plot <- ggplot(data = melted_psi, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_viridis(option = "C") + # 使用 viridis 顏色方案
  labs(
    title = paste("Heatmap of Failed Psi Matrix"),
    subtitle = paste("Case:", file_info),
    x = "Data Point Index",
    y = "Data Point Index",
    fill = "Value"
  ) +
  theme_minimal() +
  coord_fixed() # 確保格子是正方形

# 顯示圖表
print(heatmap_plot)

# 儲存圖表
ggsave(paste0(file_info, "_heatmap.png"), plot = heatmap_plot, width = 8, height = 7, path = "debug_matrices")


# (2) 特徵值分佈圖
eigen_df <- data.frame(index = 1:length(eigen_values), value = sort(eigen_values))

eigen_plot <- ggplot(eigen_df, aes(x = index, y = value)) +
  geom_line() +
  geom_point(aes(color = value < 0)) + # 將負特徵值標為紅色
  scale_color_manual(values = c("black", "red")) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "blue") +
  labs(
    title = "Distribution of Eigenvalues",
    subtitle = paste("Case:", file_info),
    x = "Eigenvalue Index (Sorted)",
    y = "Eigenvalue"
  ) +
  theme_bw() +
  theme(legend.position = "none")

# 顯示圖表
print(eigen_plot)

# 儲存圖表
ggsave(paste0(file_info, "_eigenvalues.png"), plot = eigen_plot, width = 8, height = 6, path = "debug_matrices")



# --- 5. 視覺化診斷  ---

# 帶有數值的熱圖 (Heatmap with Numerical Labels)

# 再次使用 melted_psi data frame
# 新增一個 'label' 欄位，將數值格式化到小數點後兩位
melted_psi$label <- sprintf("%.2f", melted_psi$value)


matrix_size <- nrow(failed_psi)
font_size <- if (matrix_size <= 30) 3 else if (matrix_size <= 40) 2.5 else 2

numeric_heatmap_plot <- ggplot(data = melted_psi, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") + # 加上白色邊框讓格子更清楚
  
  # 核心部分：加上文字標籤
  geom_text(aes(label = label), color = "black", size = font_size) +
  
  scale_fill_viridis(option = "C", name = "Value") + 
  labs(
    title = "Numerical Heatmap of Failed Psi Matrix",
    subtitle = paste("Case:", file_info, "| Each cell shows the correlation value (rounded)"),
    x = "Data Point Index",
    y = "Data Point Index"
  ) +
  theme_minimal() +
  coord_fixed() + # 確保格子是正方形
  # 調整 y 軸方向，讓矩陣的 (1,1) 在左上角，符合直覺
  scale_y_reverse() 

# 顯示圖表
print(numeric_heatmap_plot)

# 儲存圖表
ggsave(paste0(file_info, "_numeric_heatmap.png"), 
       plot = numeric_heatmap_plot, 
       width = 12,  # 增加圖片寬度以容納文字
       height = 11, # 增加圖片高度
       path = "debug_matrices",
       dpi = 150)   # 設定解析度

# --- 5. 視覺化診斷  ---

# (A) 將矩陣轉換為長格式資料框 (Melt the matrix)
melted_psi <- melt(failed_psi)

# (B) 準備繪圖參數
matrix_size <- nrow(failed_psi)
font_size <- if (matrix_size <= 30) 3 else if (matrix_size <= 40) 2.5 else 2

# (C) 
# 決定文字顏色的閾值 (通常是顏色範圍的中間點)
color_midpoint <- 1.0 
melted_psi$text_color <- ifelse(melted_psi$value < color_midpoint, "white", "black")

# (D) 加上數值標籤
melted_psi$label <- sprintf("%.2f", melted_psi$value)


presentation_heatmap <- ggplot(data = melted_psi, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "grey50", size = 0.1) + # 用灰色細線分隔格子
  
  # 使用智慧文字顏色
  geom_text(aes(label = label, color = text_color), size = font_size) +
  scale_color_identity() + # 讓 ggplot 直接使用我們指定的顏色
  
  # 以 1.0 為中心點，小於1為藍色，大於1為紅色
  scale_fill_gradient2(
    low = "#313695",      # 深藍 (小於 1)
    mid = "#FFFFBF",      # 淺黃 (接近 1)
    high = "#A50026",     # 深紅 (大於 1)
    midpoint = 1.0,
    name = "Value"
  ) +
  
  # 加上區塊分隔線】
  geom_vline(xintercept = c(10.5, 20.5), color = "white", size = 1.2) +
  geom_hline(yintercept = c(10.5, 20.5), color = "white", size = 1.2) +
  
  labs(
    title = "Systematic Failure of the aLadder Covariance Matrix",
    subtitle = paste("Case:", file_info, "| Values > 1.0 indicate structural model failure"),
    x = "Data Point Index (by Stage)",
    y = "Data Point Index (by Stage)"
  ) +
  theme_minimal(base_size = 14) + # 加大基礎字體
  coord_fixed() +
  scale_y_reverse(breaks = c(1, 10, 20, 30)) + # 調整座標軸刻度
  scale_x_continuous(breaks = c(1, 10, 20, 30))

# 顯示圖表
print(presentation_heatmap)

# 儲存更高品質的圖檔
ggsave(file.path("debug_matrices", paste0(file_info, "_PRESENTATION_heatmap.png")), 
       plot = presentation_heatmap, 
       width = 11, height = 10, dpi = 300) # 提高解析度以供發表使用

# --- 6. 產生輔助文字檔 ---

# 設定檔案路徑
log_filename <- file.path("debug_matrices", paste0(file_info, "_diagnostics.txt"))

# 使用 sink() 將所有後續的輸出都導向到檔案
sink(log_filename)

# --- 開始寫入內容 ---
cat("=========================================================\n")
cat(" Diagnostic Report for Failed Psi Matrix\n")
cat("=========================================================\n\n")

cat("Case:", file_info, "\n\n")

cat("--- Key Metrics ---\n")
cat("Matrix Dimensions:", dim(failed_psi)[1], "x", dim(failed_psi)[2], "\n")
cat("Condition Number:", format(condition_number, scientific = TRUE, digits = 4), "\n")
cat("\nSummary of Matrix Values:\n")
print(summary(as.vector(failed_psi)))
cat("\nSummary of Eigenvalues:\n")
print(summary(eigen_values))
cat("\nNumber of Negative Eigenvalues:", sum(eigen_values < 0), "\n\n")

cat("--- Full Psi Matrix (rounded to 2 decimal places) ---\n\n")

# 使用 print 函數並設定 options 來獲得更好的排版
options(width = 200) # 加寬輸出寬度
print(round(failed_psi, 2))

# --- 寫入結束 ---
# 關閉 sink()，將輸出還原到 Console
sink()

cat("\n已成功產生診斷文字檔:", log_filename, "\n")