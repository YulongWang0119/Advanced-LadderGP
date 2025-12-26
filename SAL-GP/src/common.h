// DECLARE FUNCTIONS
void matrixPrintf(const mat &m);
void vecPrintf(const vec &v);
void rvecPrintf(const rowvec &v);
void autoNugget(arma::mat &psi, double &nugget);
void calcMatrixInv(bool &invSucc, arma::mat &invPsi, arma::mat &psi, double &nugget);

//*   `void`: 表示這個函式沒有回傳值。
//    *   `matrixPrintf`: 函式的名稱。
//    *   `(const mat &m)`: 函式的參數。
//        *   `const`: 代表這個函式不會修改傳入的物件 `m`。
//        *   `mat`: Armadillo 的矩陣型別 (`arma::mat`)。
//        *   `&m`: `&` 符號表示「傳參考」(pass by reference)，意思是直接傳遞物件 `m` 本身，而不是複製一份。這樣做效率較高，特別是當矩陣很大時。
//    *   **功能總結**: 這個函式會接收一個矩陣 `m`，然後把它印出來。

//*   `bool &invSucc`: 一個布林值 (true/false)，用來回報求逆是否成功。
//    *   `arma::mat &invPsi`: 用來儲存計算出來的逆矩陣。
//    *   `arma::mat &psi`: 要被求逆的原始矩陣。
//    *   `double &nugget`: 計算過程中使用的 nugget 值。
//    *   **功能總結**: 這個函式會嘗試計算 `psi` 矩陣的逆矩陣，並將結果存入 `invPsi`。它會使用傳入的 `nugget` 來增加數值穩定性，並透過 `invSucc` 告知呼叫者是否成功。

// BODY
void matrixPrintf(const mat &m)
{
  for (uword i = 0; i < m.n_rows; i++)
  {
    for (uword j = 0; j < m.n_cols; j++)
      Rprintf("%4.4f\t", m(i, j));
    Rprintf("\n");
  }
  Rprintf("\n\n");
}

// 定義一個名為 rvecPrintf 的函式，它不回傳任何值。它需要一個名為 v 的參數，這個 v 是一個 Armadillo 的 rowvec (列向量) 的參考，並且我保證在函式內不會對它進行任何修改。
// uword: Armadillo 定義的「無號整數」(unsigned word)。因為索引值（位置）永遠不會是負數，所以使用無號整數很合適。
// i++: 這是 i = i + 1 的簡寫。它表示每執行完一次迴圈內的程式碼後，就把計數器 i 的值加 1。
// "%4.4f\t": 這是「格式化字串」(Format String)，它像一個模板，告訴 Rprintf 要如何顯示後面的資料。
//  %: 特殊字元，表示「這裡要插入一個變數」。
//  f: 代表要插入的變數是個「浮點數」(float/double)。
//  .4: 表示小數點後面要顯示恰好 4 位。
//  4.: 表示輸出的總寬度至少為 4 個字元（如果數字本身不夠寬，可能會在前面補空格）。
//  \t: 這是一個「跳脫字元」，代表一個 Tab 鍵。它會在數字後面插入一個定位字元，使得下一筆輸出的資料能對齊，形成整齊的欄位。
//  , (逗號): 分隔「格式化字串」與要被印出的實際資料。
void rvecPrintf(const rowvec &v)
{
  for (uword i = 0; i < v.n_elem; i++)
    Rprintf("%4.4f\t", v(i));
  Rprintf("\n\n");
}

void vecPrintf(const vec &v)
{
  for (uword i = 0; i < v.n_elem; i++)
    Rprintf("%4.4f\n", v(i));
  Rprintf("\n\n");
}

void autoNugget(arma::mat &psi, double &nugget)
{
  arma::uword n = psi.n_rows;
  arma::mat eyemat(n, n, fill::eye);
  bool ISSYMPD = psi.is_sympd();
  if (!ISSYMPD && (nugget == 0))
  {
    for (arma::uword ng = 0; ng < 101; ng++)
    {
      nugget = std::exp((100. - (double)(ng)) * (-52.) / 100.);
      psi += nugget * eyemat;
      ISSYMPD = psi.is_sympd();
      if (ISSYMPD)
      {
        break;
      }
    }
  }
}

// 修正后的 calcMatrixInv 函數：仅应用传入的 nugget，失败则返回 false
void calcMatrixInv(bool &invSucc, arma::mat &invPsi, arma::mat &psi, double &nugget)
{
  arma::uword n = psi.n_rows;
  arma::mat eyemat(n, n, fill::eye);

  // 1. 確保至少有一個極小的基礎 nugget (使用傳入的 nugget)
  double current_nugget = nugget;

  // 如果 nugget 太小，我們使用一個極小的數值底線確保 Arma::inv_sympd 不會報錯
  if (current_nugget < 1e-12)
  {
    current_nugget = 1e-12;
  }

  // 2. 僅應用我們指定的 nugget
  psi += current_nugget * eyemat;

  // 3. 嘗試求逆
  invSucc = arma::inv_sympd(invPsi, psi);

  // 4. 如果失敗，我們不進行救助 (讓 invSucc = false，讓 R 端的 tryCatch 處理)
  // 如果成功，invSucc = true

  // 重要：如果成功，我們讓 nugget 保持原樣（不更新），因為我們是固定的。
}