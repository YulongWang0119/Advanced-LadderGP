// HEADER
void aIntParam2vec(arma::mat &thetaZ, arma::vec &sigmaF, arma::mat &sigmaInt,
                   const arma::rowvec &param, const arma::uword &xzDim, const arma::uword &zMax);

double aIntCorrKern(const arma::rowvec &xi, const arma::rowvec &xj, const arma::uword &zi, const arma::uword &zj, const arma::uword &xzDim,
                    const arma::mat &thetaZ, const arma::vec &sigmaF, const arma::mat &sigmaInt);

void aIntCorrMat(arma::mat &psi, const arma::mat &x, const arma::uvec &z, const arma::uword &xzDim,
                 const arma::mat &thetaZ, const arma::vec &sigmaF, const arma::mat &sigmaInt);

void aIntCorrVecs(arma::mat &phi, const arma::mat &x0, const arma::uvec &z0,
                  const arma::mat &x, const arma::uvec &z, const arma::uword &xzDim,
                  const arma::mat &thetaZ, const arma::vec &sigmaF, const arma::mat &sigmaInt);

/*void aIntLogLik(double &negloglik, arma::mat &psi, arma::mat &invPsi, double &mu, double &nugget,
                const arma::vec &y, const arma::mat &x, const arma::uvec &z, const arma::uword &xzDim,
                const arma::mat &thetaZ, const arma::vec &sigmaF, const arma::mat &sigmaInt);*/

// 新的聲明 (增加 invSucc_out):

void aIntLogLik(double &negloglik, arma::mat &psi, arma::mat &invPsi, double &mu, double &nugget, bool &invSucc_out,
                const arma::vec &y, const arma::mat &x, const arma::uvec &z, const arma::uword &xzDim,
                const arma::mat &thetaZ, const arma::vec &sigmaF, const arma::mat &sigmaInt);

void aIntNewData(arma::vec &y0, arma::vec &mse, arma::vec &ei, arma::vec &ei_1, arma::vec &ei_2, double &ei_alpha, double &min_y,
                 const arma::mat &x0, const arma::uvec &z0, const arma::vec &y, const arma::mat &x, const arma::uvec &z, const arma::uword &xzDim,
                 double &mu, arma::mat &invPsi, const arma::mat &thetaZ, const arma::vec &sigmaF, const arma::mat &sigmaInt);

// BODY
void aIntParam2vec(arma::mat &thetaZ, arma::vec &sigmaF, arma::mat &sigmaInt,
                   const arma::rowvec &param, const arma::uword &xzDim, const arma::uword &zMax)
{
  /*
   START ASSIGN PARAMETER POSITION
   */
  thetaZ.set_size(zMax, xzDim);
  sigmaF.set_size(zMax);
  sigmaInt.set_size(zMax, zMax);
  /*
   Parameters for Continuous variables
   */
  arma::uword n_thetaZ = zMax * xzDim;
  thetaZ = arma::reshape(param.subvec(0, n_thetaZ - 1), zMax, xzDim);
  /*
   Parameters for Variances
   */
  arma::uword ct = n_thetaZ;
  //
  for (arma::uword u = 0; u < zMax; u++)
  {
    sigmaF(u) = param(ct);
    ct++;
  }
  //
  for (arma::uword i = 0; i < zMax; i++)
  {
    sigmaInt(i, i) = 1.0;
    for (arma::uword j = 0; j < zMax; j++)
    {
      if (i < j)
      {
        sigmaInt(i, j) = param(ct);
        sigmaInt(j, i) = param(ct);
        ct++;
      }
    }
  }
}
/*
// CORRELATION KERNEL OF GAUSSIAN PROCESS
double aIntCorrKern(const arma::rowvec &xi, const arma::rowvec &xj, const arma::uword &zi, const arma::uword &zj, const arma::uword &xzDim,
                    const arma::mat &thetaZ, const arma::vec &sigmaF, const arma::mat &sigmaInt)
{

  arma::rowvec xDiffSq = arma::pow(xi - xj, 2);
  arma::uword zComm = 0;
  if (zi > zj)
  {
    zComm = zj;
  }
  else
  {
    zComm = zi;
  }
  arma::vec corrXvec(zComm, fill::zeros);
  for (arma::uword i = 0; i < zComm; i++)
  {
    arma::rowvec xdtmp = xDiffSq.subvec(i * xzDim, (i + 1) * xzDim - 1);
    corrXvec(i) = std::exp(-(1.0) * arma::accu(thetaZ.row(i) % xdtmp));
  }
  double val = 0.0;
  //
  for (arma::uword i = 0; i < zComm; i++)
  {
    val += sigmaF(i) * corrXvec(i);
  }
  //
  for (arma::uword i = 0; i < zComm; i++)
  {
    for (arma::uword j = 0; j < zComm; j++)
    {
      if (i < j)
      {
        val += sigmaInt(i, j) * corrXvec(i) * corrXvec(j);
      }
    }
  }
  return val;
}
*/

// 第三版
double aIntCorrKern(const arma::rowvec &xi, const arma::rowvec &xj, const arma::uword &zi, const arma::uword &zj, const arma::uword &xzDim,
                    const arma::mat &thetaZ, const arma::vec &sigmaF, const arma::mat &sigmaInt)
{
  // 1. 判斷共同階數
  // 如果 zi(訓練)=4, zj(測試)=3，那麼 zComm = 3
  // 只預測前第三階的部分
  arma::uword zComm = 0;
  if (zi > zj)
  {
    zComm = zj;
  }
  else
  {
    zComm = zi;
  }

  // 如果完全沒交集，回傳 0
  if (zComm == 0)
  {
    return 0.0;
  }

  // 2. 計算距離 (subvec 解決維度報錯問題)
  arma::uword common_dim = zComm * xzDim;
  arma::rowvec xDiffSq = arma::pow(xi.subvec(0, common_dim - 1) - xj.subvec(0, common_dim - 1), 2);

  arma::vec corrXvec(zComm, arma::fill::zeros);

  // 3. 計算每一階的相關性
  // 迴圈只跑 i < zComm (例如只跑到 2，也就是第3階)
  // 即使 thetaZ 裡面有第 4 階的參數，這裡完全不會讀取它
  for (arma::uword i = 0; i < zComm; i++)
  {
    arma::rowvec xdtmp = xDiffSq.subvec(i * xzDim, (i + 1) * xzDim - 1);
    corrXvec(i) = std::exp(-(1.0) * arma::accu(thetaZ.row(i) % xdtmp));
  }

  double val = 0.0;

  // 4. 加總主效應 (Main Effects)
  // 同樣只加到 zComm，第 4 階的 sigmaF 被忽略
  for (arma::uword i = 0; i < zComm; i++)
  {
    val += sigmaF(i) * corrXvec(i);
  }

  // 5. 加總交互作用 (Interaction Effects)
  // 同樣只計算 3 階以內的交互作用
  for (arma::uword i = 0; i < zComm; i++)
  {
    for (arma::uword j = 0; j < zComm; j++)
    {
      if (i < j)
      {
        val += sigmaInt(i, j) * corrXvec(i) * corrXvec(j);
      }
    }
  }
  return val;
}

/*void aIntCorrMat(arma::mat &psi, const arma::mat &x, const arma::uvec &z, const arma::uword &xzDim,
                 const arma::mat &thetaZ, const arma::vec &sigmaF, const arma::mat &sigmaInt)
{
  arma::uword n = x.n_rows;
  for (uword i = 0; i < n; i++)
  {
    for (uword j = 0; j < i; j++)
    {
      arma::rowvec xi = x.row(i);
      arma::rowvec xj = x.row(j);
      arma::uword zi = z(i);
      arma::uword zj = z(j);
      double ker = aIntCorrKern(xi, xj, zi, zj, xzDim, thetaZ, sigmaF, sigmaInt);
      psi(i, j) = ker;
      psi(j, i) = ker;
    }
  }
}
*/

void aIntCorrMat(arma::mat &psi, const arma::mat &x, const arma::uvec &z, const arma::uword &xzDim,
                 const arma::mat &thetaZ, const arma::vec &sigmaF, const arma::mat &sigmaInt)
{
  arma::uword n = x.n_rows;
  for (uword i = 0; i < n; i++)
  {
    for (uword j = 0; j <= i; j++)
    {
      arma::rowvec xi = x.row(i);
      arma::rowvec xj = x.row(j);
      arma::uword zi = z(i);
      arma::uword zj = z(j);
      double ker = aIntCorrKern(xi, xj, zi, zj, xzDim, thetaZ, sigmaF, sigmaInt);
      psi(i, j) = ker;
      psi(j, i) = ker;
    }
  }
}

void aIntCorrVecs(arma::mat &phi, const arma::mat &x0, const arma::uvec &z0,
                  const arma::mat &x, const arma::uvec &z, const arma::uword &xzDim,
                  const arma::mat &thetaZ, const arma::vec &sigmaF, const arma::mat &sigmaInt)
{
  arma::uword n = x.n_rows;
  arma::uword n0 = x0.n_rows;
  for (uword j = 0; j < n0; j++)
  {
    arma::rowvec x0j = x0.row(j);
    arma::uword z0j = z0(j);
    for (uword i = 0; i < n; i++)
    {
      arma::rowvec xi = x.row(i);
      arma::uword zi = z(i);
      double ker = aIntCorrKern(xi, x0j, zi, z0j, xzDim, thetaZ, sigmaF, sigmaInt);
      phi(i, j) = ker;
    }
  }
}

void aIntLogLik(double &negloglik, arma::mat &psi, arma::mat &invPsi, double &mu, double &nugget, bool &invSucc_out,
                const arma::vec &y, const arma::mat &x, const arma::uvec &z, const arma::uword &xzDim,
                const arma::mat &thetaZ, const arma::vec &sigmaF, const arma::mat &sigmaInt)
{
  arma::uword n = y.n_elem;
  arma::vec onevec(n, fill::ones);
  aIntCorrMat(psi, x, z, xzDim, thetaZ, sigmaF, sigmaInt);

  // 只保留這一次正確的呼叫
  calcMatrixInv(invSucc_out, invPsi, psi, nugget);

  double detPsi;
  double signDetPsi;
  arma::log_det(detPsi, signDetPsi, psi);

  if (invSucc_out) // 現在 invSucc_out 在這裡就是一個有效的變數了
  {
    double yPsiY = arma::as_scalar(y.t() * invPsi * y);
    double onePsiY = arma::as_scalar(onevec.t() * invPsi * y);
    double onePsiOne = arma::as_scalar(onevec.t() * invPsi * onevec);
    mu = onePsiY / onePsiOne;
    negloglik = (-1.0) * (-0.5) * (detPsi + yPsiY - (onePsiY * onePsiY) / onePsiOne);
  }
  else
  {
    negloglik = 1e20;
  }
}

void aIntNewData(arma::vec &y0, arma::vec &mse, arma::vec &ei, arma::vec &ei_1, arma::vec &ei_2, double &ei_alpha, double &min_y,
                 const arma::mat &x0, const arma::uvec &z0, const arma::vec &y, const arma::mat &x, const arma::uvec &z, const arma::uword &xzDim,
                 double &mu, arma::mat &invPsi, const arma::mat &thetaZ, const arma::vec &sigmaF, const arma::mat &sigmaInt)
{
  arma::uword n = x.n_rows;
  arma::uword n0 = x0.n_rows;
  arma::mat phi(n, n0, fill::zeros);
  aIntCorrVecs(phi, x0, z0, x, z, xzDim, thetaZ, sigmaF, sigmaInt);
  arma::vec onevec(n, fill::ones);
  arma::vec resid = y - mu * onevec;
  arma::vec psiinvresid = invPsi * resid;
  for (uword j = 0; j < n0; j++)
  {
    y0(j) = mu + arma::as_scalar(phi.col(j).t() * psiinvresid);
    mse(j) = std::abs(1. - arma::as_scalar(phi.col(j).t() * invPsi * phi.col(j))) + datum::eps;
  }
  // Compute expected improvement
  // double min_val = arma::min(y);
  arma::vec rmse = arma::sqrt(mse);
  arma::vec yd = min_y - y0;
  // The improvement part
  ei_1 = yd % (.5 + .5 * arma::erf((1. / std::sqrt(2.)) * (yd / rmse)));
  // The uncertainty part
  ei_2 = (rmse / std::sqrt(2. * datum::pi)) % arma::exp(-.5 * (yd % yd) / mse);
  // The EI value
  ei = 2. * (ei_alpha * ei_1 + (1. - ei_alpha) * ei_2);
  ei.elem(arma::find(ei <= .0)).fill(datum::eps);
}