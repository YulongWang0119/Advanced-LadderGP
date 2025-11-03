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

void aIntLogLik(double &negloglik, arma::mat &psi, arma::mat &invPsi, double &mu, double &nugget,
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

// CORRELATION KERNEL OF GAUSSIAN PROCESS
double aIntCorrKern(const arma::rowvec &xi, const arma::rowvec &xj, const arma::uword &zi, const arma::uword &zj, const arma::uword &xzDim,
                    const arma::mat &thetaZ, const arma::vec &sigmaF, const arma::mat &sigmaInt)
{
  /* corr X*/
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

void aIntCorrMat(arma::mat &psi, const arma::mat &x, const arma::uvec &z, const arma::uword &xzDim,
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

/*
void aIntLogLik(double &negloglik, arma::mat &psi, arma::mat &invPsi, double &mu, double &nugget,
                const arma::vec &y, const arma::mat &x, const arma::uvec &z, const arma::uword &xzDim,
                const arma::mat &thetaZ, const arma::vec &sigmaF, const arma::mat &sigmaInt)
{
  arma::uword n = y.n_elem;
  arma::vec onevec(n, fill::ones);
  aIntCorrMat(psi, x, z, xzDim, thetaZ, sigmaF, sigmaInt);

  arma::vec eigval;
  arma::mat eigvec;
  arma::eig_sym(eigval, eigvec, psi);
  arma::mat eyemat(n, n, fill::eye);
  double checkCond = std::abs(eigval.max()) - 1e8*std::abs(eigval.min());

  if ((nugget == 0) & (checkCond >= 0)) {
    nugget = checkCond/(1e8 - 1);
  }
  psi += nugget*eyemat;

  double detPsi;
  double signDetPsi;
  bool invSucc;
  invSucc = arma::inv_sympd(invPsi, psi, inv_opts::allow_approx);

  arma::log_det(detPsi, signDetPsi, psi);
  //if (std::isfinite(detPsi) & (signDetPsi >= 0))
  if (invSucc) {
    double yPsiY = arma::as_scalar(y.t()*invPsi*y);
    double onePsiY = arma::as_scalar(onevec.t()*invPsi*y);
    double onePsiOne = arma::as_scalar(onevec.t()*invPsi*onevec);
    mu = onePsiY/onePsiOne;

    negloglik = (-1.0)*(-0.5)*(detPsi + yPsiY - (onePsiY*onePsiY)/onePsiOne);
  } else {
    negloglik = 1e20;
  }
}
*/
/*#include <iomanip>*/

// =========================================================================
//           aIntLogLik (Cholesky  +  invPsi)
// =========================================================================
void aIntLogLik(double &negloglik, arma::mat &psi, arma::mat &invPsi, double &mu, double &nget,
                const arma::vec &y, const arma::mat &x, const arma::uvec &z, const arma::uword &xzDim,
                const arma::mat &thetaZ, const arma::vec &sigmaF, const arma::mat &sigmaInt)
{
  // --- 步驟 1: 構建 Psi 矩陣 ---
  arma::uword n = y.n_elem;
  arma::vec onevec(n, arma::fill::ones);
  aIntCorrMat(psi, x, z, xzDim, thetaZ, sigmaF, sigmaInt);

  // --- 步驟 2: 穩定化與 Cholesky 分解檢查 跟教授原始版不同 ---
  psi += nget * arma::eye(n, n);
  arma::mat U;
  bool chol_success = arma::chol(U, psi);

  if (!chol_success)
  {
    negloglik = 1e10;
    // 將 invPsi 設為空或單位矩陣
    invPsi.eye(n, n);
    return;
  }

  // --- 步驟 3: 計算數學式 ---

  // Cholesky 成功了可以安全地計算 invPsi
  // arma::inv(U) 計算上三角矩陣 U 的逆
  // inv(U) * inv(U.t()) = inv(U.t() * U) = inv(psi)
  // 從 Cholesky 分解結果計算逆矩陣的
  invPsi = arma::inv(arma::trimatu(U)) * arma::inv(arma::trimatu(U)).t();

  arma::vec psi_inv_y = invPsi * y;
  arma::vec psi_inv_1 = invPsi * onevec;

  mu = arma::as_scalar(onevec.t() * psi_inv_y) / arma::as_scalar(onevec.t() * psi_inv_1);

  double detPsi_log = 2.0 * arma::accu(arma::log(U.diag()));
  double yPsiY = arma::as_scalar(y.t() * psi_inv_y);
  double onePsiY = arma::as_scalar(onevec.t() * psi_inv_y);
  double onePsiOne = arma::as_scalar(onevec.t() * psi_inv_1);

  negloglik = 0.5 * (detPsi_log + yPsiY - (onePsiY * onePsiY) / onePsiOne);

  if (!arma::is_finite(negloglik))
  {
    negloglik = 1e10;
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
