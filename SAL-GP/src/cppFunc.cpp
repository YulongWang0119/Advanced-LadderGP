
#include "header.h"

// RCPP FUNCTIONS

//[[Rcpp::export]]
double gpObjCpp(arma::vec param, arma::vec y, arma::mat x, double nugget)
{
  arma::uword n = x.n_rows;
  double negloglik, mu, sigma2;
  arma::mat psi(n, n, fill::eye);
  arma::mat invPsi(n, n, fill::eye);
  // double nugget = 0.;
  gpLogLik(negloglik, psi, invPsi, mu, sigma2, nugget, y, x, param);
  return negloglik;
}

//[[Rcpp::export]]
Rcpp::List gpModel(arma::vec param, arma::vec y, arma::mat x, double nugget)
{
  arma::uword n = x.n_rows;
  double negloglik, mu, sigma2;
  arma::mat psi(n, n, fill::eye);
  arma::mat invPsi(n, n, fill::eye);
  // double nugget = 0.;
  gpLogLik(negloglik, psi, invPsi, mu, sigma2, nugget, y, x, param);
  return List::create(Named("alpha") = wrap(param),
                      Named("psi") = wrap(psi),
                      Named("invPsi") = wrap(invPsi),
                      Named("mu") = wrap(mu),
                      Named("sigma2") = wrap(sigma2),
                      Named("negloglik") = wrap(negloglik),
                      Named("nugget") = wrap(nugget));
}

//[[Rcpp::export]]
Rcpp::List gpPred(arma::mat x0, arma::vec y, arma::mat x,
                  arma::vec param, arma::mat invPsi, double mu, double sigma2, double ei_alpha, double min_y)
{
  arma::uword n0 = x0.n_rows;
  arma::vec y0(n0, fill::zeros);
  arma::vec mse(n0, fill::zeros);
  arma::vec ei(n0, fill::zeros);
  arma::vec ei_1(n0, fill::zeros);
  arma::vec ei_2(n0, fill::zeros);
  gpNewData(y0, mse, ei, ei_1, ei_2, ei_alpha, min_y, x0, y, x, mu, sigma2, invPsi, param);
  //
  return List::create(Named("pred") = wrap(y0),
                      Named("mse") = wrap(mse),
                      Named("ei") = wrap(ei),
                      Named("improvement") = wrap(ei_1),
                      Named("uncertainty") = wrap(ei_2));
}

// Stage as a ordinal variable
//[[Rcpp::export]]
double lgpOBObjCpp(arma::rowvec param, arma::vec y, arma::mat x, arma::uvec z, arma::uword xzDim, double nugget, bool logParam)
{
  arma::uword n = x.n_rows;
  arma::uword zMax = z.max();
  double negloglik, mu, sigma2;
  arma::mat psi(n, n, fill::eye);
  arma::mat invPsi(n, n, fill::eye);
  //
  double tau;
  arma::mat theta;
  // double nugget = 0.;
  if (logParam)
  {
    lgpOBParam2vec(tau, theta, arma::exp(param), xzDim, zMax);
  }
  else
  {
    lgpOBParam2vec(tau, theta, param, xzDim, zMax);
  }
  lgpOBLogLik(negloglik, psi, invPsi, mu, sigma2, nugget, y, x, z, xzDim,
              tau, theta);
  return negloglik;
}

//[[Rcpp::export]]
Rcpp::List lgpOBModel(arma::rowvec param, arma::vec y, arma::mat x, arma::uvec z, arma::uword xzDim, double nugget, bool logParam)
{
  arma::uword n = x.n_rows;
  arma::uword zMax = z.max();
  double negloglik, mu, sigma2;
  arma::mat psi(n, n, fill::eye);
  arma::mat invPsi(n, n, fill::eye);
  //
  double tau;
  arma::mat theta;
  // double nugget = 0.;
  if (logParam)
  {
    lgpOBParam2vec(tau, theta, arma::exp(param), xzDim, zMax);
  }
  else
  {
    lgpOBParam2vec(tau, theta, param, xzDim, zMax);
  }
  lgpOBLogLik(negloglik, psi, invPsi, mu, sigma2, nugget, y, x, z, xzDim,
              tau, theta);
  //
  return List::create(Named("mu") = wrap(mu),
                      Named("sigma2") = wrap(sigma2),
                      Named("tau") = wrap(tau),
                      Named("theta") = wrap(theta),
                      Named("psi") = wrap(psi),
                      Named("invPsi") = wrap(invPsi),
                      Named("negloglik") = wrap(negloglik),
                      Named("nugget") = wrap(nugget),
                      Named("vecParams") = wrap(param));
}

//[[Rcpp::export]]
Rcpp::List lgpOBPred(arma::mat x0, arma::uvec z0, arma::vec y, arma::mat x, arma::uvec z, arma::uword xzDim,
                     arma::rowvec param, arma::mat invPsi, double mu, double sigma2, double ei_alpha, double min_y, bool logParam)
{
  arma::uword n0 = x0.n_rows;
  arma::uword zMax = z.max();
  //
  double tau;
  arma::mat theta;
  if (logParam)
  {
    lgpOBParam2vec(tau, theta, arma::exp(param), xzDim, zMax);
  }
  else
  {
    lgpOBParam2vec(tau, theta, param, xzDim, zMax);
  }
  //
  arma::vec y0(n0, fill::zeros);
  arma::vec mse(n0, fill::zeros);
  arma::vec ei(n0, fill::zeros);
  arma::vec ei_1(n0, fill::zeros);
  arma::vec ei_2(n0, fill::zeros);
  lgpOBNewData(y0, mse, ei, ei_1, ei_2, ei_alpha, min_y, x0, z0, y, x, z, xzDim,
               mu, sigma2, invPsi, tau, theta);
  //
  return List::create(Named("pred") = wrap(y0),
                      Named("mse") = wrap(mse),
                      Named("ei") = wrap(ei),
                      Named("improvement") = wrap(ei_1),
                      Named("uncertainty") = wrap(ei_2));
}

// Stage as a nominal variable

//[[Rcpp::export]]
double lgpNBObjCpp(arma::rowvec param, arma::vec y, arma::mat x, arma::uvec z, arma::uword xzDim, double nugget, bool logParam)
{
  arma::uword n = x.n_rows;
  arma::uword zMax = z.max();
  double negloglik, mu, sigma2;
  arma::mat psi(n, n, fill::eye);
  arma::mat invPsi(n, n, fill::eye);
  //
  arma::mat tau, theta;
  // double nugget = 0.;
  if (logParam)
  {
    lgpNBParam2vec(tau, theta, arma::exp(param), xzDim, zMax);
  }
  else
  {
    lgpNBParam2vec(tau, theta, param, xzDim, zMax);
  }
  lgpNBLogLik(negloglik, psi, invPsi, mu, sigma2, nugget, y, x, z, xzDim,
              tau, theta);
  return negloglik;
}

//[[Rcpp::export]]
Rcpp::List lgpNBModel(arma::rowvec param, arma::vec y, arma::mat x, arma::uvec z, arma::uword xzDim, double nugget, bool logParam)
{
  arma::uword n = x.n_rows;
  arma::uword zMax = z.max();
  double negloglik, mu, sigma2;
  arma::mat psi(n, n, fill::eye);
  arma::mat invPsi(n, n, fill::eye);
  //
  arma::mat tau, theta;
  // double nugget = 0.;
  if (logParam)
  {
    lgpNBParam2vec(tau, theta, arma::exp(param), xzDim, zMax);
  }
  else
  {
    lgpNBParam2vec(tau, theta, param, xzDim, zMax);
  }
  lgpNBLogLik(negloglik, psi, invPsi, mu, sigma2, nugget, y, x, z, xzDim,
              tau, theta);
  //
  return List::create(Named("mu") = wrap(mu),
                      Named("sigma2") = wrap(sigma2),
                      Named("tau") = wrap(tau),
                      Named("theta") = wrap(theta),
                      Named("psi") = wrap(psi),
                      Named("invPsi") = wrap(invPsi),
                      Named("negloglik") = wrap(negloglik),
                      Named("nugget") = wrap(nugget),
                      Named("vecParams") = wrap(param));
}

//[[Rcpp::export]]
Rcpp::List lgpNBPred(arma::mat x0, arma::uvec z0, arma::vec y, arma::mat x, arma::uvec z, arma::uword xzDim,
                     arma::rowvec param, arma::mat invPsi, double mu, double sigma2, double ei_alpha, double min_y, bool logParam)
{
  arma::uword n0 = x0.n_rows;
  arma::uword zMax = z.max();
  //
  arma::mat tau, theta;
  if (logParam)
  {
    lgpNBParam2vec(tau, theta, arma::exp(param), xzDim, zMax);
  }
  else
  {
    lgpNBParam2vec(tau, theta, param, xzDim, zMax);
  }
  //
  arma::vec y0(n0, fill::zeros);
  arma::vec mse(n0, fill::zeros);
  arma::vec ei(n0, fill::zeros);
  arma::vec ei_1(n0, fill::zeros);
  arma::vec ei_2(n0, fill::zeros);
  lgpNBNewData(y0, mse, ei, ei_1, ei_2, ei_alpha, min_y, x0, z0, y, x, z, xzDim,
               mu, sigma2, invPsi, tau, theta);
  //
  return List::create(Named("pred") = wrap(y0),
                      Named("mse") = wrap(mse),
                      Named("ei") = wrap(ei),
                      Named("improvement") = wrap(ei_1),
                      Named("uncertainty") = wrap(ei_2));
}

/*
// Stage as interaction effect
//[[Rcpp::export]]
double aIntObjCpp(arma::rowvec param, arma::vec y, arma::mat x, arma::uvec z, arma::uword xzDim, double nugget, bool logParam)
{
  arma::uword n = x.n_rows;
  arma::uword zMax = z.max();
  double negloglik, mu;
  arma::mat psi(n, n, fill::eye);
  arma::mat invPsi(n, n, fill::eye);
  //
  arma::mat thetaZ;
  arma::vec sigmaF;
  arma::mat sigmaInt;
  //double nugget = 0.;
  if (logParam) {
    aIntParam2vec(thetaZ, sigmaF, sigmaInt, arma::exp(param), xzDim, zMax);
  } else{
    aIntParam2vec(thetaZ, sigmaF, sigmaInt, param, xzDim, zMax);
  }
  aIntLogLik(negloglik, psi, invPsi, mu, nugget, y, x, z, xzDim,
             thetaZ, sigmaF, sigmaInt);
  return negloglik;
}
*/

//[[Rcpp::export]]
double aIntObjCpp(arma::rowvec param, arma::vec y, arma::mat x, arma::uvec z, arma::uword xzDim, double nugget, bool logParam)
{
  arma::uword n = x.n_rows;
  arma::uword zMax = z.max();
  double negloglik, mu;
  arma::mat psi(n, n, fill::eye);
  arma::mat invPsi(n, n, fill::eye);

  // 同樣需要一個狀態變數，即使這裡不會用它
  bool inv_success_status = false;

  arma::mat thetaZ;
  arma::vec sigmaF;
  arma::mat sigmaInt;

  if (logParam)
  {
    aIntParam2vec(thetaZ, sigmaF, sigmaInt, arma::exp(param), xzDim, zMax);
  }
  else
  {
    aIntParam2vec(thetaZ, sigmaF, sigmaInt, param, xzDim, zMax);
  }

  // 呼叫修改後的 aIntLogLik
  aIntLogLik(negloglik, psi, invPsi, mu, nugget, inv_success_status,
             y, x, z, xzDim, thetaZ, sigmaF, sigmaInt);

  return negloglik;
}

/*
//[[Rcpp::export]]
Rcpp::List aIntModel(arma::rowvec param, arma::vec y, arma::mat x, arma::uvec z, arma::uword xzDim, double nugget, bool logParam)
{
  arma::uword n = x.n_rows;
  arma::uword zMax = z.max();
  double negloglik, mu;
  arma::mat psi(n, n, fill::eye);
  arma::mat invPsi(n, n, fill::eye);
  //
  arma::mat thetaZ;
  arma::vec sigmaF;
  arma::mat sigmaInt;
  //double nugget = 0.;
  if (logParam) {
    aIntParam2vec(thetaZ, sigmaF, sigmaInt, arma::exp(param), xzDim, zMax);
  } else{
    aIntParam2vec(thetaZ, sigmaF, sigmaInt, param, xzDim, zMax);
  }
  aIntLogLik(negloglik, psi, invPsi, mu, nugget, y, x, z, xzDim,
             thetaZ, sigmaF, sigmaInt);
  //
  return List::create(Named("mu") = wrap(mu),
                      Named("thetaZ") = wrap(thetaZ),
                      Named("sigmaF") = wrap(sigmaF),
                      Named("sigmaInt") = wrap(sigmaInt),
                      Named("psi") = wrap(psi),
                      Named("invPsi") = wrap(invPsi),
                      Named("negloglik") = wrap(negloglik),
                      Named("nugget") = wrap(nugget),
                      Named("vecParams") = wrap(param)
  );
}
*/

//[[Rcpp::export]]
Rcpp::List aIntModel(arma::rowvec param, arma::vec y, arma::mat x, arma::uvec z, arma::uword xzDim, double nugget, bool logParam)
{
  arma::uword n = x.n_rows;
  arma::uword zMax = z.max();
  double negloglik, mu;
  arma::mat psi(n, n, fill::eye);
  arma::mat invPsi(n, n, fill::eye);

  // 新增一個變數來儲存求逆的成功狀態
  bool inv_success_status = false;

  arma::mat thetaZ;
  arma::vec sigmaF;
  arma::mat sigmaInt;

  if (logParam)
  {
    aIntParam2vec(thetaZ, sigmaF, sigmaInt, arma::exp(param), xzDim, zMax);
  }
  else
  {
    aIntParam2vec(thetaZ, sigmaF, sigmaInt, param, xzDim, zMax);
  }

  // 呼叫修改後的 aIntLogLik
  aIntLogLik(negloglik, psi, invPsi, mu, nugget, inv_success_status, // <--- 傳入狀態變數
             y, x, z, xzDim, thetaZ, sigmaF, sigmaInt);

  // 無論成功或失敗，都回傳完整的 List
  return List::create(Named("mu") = wrap(mu),
                      Named("thetaZ") = wrap(thetaZ),
                      Named("sigmaF") = wrap(sigmaF),
                      Named("sigmaInt") = wrap(sigmaInt),
                      Named("psi") = wrap(psi),
                      Named("invPsi") = wrap(invPsi),
                      Named("negloglik") = wrap(negloglik),
                      Named("nugget") = wrap(nugget),
                      Named("vecParams") = wrap(param),
                      Named("inv_success") = wrap(inv_success_status) // <--- 新增的回傳值
  );
}

//[[Rcpp::export]]
Rcpp::List aIntPred(arma::mat x0, arma::uvec z0, arma::vec y, arma::mat x, arma::uvec z, arma::uword xzDim,
                    arma::rowvec param, arma::mat invPsi, double mu, double ei_alpha, double min_y, bool logParam)
{
  arma::uword n0 = x0.n_rows;
  arma::uword zMax = z.max();
  //
  arma::mat thetaZ;
  arma::vec sigmaF;
  arma::mat sigmaInt;
  if (logParam)
  {
    aIntParam2vec(thetaZ, sigmaF, sigmaInt, arma::exp(param), xzDim, zMax);
  }
  else
  {
    aIntParam2vec(thetaZ, sigmaF, sigmaInt, param, xzDim, zMax);
  }
  //
  arma::vec y0(n0, fill::zeros);
  arma::vec mse(n0, fill::zeros);
  arma::vec ei(n0, fill::zeros);
  arma::vec ei_1(n0, fill::zeros);
  arma::vec ei_2(n0, fill::zeros);
  aIntNewData(y0, mse, ei, ei_1, ei_2, ei_alpha, min_y, x0, z0, y, x, z, xzDim,
              mu, invPsi, thetaZ, sigmaF, sigmaInt);
  //
  return List::create(Named("pred") = wrap(y0),
                      Named("mse") = wrap(mse),
                      Named("ei") = wrap(ei),
                      Named("improvement") = wrap(ei_1),
                      Named("uncertainty") = wrap(ei_2));
}

// Independence across Stages

//[[Rcpp::export]]
double lgpNvObjCpp(arma::rowvec param, arma::vec y, arma::mat x, arma::uvec z, arma::uword xzDim, double nugget, bool logParam)
{
  arma::uword n = x.n_rows;
  arma::uword zMax = z.max();
  double negloglik, mu, sigma;
  arma::mat psi(n, n, fill::eye);
  arma::mat invPsi(n, n, fill::eye);
  //
  arma::mat theta;
  // double nugget = 0.;
  if (logParam)
  {
    nvParam2vec(theta, arma::exp(param), xzDim, zMax);
  }
  else
  {
    nvParam2vec(theta, param, xzDim, zMax);
  }
  nvLogLik(negloglik, psi, invPsi, mu, sigma, nugget, y, x, z, xzDim, theta);
  return negloglik;
}

//[[Rcpp::export]]
Rcpp::List lgpNvModel(arma::rowvec param, arma::vec y, arma::mat x, arma::uvec z, arma::uword xzDim, double nugget, bool logParam)
{
  arma::uword n = x.n_rows;
  arma::uword zMax = z.max();
  double negloglik, mu, sigma;
  arma::mat psi(n, n, fill::eye);
  arma::mat invPsi(n, n, fill::eye);
  //
  arma::mat theta;
  // double nugget = 0.;
  if (logParam)
  {
    nvParam2vec(theta, arma::exp(param), xzDim, zMax);
  }
  else
  {
    nvParam2vec(theta, param, xzDim, zMax);
  }
  nvLogLik(negloglik, psi, invPsi, mu, sigma, nugget, y, x, z, xzDim, theta);
  //
  return List::create(Named("mu") = wrap(mu),
                      Named("sigma2") = wrap(sigma),
                      Named("theta") = wrap(theta),
                      Named("psi") = wrap(psi),
                      Named("invPsi") = wrap(invPsi),
                      Named("negloglik") = wrap(negloglik),
                      Named("nugget") = wrap(nugget),
                      Named("vecParams") = wrap(param));
}

//[[Rcpp::export]]
Rcpp::List lgpNvPred(arma::mat x0, arma::uvec z0, arma::vec y, arma::mat x, arma::uvec z, arma::uword xzDim,
                     arma::rowvec param, arma::mat invPsi, double mu, double sigma, double ei_alpha, double min_y, bool logParam)
{
  arma::uword n0 = x0.n_rows;
  arma::uword zMax = z.max();
  //
  arma::mat theta;
  if (logParam)
  {
    nvParam2vec(theta, arma::exp(param), xzDim, zMax);
  }
  else
  {
    nvParam2vec(theta, param, xzDim, zMax);
  }
  //
  arma::vec y0(n0, fill::zeros);
  arma::vec mse(n0, fill::zeros);
  arma::vec ei(n0, fill::zeros);
  arma::vec ei_1(n0, fill::zeros);
  arma::vec ei_2(n0, fill::zeros);
  nvNewData(y0, mse, ei, ei_1, ei_2, ei_alpha, min_y, x0, z0, y, x, z, xzDim,
            mu, sigma, invPsi, theta);
  //
  return List::create(Named("pred") = wrap(y0),
                      Named("mse") = wrap(mse),
                      Named("ei") = wrap(ei),
                      Named("improvement") = wrap(ei_1),
                      Named("uncertainty") = wrap(ei_2));
}

// =============================================================================
// R-facing functions for SAL-GP (Softmax Attention Ladder GP)
// =============================================================================

//[[Rcpp::export]]
double SALGPObjCpp(arma::rowvec param, arma::vec y, arma::mat x, arma::uvec z, arma::uword xzDim, double nugget, bool logParam)
{
  arma::uword zMax = z.max();
  double negloglik, mu;
  arma::mat psi, invPsi;
  bool inv_success_status = false;

  arma::mat theta;
  arma::vec logits;

  // Use a simplified parameter parsing for SAL-GP
  if (logParam)
  {
    arma::rowvec exp_param = arma::exp(param);
    arma::uword nTheta = xzDim * zMax;
    // only exp the theta part, not the logits
    param.subvec(0, nTheta - 1) = exp_param.subvec(0, nTheta - 1);
  }
  salgpParam2vec(theta, logits, param, xzDim, zMax);

  salgpLogLik(negloglik, psi, invPsi, mu, nugget, inv_success_status,
              y, x, z, xzDim, theta, logits);

  return negloglik;
}

//[[Rcpp::export]]
Rcpp::List SALGPModel(arma::rowvec param, arma::vec y, arma::mat x, arma::uvec z, arma::uword xzDim, double nugget, bool logParam)
{
  arma::uword zMax = z.max();
  double negloglik, mu;
  arma::mat psi, invPsi;
  bool inv_success_status = false;

  arma::mat theta;
  arma::vec logits;

  arma::rowvec current_param = param;
  if (logParam)
  {
    arma::uword nTheta = xzDim * zMax;
    arma::rowvec exp_theta_part = arma::exp(param.subvec(0, nTheta - 1));
    current_param.subvec(0, nTheta - 1) = exp_theta_part;
  }
  salgpParam2vec(theta, logits, current_param, xzDim, zMax);

  salgpLogLik(negloglik, psi, invPsi, mu, nugget, inv_success_status,
              y, x, z, xzDim, theta, logits);

  return Rcpp::List::create(
      Rcpp::Named("mu") = wrap(mu),
      Rcpp::Named("theta") = wrap(theta),
      Rcpp::Named("logits") = wrap(logits),
      Rcpp::Named("psi") = wrap(psi),
      Rcpp::Named("invPsi") = wrap(invPsi),
      Rcpp::Named("negloglik") = wrap(negloglik),
      Rcpp::Named("nugget") = wrap(nugget),
      Rcpp::Named("vecParams") = wrap(param),
      Rcpp::Named("inv_success") = wrap(inv_success_status));
}

// [[Rcpp::export]]
Rcpp::List SALGPPredCpp(arma::mat x0, arma::uvec z0, arma::vec y, arma::mat x, arma::uvec z, arma::uword xzDim,
                        arma::rowvec param, arma::mat invPsi, double mu, bool logParam)
{

  // ===== 偵錯訊息 Stage 1: 函數進入點 =====
  Rcpp::Rcout << "\n--- DEBUG: Entered SALGPPredCpp function ---\n";

  arma::uword n0 = x0.n_rows;
  arma::uword zMax = z.max();

  // ===== 偵錯訊息 Stage 2: 變數維度檢查 =====
  Rcpp::Rcout << "  - n0 (new points): " << n0 << "\n";
  Rcpp::Rcout << "  - zMax (max stages): " << zMax << "\n";
  Rcpp::Rcout << "  - xzDim (base dimension): " << xzDim << "\n";
  Rcpp::Rcout << "  - param vector length: " << param.n_elem << "\n";

  arma::mat theta_mat;
  arma::vec logits;

  // ===== 偵錯訊息 Stage 3: 參數解析前 =====
  Rcpp::Rcout << "  - About to parse parameters...\n";
  arma::rowvec current_param = param;
  if (logParam)
  {
    arma::uword nMainLogits = zMax;
    arma::uword nInterLogits = (zMax > 1) ? (0.5 * zMax * (zMax - 1)) : 0;
    arma::uword nAttnPar = nMainLogits + nInterLogits;
    arma::uword nTheta = param.n_elem - nAttnPar;

    // ===== 偵錯訊息 Stage 3.1: Log 轉換前 =====
    Rcpp::Rcout << "    - logParam is TRUE. nTheta=" << nTheta << ", nAttnPar=" << nAttnPar << "\n";

    arma::rowvec exp_theta_part = arma::exp(param.subvec(0, nTheta - 1));
    current_param.subvec(0, nTheta - 1) = exp_theta_part;

    // ===== 偵錯訊息 Stage 3.2: Log 轉換後 =====
    Rcpp::Rcout << "    - Parameter log transformation complete.\n";
  }
  salgpParam2vec(theta_mat, logits, current_param, xzDim, zMax);

  // ===== 偵錯訊息 Stage 4: 參數解析後 =====
  Rcpp::Rcout << "  - Parameter parsing complete.\n";
  Rcpp::Rcout << "    - theta_mat dimensions: " << theta_mat.n_rows << "x" << theta_mat.n_cols << "\n";
  Rcpp::Rcout << "    - logits dimensions: " << logits.n_rows << "x" << logits.n_cols << "\n";

  arma::vec y0(n0, arma::fill::zeros);
  arma::vec mse(n0, arma::fill::zeros);

  // ===== 偵錯訊息 Stage 5: 呼叫核心預測函數前 =====
  Rcpp::Rcout << "  - About to call salgpNewData...\n";

  salgpNewData(y0, mse, x0, z0, y, x, z, xzDim, mu, invPsi, theta_mat, logits);

  // ===== 偵錯訊息 Stage 6: 核心預測函數執行完畢 =====
  Rcpp::Rcout << "  - Call to salgpNewData finished.\n";

  Rcpp::List result = Rcpp::List::create(
      Rcpp::Named("pred") = wrap(y0),
      Rcpp::Named("mse") = wrap(mse));

  // ===== 偵錯訊息 Stage 7: 準備返回 R =====
  Rcpp::Rcout << "--- DEBUG: Exiting SALGPPredCpp successfully ---\n";

  return result;
}