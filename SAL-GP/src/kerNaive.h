// HEADER
void nvParam2vec(arma::mat &thetaZ, 
                  const arma::rowvec &param, const arma::uword &xzDim, const arma::uword &zMax);
                  
double nvCorrKern(const arma::rowvec &xi, const arma::rowvec &xj, const arma::uword &zi, const arma::uword &zj, const arma::uword &xzDim, 
                   const arma::mat &thetaZ);

void nvCorrMat(arma::mat &psi, const arma::mat &x, const arma::uvec &z, const arma::uword &xzDim, 
                const arma::mat &thetaZ);

void nvCorrVecs(arma::mat &phi, const arma::mat &x0, const arma::uvec &z0, 
                 const arma::mat &x, const arma::uvec &z, const arma::uword &xzDim, 
                 const arma::mat &thetaZ);
  
void nvLogLik(double &negloglik, arma::mat &psi, arma::mat &invPsi, double &mu, double &sigma, double &nugget,
               const arma::vec &y, const arma::mat &x, const arma::uvec &z, const arma::uword &xzDim, 
               const arma::mat &thetaZ);

void nvNewData(arma::vec &y0, arma::vec &mse, arma::vec &ei, arma::vec &ei_1, arma::vec &ei_2, double &ei_alpha, double &min_y,
                const arma::mat &x0, const arma::uvec &z0, const arma::vec &y, const arma::mat &x, const arma::uvec &z, const arma::uword &xzDim, 
                double &mu, double &sigma, arma::mat &invPsi, const arma::mat &thetaZ);

// BODY
void nvParam2vec(arma::mat &thetaZ,  
                  const arma::rowvec &param, const arma::uword &xzDim, const arma::uword &zMax)
{
  /* 
   START ASSIGN PARAMETER POSITION 
   */
  thetaZ.set_size(zMax, xzDim); 
  /* 
   Parameters for Continuous variables
   */
  arma::uword n_thetaZ = zMax*xzDim;
  thetaZ = arma::reshape(param.subvec(0, n_thetaZ - 1), zMax, xzDim);
}

// CORRELATION KERNEL OF GAUSSIAN PROCESS
double nvCorrKern(const arma::rowvec &xi, const arma::rowvec &xj, const arma::uword &zi, const arma::uword &zj, const arma::uword &xzDim, 
                   const arma::mat &thetaZ) 
{
  /* corr X*/
  arma::rowvec xDiffSq = arma::pow(xi - xj, 2);
  arma::uword zComm = 0;
  if (zi > zj) { zComm = zj; } else { zComm = zi; }
  double logCorrX = 0.0;
  for (arma::uword i = 0; i < zComm; i++) {
    arma::rowvec xdtmp = xDiffSq.subvec(i*xzDim, (i+1)*xzDim - 1);
    logCorrX += arma::accu(thetaZ.row(i) % xdtmp);
  }
  double val = std::exp((-1.0)*logCorrX);
  return val;
}


void nvCorrMat(arma::mat &psi, const arma::mat &x, const arma::uvec &z, const arma::uword &xzDim, 
                const arma::mat &thetaZ)
{
  arma::uword n = x.n_rows;
  for (uword i = 0; i < n; i++) {
    for (uword j = 0; j < i; j++) {
      arma::rowvec xi = x.row(i); 
      arma::rowvec xj = x.row(j);
      arma::uword zi = z(i);
      arma::uword zj = z(j);
      double ker = nvCorrKern(xi, xj, zi, zj, xzDim, thetaZ);
      psi(i, j) = ker;
      psi(j, i) = ker;
    }
  }
}

void nvCorrVecs(arma::mat &phi, const arma::mat &x0, const arma::uvec &z0, 
                 const arma::mat &x, const arma::uvec &z, const arma::uword &xzDim, 
                 const arma::mat &thetaZ)
{
  arma::uword n = x.n_rows;
  arma::uword n0 = x0.n_rows;
  for (uword j = 0; j < n0; j++) {
    arma::rowvec x0j = x0.row(j);
    arma::uword z0j = z0(j);
    for (uword i = 0; i < n; i++) {
      arma::rowvec xi = x.row(i); 
      arma::uword zi = z(i); 
      double ker = nvCorrKern(xi, x0j, zi, z0j, xzDim, thetaZ);
      phi(i, j) = ker;
    }
  }
}


void nvLogLik(double &negloglik, arma::mat &psi, arma::mat &invPsi, double &mu, double &sigma, double &nugget,
               const arma::vec &y, const arma::mat &x, const arma::uvec &z, const arma::uword &xzDim, 
               const arma::mat &thetaZ)
{
  arma::uword n = y.n_elem;
  double n_double = (double)n;
  arma::vec onevec(n, fill::ones);
  nvCorrMat(psi, x, z, xzDim, thetaZ);
  bool invSucc = 0;
  calcMatrixInv(invSucc, invPsi, psi, nugget);
  double detPsi;
  double signDetPsi;
  arma::log_det(detPsi, signDetPsi, psi);
  //if (std::isfinite(detPsi) & (signDetPsi >= 0)) 
  if (invSucc) {
    //double yPsiY = arma::as_scalar(y.t()*invPsi*y);
    double onePsiY = arma::as_scalar(onevec.t()*invPsi*y);
    double onePsiOne = arma::as_scalar(onevec.t()*invPsi*onevec);
    mu = onePsiY/onePsiOne;
    arma::vec res = y - (mu*onevec);
    sigma = arma::as_scalar(res.t()*invPsi*res)/n_double;
    negloglik = .5*(n_double*std::log(sigma + datum::eps) + detPsi);
  } else {
    negloglik = 1e20;
  }
}


void nvNewData(arma::vec &y0, arma::vec &mse, arma::vec &ei, arma::vec &ei_1, arma::vec &ei_2, double &ei_alpha, double &min_y,
                const arma::mat &x0, const arma::uvec &z0, const arma::vec &y, const arma::mat &x, const arma::uvec &z, const arma::uword &xzDim, 
                double &mu, double &sigma, arma::mat &invPsi, const arma::mat &thetaZ)
{
  arma::uword n = x.n_rows;
  arma::uword n0 = x0.n_rows;
  arma::mat phi(n, n0, fill::zeros);
  nvCorrVecs(phi, x0, z0, x, z, xzDim, thetaZ);
  arma::vec onevec(n, fill::ones);
  arma::vec resid = y - mu*onevec; 
  arma::vec psiinvresid = invPsi*resid;
  for (uword j = 0; j < n0; j++) {
    y0(j) = mu + arma::as_scalar(phi.col(j).t()*psiinvresid);
    mse(j) = std::abs(sigma*(1. - arma::as_scalar(phi.col(j).t()*invPsi*phi.col(j)))) + datum::eps;
  }
  // Compute expected improvement
  //double min_val = arma::min(y);
  arma::vec rmse = arma::sqrt(mse);
  arma::vec yd = min_y - y0;
  // The improvement part
  ei_1 = yd % (.5 + .5*arma::erf((1./std::sqrt(2.))*(yd/rmse)));
  // The uncertainty part
  ei_2 = (rmse/std::sqrt(2.*datum::pi)) % arma::exp(-.5*(yd % yd)/mse);
  // The EI value
  ei = 2.*(ei_alpha*ei_1 + (1. - ei_alpha)*ei_2);
  ei.elem( arma::find(ei <= .0) ).fill(datum::eps);  
}

