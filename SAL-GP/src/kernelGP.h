// HEADER
double gpCorrKern(const arma::rowvec &xi, const arma::rowvec &xj, const arma::vec &alpha);

void gpCorrMat(arma::mat &psi, const arma::mat &x, const arma::vec &alpha);

void gpCorrVecs(arma::mat &phi, const arma::mat &x0, const arma::mat &x, const arma::vec &alpha);

void gpLogLik(double &negloglik, arma::mat &psi, arma::mat &invPsi, double &mu, double &sigma, double &nugget,
              const arma::vec &y, const arma::mat &x, const arma::vec &alpha);

void gpNewData(arma::vec &y0, arma::vec &mse, arma::vec &ei, arma::vec &ei_1, arma::vec &ei_2, double &ei_alpha, double &min_y,
               const arma::mat &x0, const arma::vec &y, const arma::mat &x, 
               double &mu, double &sigma, arma::mat &invPsi, const arma::vec &alpha);

// BODY
// CORRELATION KERNEL OF GAUSSIAN PROCESS
double gpCorrKern(const arma::rowvec &xi, const arma::rowvec &xj, const arma::vec &alpha) 
{
  arma::rowvec xDiffSq = arma::pow(xi - xj, 2);
  double rx = arma::as_scalar(xDiffSq*alpha);
  double val = std::exp(-(1.0)*rx);
  return val;
}


void gpCorrMat(arma::mat &psi, const arma::mat &x, const arma::vec &alpha)
{
  arma::uword n = x.n_rows;
  for (uword i = 0; i < n; i++) {
    for (uword j = 0; j < i; j++) {
      arma::rowvec xi = x.row(i); arma::rowvec xj = x.row(j);
      double ker = gpCorrKern(xi, xj, alpha);
      psi(i, j) = ker;
      psi(j, i) = ker;
    }
  }
}

void gpCorrVecs(arma::mat &phi, const arma::mat &x0, const arma::mat &x, const arma::vec &alpha)
{
  arma::uword n = x.n_rows;
  arma::uword n0 = x0.n_rows;
  for (uword j = 0; j < n0; j++) {
    arma::rowvec x0j = x0.row(j); 
    for (uword i = 0; i < n; i++) {
      arma::rowvec xi = x.row(i); 
      double ker = gpCorrKern(xi, x0j, alpha);
      phi(i, j) = ker;
    }
  }
}

void gpLogLik(double &negloglik, arma::mat &psi, arma::mat &invPsi, double &mu, double &sigma, double &nugget,
              const arma::vec &y, const arma::mat &x, const arma::vec &alpha)
{
  arma::uword n = y.n_elem;
  double n_double = (double)n;
  arma::vec onevec(n, fill::ones);
  gpCorrMat(psi, x, alpha);
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
  invSucc = arma::inv_sympd(invPsi, psi);
  arma::log_det(detPsi, signDetPsi, psi);
  //if (std::isfinite(detPsi) & (signDetPsi >= 0)) 
  if (invSucc) {
    mu = arma::as_scalar(onevec.t()*invPsi*y)/arma::as_scalar(onevec.t()*invPsi*onevec);
    arma::vec res = y - (mu*onevec);
    sigma = arma::as_scalar(res.t()*invPsi*res)/n_double;
    negloglik = .5*(n_double*std::log(sigma + datum::eps) + detPsi);
  } else {
    negloglik = 1e20;
  }
}


void gpNewData(arma::vec &y0, arma::vec &mse, arma::vec &ei, arma::vec &ei_1, arma::vec &ei_2, double &ei_alpha, double &min_y,
               const arma::mat &x0, const arma::vec &y, const arma::mat &x, 
               double &mu, double &sigma, arma::mat &invPsi, const arma::vec &alpha)
{
  arma::uword n = x.n_rows;
  arma::uword n0 = x0.n_rows;
  arma::mat phi(n, n0, fill::zeros);
  gpCorrVecs(phi, x0, x, alpha);
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

