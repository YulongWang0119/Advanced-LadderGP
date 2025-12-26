// Rcpp Header File
#include <cmath>
#include <RcppArmadillo.h>

using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

// DECLARE FUNCTIONS
void matrixPrintf(const mat &m);
void vecPrintf(const vec &v);
void rvecPrintf(const rowvec &v);
arma::field<arma::mat> list2field(const Rcpp::List vList);
Rcpp::List field2list(const arma::field<arma::mat> gammas);

#include "common.h"
#include "kernelGP.h"
#include "kerOB.h"
#include "kerNB.h"
#include "kerAInt.h"
#include "kerNaive.h"
#include "kerSALGP.h" // <-- 新增

// BODY
arma::field<arma::mat> list2field(const Rcpp::List vList)
{
  int n = vList.size();
  arma::field<arma::mat> vfield(n, 1);
  for (int i = 0; i < n; i++)
  {
    Rcpp::NumericMatrix vmat_r = Rcpp::as<Rcpp::NumericMatrix>(vList[i]);
    arma::mat vmat(vmat_r.begin(), vmat_r.nrow(), vmat_r.ncol(), false);
    vfield(i, 0) = vmat;
  }
  return vfield;
}

Rcpp::List field2list(const arma::field<arma::mat> gammas)
{
  int n = gammas.n_rows;
  Rcpp::List gammalist(n);
  for (int i = 0; i < n; i++)
  {
    gammalist[i] = Rcpp::wrap(gammas(i));
  }
  return gammalist;
}