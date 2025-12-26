// =============================================================================
// File: kerSALGP.h
// Kernel functions for Softmax Additive Ladder Gaussian Process (SAL-GP)
// ANISOTROPIC VERSION
// =============================================================================

#ifndef KERSALGP_H
#define KERSALGP_H

#include <RcppArmadillo.h>

// --- Helper function to parse the flat parameter vector into structured components ---
// No changes needed here. R-side code already provides the correct number of parameters.
void salgpParam2vec(arma::mat &theta, arma::vec &logits, const arma::rowvec &param, arma::uword xzDim, arma::uword zMax)
{
  arma::uword nTheta = xzDim * zMax;
  theta = arma::reshape(param.subvec(0, nTheta - 1), xzDim, zMax).t();
  logits = param.subvec(nTheta, param.n_elem - 1).t();
}

// --- Main function to calculate the Negative Log-Likelihood for SAL-GP ---
void salgpLogLik(double &negloglik, arma::mat &psi, arma::mat &invPsi, double &mu,
                 double nugget, bool &inv_success_status,
                 const arma::vec &y, const arma::mat &x, const arma::uvec &z, arma::uword xzDim,
                 const arma::mat &theta, const arma::vec &logits)
{
  arma::uword n = x.n_rows;
  arma::uword zMax = z.max();

  // --- 1. Calculate attention weights using Softmax ---
  arma::vec exp_logits = arma::exp(logits);
  double Z_norm = arma::sum(exp_logits);
  arma::vec weights = exp_logits / Z_norm;

  arma::vec w_main = weights.subvec(0, zMax - 1);
  arma::vec w_inter = weights.subvec(zMax, weights.n_elem - 1);

  // --- 2. Build the correlation matrix Psi ---
  psi.set_size(n, n);
  for (arma::uword i = 0; i < n; ++i)
  {
    for (arma::uword j = 0; j <= i; ++j)
    {
      arma::uword zi = z(i);
      arma::uword zj = z(j);
      arma::uword min_z = std::min(zi, zj);
      double psi_ij = 0.0;

      // --- A. Weighted Main Effects (MODIFIED FOR ANISOTROPIC KERNEL) ---
      for (arma::uword k = 1; k <= min_z; ++k)
      {
        arma::uword start_idx = (k - 1) * xzDim;
        arma::uword end_idx = k * xzDim - 1;

        // *** MODIFICATION 1 of 4: Anisotropic kernel for Ck in salgpLogLik ***
        arma::rowvec x_diff_sq = arma::square(x.row(i).subvec(start_idx, end_idx) - x.row(j).subvec(start_idx, end_idx));
        arma::rowvec theta_k_row = theta.row(k - 1);
        double weighted_dist_sq = arma::accu(theta_k_row % x_diff_sq);
        double Ck = std::exp(-weighted_dist_sq);

        psi_ij += w_main(k - 1) * Ck;
      }

      // --- B. Weighted Interaction Effects (MODIFIED FOR ANISOTROPIC KERNEL) ---
      if (min_z > 1)
      {
        int inter_idx_counter = 0;
        for (arma::uword u = 1; u < min_z; ++u)
        {
          for (arma::uword v = u + 1; v <= min_z; ++v)
          {
            arma::uword u_start = (u - 1) * xzDim;
            arma::uword u_end = u * xzDim - 1;

            // *** MODIFICATION 2 of 4: Anisotropic kernel for Cu and Cv in salgpLogLik ***
            arma::rowvec x_diff_sq_u = arma::square(x.row(i).subvec(u_start, u_end) - x.row(j).subvec(u_start, u_end));
            arma::rowvec theta_u_row = theta.row(u - 1);
            double weighted_dist_sq_u = arma::accu(theta_u_row % x_diff_sq_u);
            double Cu = std::exp(-weighted_dist_sq_u);

            arma::uword v_start = (v - 1) * xzDim;
            arma::uword v_end = v * xzDim - 1;

            arma::rowvec x_diff_sq_v = arma::square(x.row(i).subvec(v_start, v_end) - x.row(j).subvec(v_start, v_end));
            arma::rowvec theta_v_row = theta.row(v - 1);
            double weighted_dist_sq_v = arma::accu(theta_v_row % x_diff_sq_v);
            double Cv = std::exp(-weighted_dist_sq_v);

            psi_ij += w_inter(inter_idx_counter) * (Cu * Cv);
            inter_idx_counter++;
          }
        }
      }

      psi(i, j) = psi_ij;
      psi(j, i) = psi_ij;
    }
  }
  psi.diag() += nugget;

  // --- 3. Calculate Negative Log-Likelihood ---
  arma::vec one_vec(n, arma::fill::ones);
  double term_top;
  inv_success_status = arma::inv_sympd(invPsi, psi);

  if (!inv_success_status)
  {
    negloglik = std::numeric_limits<double>::infinity();
    return;
  }

  term_top = arma::as_scalar(one_vec.t() * invPsi * y);
  mu = term_top / arma::as_scalar(one_vec.t() * invPsi * one_vec);
  double term1 = arma::as_scalar((y - mu * one_vec).t() * invPsi * (y - mu * one_vec));

  double val_det_psi, log_det_psi;
  arma::log_det(val_det_psi, log_det_psi, psi);

  negloglik = 0.5 * n * log(term1 / n) + 0.5 * log_det_psi;
}

// --- Function for prediction at new data points for SAL-GP ---
void salgpNewData(arma::vec &y0, arma::vec &mse,
                  const arma::mat &x0, const arma::uvec &z0,
                  const arma::vec &y, const arma::mat &x, const arma::uvec &z, arma::uword xzDim,
                  double mu, const arma::mat &invPsi,
                  const arma::mat &theta, const arma::vec &logits)
{
  // This section contains debugging messages, leave them as is.
  Rcpp::Rcout << "\n--- DEBUG: Entering salgpNewData ---" << std::endl;
  arma::uword n0 = x0.n_rows;
  arma::uword n = x.n_rows;
  arma::uword zMax = z.max();
  Rcpp::Rcout << "  - n0 (new points): " << n0 << ", n (train points): " << n << std::endl;
  Rcpp::Rcout << "  - zMax: " << zMax << ", xzDim: " << xzDim << std::endl;
  Rcpp::Rcout << "  - theta dimensions: " << theta.n_rows << "x" << theta.n_cols << std::endl;
  Rcpp::Rcout << "  - logits dimensions: " << logits.n_rows << "x" << logits.n_cols << std::endl;

  // --- 1. Calculate attention weights (same as in LogLik) ---
  arma::vec exp_logits = arma::exp(logits);
  double Z_norm = arma::sum(exp_logits);
  arma::vec weights = exp_logits / Z_norm;
  arma::vec w_main = weights.subvec(0, zMax - 1);
  arma::vec w_inter = weights.subvec(zMax, weights.n_elem - 1);

  // --- 2. Build the cross-correlation matrix psi0 ---
  arma::mat psi0(n0, n, arma::fill::zeros);
  for (arma::uword i = 0; i < n0; ++i)
  {
    Rcpp::Rcout << "  - Processing new point i = " << i << std::endl;
    for (arma::uword j = 0; j < n; ++j)
    {
      arma::uword zi = z0(i);
      arma::uword zj = z(j);
      arma::uword min_z = std::min(zi, zj);
      double psi_ij = 0.0;

      // Weighted Main Effects (MODIFIED FOR ANISOTROPIC KERNEL)
      for (arma::uword k = 1; k <= min_z; ++k)
      {
        arma::uword start_idx = (k - 1) * xzDim;
        arma::uword end_idx = k * xzDim - 1;

        // *** MODIFICATION 3 of 4: Anisotropic kernel for Ck in salgpNewData ***
        arma::rowvec x_diff_sq = arma::square(x0.row(i).subvec(start_idx, end_idx) - x.row(j).subvec(start_idx, end_idx));
        arma::rowvec theta_k_row = theta.row(k - 1);
        double weighted_dist_sq = arma::accu(theta_k_row % x_diff_sq);
        double Ck = std::exp(-weighted_dist_sq);

        psi_ij += w_main(k - 1) * Ck;
      }

      // Weighted Interaction Effects (MODIFIED FOR ANISOTROPIC KERNEL)
      if (min_z > 1)
      {
        int inter_idx_counter = 0;
        for (arma::uword u = 1; u < min_z; ++u)
        {
          for (arma::uword v = u + 1; v <= min_z; ++v)
          {
            arma::uword u_start = (u - 1) * xzDim;
            arma::uword u_end = u * xzDim - 1;

            // *** MODIFICATION 4 of 4: Anisotropic kernel for Cu and Cv in salgpNewData ***
            // Notice the use of x0.row(i) and x.row(j)
            arma::rowvec x_diff_sq_u = arma::square(x0.row(i).subvec(u_start, u_end) - x.row(j).subvec(u_start, u_end));
            arma::rowvec theta_u_row = theta.row(u - 1);
            double weighted_dist_sq_u = arma::accu(theta_u_row % x_diff_sq_u);
            double Cu = std::exp(-weighted_dist_sq_u);

            arma::uword v_start = (v - 1) * xzDim;
            arma::uword v_end = v * xzDim - 1;

            arma::rowvec x_diff_sq_v = arma::square(x0.row(i).subvec(v_start, v_end) - x.row(j).subvec(v_start, v_end));
            arma::rowvec theta_v_row = theta.row(v - 1);
            double weighted_dist_sq_v = arma::accu(theta_v_row % x_diff_sq_v);
            double Cv = std::exp(-weighted_dist_sq_v);

            psi_ij += w_inter(inter_idx_counter) * (Cu * Cv);
            inter_idx_counter++;
          }
        }
      }
      psi0(i, j) = psi_ij;
    }
  }
  Rcpp::Rcout << "  - Successfully built psi0 matrix (" << psi0.n_rows << "x" << psi0.n_cols << ")" << std::endl;

  // --- 3. Calculate prediction mean and variance ---
  arma::vec one_vec(n, arma::fill::ones);
  y0 = mu * arma::vec(n0, arma::fill::ones) + psi0 * invPsi * (y - mu * one_vec);

  mse = 1.0 - arma::sum((psi0 * invPsi) % psi0, 1);
  mse.elem(find(mse < 0)).zeros();

  Rcpp::Rcout << "--- DEBUG: Exiting salgpNewData Successfully ---" << std::endl;
}

#endif // KERSALGP_H