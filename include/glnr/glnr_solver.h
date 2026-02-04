#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <cstddef>
#include <limits>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace glnr {

/**
 * Noise model for covariance propagation (paper Section II-E).
 *
 * We model per-point measurement noise as zero-mean Gaussian with
 * constant covariance for all points in each set:
 *   r_i = r_i_true + n_r,  n_r ~ N(0, Sigma_src_point)
 *   b_i = b_i_true + n_b,  n_b ~ N(0, Sigma_tgt_point)
 *
 * Assumptions match the paper: points are independent within a set, and
 * the two sets are independent. (High-order terms ignored.)
 */
struct GLnRNoiseModel
{
  Eigen::Matrix3d Sigma_src_point = Eigen::Matrix3d::Zero();  // Σ_{r_i}
  Eigen::Matrix3d Sigma_tgt_point = Eigen::Matrix3d::Zero();  // Σ_{b_i}

  static GLnRNoiseModel isotropic(double sigma_src, double sigma_tgt)
  {
    GLnRNoiseModel nm;
    nm.Sigma_src_point = (sigma_src * sigma_src) * Eigen::Matrix3d::Identity();
    nm.Sigma_tgt_point = (sigma_tgt * sigma_tgt) * Eigen::Matrix3d::Identity();
    return nm;
  }
};

struct GLnRResult
{
  bool success = false;
  std::string message;

  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();

  // Internal GLnR parameters (3D specialization)
  Eigen::Vector3d g = Eigen::Vector3d::Zero();         // Cayley parameter vector (paper Eq. (15))
  Eigen::Matrix3d G = Eigen::Matrix3d::Zero();         // skew-symmetric matrix from g (paper Eq. (12) with n=3)
  Eigen::Matrix3d H = Eigen::Matrix3d::Zero();         // normal matrix (paper Eq. (16), with n=3)
  Eigen::Vector3d v = Eigen::Vector3d::Zero();         // RHS vector (paper Eq. (16), with n=3)
  Eigen::Vector3d src_centroid = Eigen::Vector3d::Zero();  // r̄ (paper Eq. (5))
  Eigen::Vector3d tgt_centroid = Eigen::Vector3d::Zero();  // b¯ (paper Eq. (5))

  // Covariance outputs (paper Eq. (37), (40), (41))
  bool has_covariance = false;
  Eigen::Matrix3d cov_g = Eigen::Matrix3d::Zero();  // Σ_g
  Eigen::Matrix3d cov_R = Eigen::Matrix3d::Zero();  // Σ_R = ⟨δR δRᵀ⟩
  Eigen::Matrix3d cov_t = Eigen::Matrix3d::Zero();  // Σ_t

  std::size_t num_pairs = 0;
  double rmse = std::numeric_limits<double>::quiet_NaN();
};

inline Eigen::Matrix3d skewFromGVector(const Eigen::Vector3d& g)
{
  // 3D specialization of the paper's "g⊗" mapping (paper Eq. (12))
  // G = [  0   g1  g2
  //       -g1  0   g3
  //       -g2 -g3  0 ]
  Eigen::Matrix3d G;
  G << 0.0,   g(0),  g(1),
      -g(0),  0.0,   g(2),
      -g(1), -g(2),  0.0;
  return G;
}

inline Eigen::Matrix3d P3(const Eigen::Vector3d& x)
{
  // Matrix P(x) such that G(g) * x = P(x) * g (paper Eq. (13)).
  // For x = (x1,x2,x3):
  //   P = [ x2  x3   0
  //        -x1  0   x3
  //         0  -x1 -x2 ]
  Eigen::Matrix3d P;
  P << x(1),  x(2),  0.0,
      -x(0),  0.0,   x(2),
       0.0,  -x(0), -x(1);
  return P;
}

inline Eigen::Matrix3d pseudoInverse3x3(const Eigen::Matrix3d& A, double rcond = 1e-12)
{
  // Robust Moore-Penrose pseudo-inverse using SVD.
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Vector3d s = svd.singularValues();
  const double s_max = s.maxCoeff();

  Eigen::Matrix3d S_pinv = Eigen::Matrix3d::Zero();
  for (int i = 0; i < 3; ++i) {
    if (s(i) > rcond * s_max) {
      S_pinv(i, i) = 1.0 / s(i);
    }
  }
  return svd.matrixV() * S_pinv * svd.matrixU().transpose();
}

/**
 * Compute A_i(x_i, g) such that:
 *   J_i g = A_i δx_i
 * where
 *   J_i = w_i ( P(δx_i)^T P(x_i) + P(x_i)^T P(δx_i) )   (paper Eq. (26)-(31))
 *
 * This corresponds to the term (Σ_j g_j K_j(x_i)) in the paper.
 *
 * 3D closed-form derived by symbolic expansion.
 */
inline Eigen::Matrix3d A3(const Eigen::Vector3d& x, const Eigen::Vector3d& g, double w_i)
{
  const double x1 = x(0), x2 = x(1), x3 = x(2);
  const double g1 = g(0), g2 = g(1), g3 = g(2);

  Eigen::Matrix3d A;
  A << (2.0 * g1 * x1 - g3 * x3), (2.0 * g1 * x2 + g2 * x3), (g2 * x2 - g3 * x1),
       (2.0 * g2 * x1 + g3 * x2), (g1 * x3 + g3 * x1),       (g1 * x2 + 2.0 * g2 * x3),
       (-g1 * x3 + g2 * x2),      (g2 * x1 + 2.0 * g3 * x2), (-g1 * x1 + 2.0 * g3 * x3);

  return w_i * A;
}

inline double computeRMSE(const std::vector<Eigen::Vector3d>& src,
                          const std::vector<Eigen::Vector3d>& tgt,
                          const Eigen::Matrix3d& R,
                          const Eigen::Vector3d& t)
{
  if (src.size() != tgt.size() || src.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  double sum_sq = 0.0;
  for (std::size_t i = 0; i < src.size(); ++i) {
    const Eigen::Vector3d e = tgt[i] - (R * src[i] + t);
    sum_sq += e.squaredNorm();
  }

  return std::sqrt(sum_sq / static_cast<double>(src.size()));
}

inline GLnRResult estimateRigidTransformGLnR(
    const std::vector<Eigen::Vector3d>& src_points,
    const std::vector<Eigen::Vector3d>& tgt_points,
    const std::vector<double>* weights,
    bool enforce_so3,
    const GLnRNoiseModel* noise_model)
{
  GLnRResult out;
  out.num_pairs = std::min(src_points.size(), tgt_points.size());

  if (src_points.size() != tgt_points.size()) {
    out.success = false;
    out.message = "src_points and tgt_points must have the same size.";
    return out;
  }

  const std::size_t N = src_points.size();
  if (N < 3) {
    out.success = false;
    out.message = "Need at least 3 point pairs.";
    return out;
  }

  if (weights && weights->size() != N) {
    out.success = false;
    out.message = "weights provided but size != number of point pairs.";
    return out;
  }

  // Normalize weights (paper assumes Σ w_i = 1, Eq. (3)).
  std::vector<double> w;
  w.resize(N, 1.0 / static_cast<double>(N));
  if (weights) {
    double s = 0.0;
    for (double wi : *weights) s += wi;
    if (!(s > 0.0)) {
      out.success = false;
      out.message = "Sum of weights must be positive.";
      return out;
    }
    for (std::size_t i = 0; i < N; ++i) {
      w[i] = (*weights)[i] / s;
    }
  }

  // Compute centroids (paper Eq. (5)).
  Eigen::Vector3d b_bar = Eigen::Vector3d::Zero();
  Eigen::Vector3d r_bar = Eigen::Vector3d::Zero();
  for (std::size_t i = 0; i < N; ++i) {
    b_bar += w[i] * tgt_points[i];
    r_bar += w[i] * src_points[i];
  }

  out.src_centroid = r_bar;
  out.tgt_centroid = b_bar;

  // Accumulate H and v (paper Eq. (16), with n=3).
  Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
  Eigen::Vector3d v = Eigen::Vector3d::Zero();

  std::vector<Eigen::Vector3d> x_list;
  std::vector<Eigen::Vector3d> d_list;
  x_list.reserve(N);
  d_list.reserve(N);

  for (std::size_t i = 0; i < N; ++i) {
    const Eigen::Vector3d b_tilde = tgt_points[i] - b_bar;
    const Eigen::Vector3d r_tilde = src_points[i] - r_bar;

    // paper Eq. (11): x_i = b~ + r~, d_i = r~ - b~
    const Eigen::Vector3d x = b_tilde + r_tilde;
    const Eigen::Vector3d d = r_tilde - b_tilde;
    x_list.push_back(x);
    d_list.push_back(d);

    const Eigen::Matrix3d P = P3(x);

    H.noalias() += w[i] * (P.transpose() * P);
    v.noalias() += w[i] * (P.transpose() * d);
  }

  // Solve g = H^{-1} v (paper Eq. (15)).
  // Use SVD solve (covers singular / near-singular H).
  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Vector3d g = svd.solve(v);
  const Eigen::Matrix3d H_pinv = pseudoInverse3x3(H);

  const Eigen::Matrix3d G = skewFromGVector(g);

  // Reconstruct R via Cayley transform (paper Eq. (17)): R = (I + G)^{-1}(I - G)
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d R_cayley = (I + G).colPivHouseholderQr().solve(I - G);

  Eigen::Matrix3d R = R_cayley;
  if (enforce_so3) {
    // Numerical safety: project to SO(3)
    Eigen::JacobiSVD<Eigen::Matrix3d> svd_R(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd_R.matrixU();
    Eigen::Matrix3d Vt = svd_R.matrixV().transpose();
    R = U * Vt;
    if (R.determinant() < 0.0) {
      // Fix reflection if it occurs.
      U.col(2) *= -1.0;
      R = U * Vt;
    }
  }

  // Translation (paper): t = b̄ - R r̄.
  const Eigen::Vector3d t = b_bar - R * r_bar;

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = R;
  T.block<3, 1>(0, 3) = t;

  out.success = true;
  out.message = "OK";
  out.R = R;
  out.t = t;
  out.T = T;
  out.g = g;
  out.G = G;
  out.H = H;
  out.v = v;
  out.rmse = computeRMSE(src_points, tgt_points, R, t);

  // Optional covariance propagation (paper Section II-E, Eq. (33)-(41)).
  if (noise_model) {
    const Eigen::Matrix3d Sigma_r = noise_model->Sigma_src_point;  // Σ_{r_i}
    const Eigen::Matrix3d Sigma_b = noise_model->Sigma_tgt_point;  // Σ_{b_i}

    const Eigen::Matrix3d Sigma_x = Sigma_b + Sigma_r;   // δx = δb + δr
    const Eigen::Matrix3d Sigma_d = Sigma_b + Sigma_r;   // δd = δr - δb
    const Eigen::Matrix3d Sigma_xd = Sigma_r - Sigma_b;  // paper Eq. (36)

    Eigen::Matrix3d Sigma_v = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d Sigma_Hg = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d Sigma_Hg_v = Eigen::Matrix3d::Zero();

    double sum_w2 = 0.0;
    for (std::size_t i = 0; i < N; ++i) sum_w2 += w[i] * w[i];

    for (std::size_t i = 0; i < N; ++i) {
      const double wi = w[i];
      const Eigen::Vector3d& x = x_list[i];
      const Eigen::Vector3d& d = d_list[i];

      const Eigen::Matrix3d P_x = P3(x);
      const Eigen::Matrix3d P_d = P3(d);

      // Σ_v (paper Eq. (33), 3D specialization)
      Sigma_v.noalias() += (wi * wi) *
        (P_d.transpose() * Sigma_x * P_d
         - P_x.transpose() * Sigma_xd * P_d
         - P_d.transpose() * Sigma_xd * P_x
         + P_x.transpose() * Sigma_d * P_x);

      // A_i term for δHg (paper Eq. (31))
      const Eigen::Matrix3d A = A3(x, g, wi);

      // Cov(δHg) contribution
      Sigma_Hg.noalias() += A * Sigma_x * A.transpose();

      // Cross-covariance Cov(δHg, δv) contribution (from paper Eq. (34))
      // Cov(δx_i, δv_i) = w_i(Σ_{x,d} P(x_i) - Σ_x P(d_i))
      const Eigen::Matrix3d C = wi * (Sigma_xd * P_x - Sigma_x * P_d);  // (n×m) but here n=m=3
      Sigma_Hg_v.noalias() += A * C;
    }

    // Σ_g (paper Eq. (32) general form; reduces to Eq. (37) when Σ_xd=0 and cross terms vanish)
    const Eigen::Matrix3d Sigma_g = H_pinv * (Sigma_v + Sigma_Hg - Sigma_Hg_v - Sigma_Hg_v.transpose()) * H_pinv.transpose();

    // Σ_R (paper Eq. (40))
    const Eigen::Matrix3d inv_IpG = (I + G).colPivHouseholderQr().solve(I);
    const Eigen::Matrix3d S = R_cayley + I;  // use Cayley-consistent R for covariance algebra
    Eigen::Matrix3d inner_R = Eigen::Matrix3d::Zero();
    for (int k = 0; k < 3; ++k) {
      const Eigen::Vector3d sk = S.col(k);
      const Eigen::Matrix3d P_sk = P3(sk);
      inner_R.noalias() += P_sk * Sigma_g * P_sk.transpose();
    }
    const Eigen::Matrix3d Sigma_R = inv_IpG * inner_R * inv_IpG.transpose();

    // Σ_t (paper Eq. (41))
    const Eigen::Matrix3d Sigma_bbar = sum_w2 * Sigma_b;
    const Eigen::Matrix3d Sigma_rbar = sum_w2 * Sigma_r;

    Eigen::Matrix3d inner_t = Eigen::Matrix3d::Zero();
    for (int k = 0; k < 3; ++k) {
      const double rbar_k = r_bar(k);
      const Eigen::Vector3d uk = rbar_k * S.col(k);
      const Eigen::Matrix3d P_uk = P3(uk);
      inner_t.noalias() += P_uk * Sigma_g * P_uk.transpose();
    }

    const Eigen::Matrix3d Sigma_t = Sigma_bbar + (R_cayley * Sigma_rbar * R_cayley.transpose()) + inv_IpG * inner_t * inv_IpG.transpose();

    out.has_covariance = true;
    out.cov_g = Sigma_g;
    out.cov_R = Sigma_R;
    out.cov_t = Sigma_t;
  }

  return out;
}

// Backward-compatible overload (no covariance)
inline GLnRResult estimateRigidTransformGLnR(
    const std::vector<Eigen::Vector3d>& src_points,
    const std::vector<Eigen::Vector3d>& tgt_points,
    const std::vector<double>* weights = nullptr,
    bool enforce_so3 = true)
{
  return estimateRigidTransformGLnR(src_points, tgt_points, weights, enforce_so3, nullptr);
}

inline void printMatrix4d(std::ostream& os, const Eigen::Matrix4d& T, const std::string& prefix = "")
{
  os << prefix;
  os << T(0,0) << " " << T(0,1) << " " << T(0,2) << " " << T(0,3) << "\n";
  os << prefix;
  os << T(1,0) << " " << T(1,1) << " " << T(1,2) << " " << T(1,3) << "\n";
  os << prefix;
  os << T(2,0) << " " << T(2,1) << " " << T(2,2) << " " << T(2,3) << "\n";
  os << prefix;
  os << T(3,0) << " " << T(3,1) << " " << T(3,2) << " " << T(3,3) << "\n";
}

inline void printMatrix3d(std::ostream& os, const Eigen::Matrix3d& M, const std::string& prefix = "")
{
  os << prefix << M(0,0) << " " << M(0,1) << " " << M(0,2) << "\n";
  os << prefix << M(1,0) << " " << M(1,1) << " " << M(1,2) << "\n";
  os << prefix << M(2,0) << " " << M(2,1) << " " << M(2,2) << "\n";
}

}  // namespace glnr
