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

struct GLnRResult
{
  bool success = false;
  std::string message;

  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();

  std::size_t num_pairs = 0;
  double rmse = std::numeric_limits<double>::quiet_NaN();
};

inline Eigen::Matrix3d skewFromGVector(const Eigen::Vector3d& g)
{
  // 3D specialization of the paper's "g⊗" mapping (Eq. (12))
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
  // Matrix P(x) such that G(g) * x = P(x) * g.
  // Derived from G(g) in skewFromGVector (Eq. (8) -> Eq. (13) in the paper).
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
    const std::vector<double>* weights = nullptr,
    bool enforce_so3 = true)
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

  // Compute centroids (Eq. (5)).
  Eigen::Vector3d b_bar = Eigen::Vector3d::Zero();
  Eigen::Vector3d r_bar = Eigen::Vector3d::Zero();
  for (std::size_t i = 0; i < N; ++i) {
    b_bar += w[i] * tgt_points[i];
    r_bar += w[i] * src_points[i];
  }

  // Accumulate H and v (Eq. (16)).
  Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
  Eigen::Vector3d v = Eigen::Vector3d::Zero();

  for (std::size_t i = 0; i < N; ++i) {
    const Eigen::Vector3d b_tilde = tgt_points[i] - b_bar;
    const Eigen::Vector3d r_tilde = src_points[i] - r_bar;

    // Eq. (11): x_i = b~ + r~, d_i = r~ - b~
    const Eigen::Vector3d x = b_tilde + r_tilde;
    const Eigen::Vector3d d = r_tilde - b_tilde;

    const Eigen::Matrix3d P = P3(x);

    H.noalias() += w[i] * (P.transpose() * P);
    v.noalias() += w[i] * (P.transpose() * d);
  }

  // Solve g = H^{-1} v (Eq. (15)).
  // Use SVD for robustness (covers singular / near-singular H).
  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Vector3d g = svd.solve(v);

  const Eigen::Matrix3d G = skewFromGVector(g);

  // Reconstruct R via Cayley transform (Eq. (17)): R = (I + G)^{-1}(I - G)
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d R = (I + G).colPivHouseholderQr().solve(I - G);

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
  out.rmse = computeRMSE(src_points, tgt_points, R, t);
  return out;
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

}  // namespace glnr
