#include <glnr/glnr_solver.h>

#include <Eigen/Dense>

#include <pcl/console/parse.h>
#include <pcl/console/print.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

static void writeMatrixCSV(const std::string& path, const Eigen::MatrixXd& M)
{
  std::ofstream ofs(path.c_str());
  if (!ofs.is_open()) {
    pcl::console::print_warn("Could not write %s\n", path.c_str());
    return;
  }
  for (int r = 0; r < M.rows(); ++r) {
    for (int c = 0; c < M.cols(); ++c) {
      ofs << M(r, c);
      if (c + 1 < M.cols()) ofs << ",";
    }
    ofs << "\n";
  }
}

static Eigen::Matrix3d randomRotationAngleAxis(std::mt19937& gen, double max_angle_rad)
{
  std::normal_distribution<double> normal(0.0, 1.0);
  Eigen::Vector3d axis(normal(gen), normal(gen), normal(gen));
  const double n = axis.norm();
  if (n < 1e-12) {
    axis = Eigen::Vector3d(1.0, 0.0, 0.0);
  } else {
    axis /= n;
  }

  std::uniform_real_distribution<double> unif(-max_angle_rad, max_angle_rad);
  const double angle = unif(gen);

  return Eigen::AngleAxisd(angle, axis).toRotationMatrix();
}

static Eigen::Vector3d randomTranslation(std::mt19937& gen, double t_scale)
{
  std::uniform_real_distribution<double> unif(-t_scale, t_scale);
  return Eigen::Vector3d(unif(gen), unif(gen), unif(gen));
}

static Eigen::Vector3d cayleyGFromRotation(const Eigen::Matrix3d& R)
{
  // Invert Cayley:  R = (I+G)^{-1}(I-G)  =>  G = (I - R)(R + I)^{-1}
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d G = (I - R) * (R + I).inverse();

  // Our 3D mapping in glnr_solver.h is:
  //   G = [  0   g1  g2
  //        -g1  0   g3
  //        -g2 -g3  0 ]
  return Eigen::Vector3d(G(0, 1), G(0, 2), G(1, 2));
}

static std::vector<Eigen::Vector3d> generateUniformPoints(std::mt19937& gen, int N, double scale)
{
  std::uniform_real_distribution<double> unif(-scale, scale);
  std::vector<Eigen::Vector3d> pts;
  pts.reserve(static_cast<std::size_t>(std::max(0, N)));
  for (int i = 0; i < N; ++i) {
    pts.emplace_back(unif(gen), unif(gen), unif(gen));
  }
  return pts;
}

static std::vector<Eigen::Vector3d> applyTransform(const std::vector<Eigen::Vector3d>& src,
                                                   const Eigen::Matrix3d& R,
                                                   const Eigen::Vector3d& t)
{
  std::vector<Eigen::Vector3d> out;
  out.reserve(src.size());
  for (const auto& p : src) {
    out.emplace_back(R * p + t);
  }
  return out;
}

static void addGaussianNoiseInPlace(std::mt19937& gen, std::vector<Eigen::Vector3d>& pts, double sigma)
{
  if (!(sigma > 0.0)) return;
  std::normal_distribution<double> normal(0.0, sigma);
  for (auto& p : pts) {
    p(0) += normal(gen);
    p(1) += normal(gen);
    p(2) += normal(gen);
  }
}

static Eigen::Matrix3d sampleCov3(const std::vector<Eigen::Vector3d>& deltas)
{
  const std::size_t M = deltas.size();
  if (M < 2) return Eigen::Matrix3d::Zero();

  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (const auto& d : deltas) mean += d;
  mean /= static_cast<double>(M);

  Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
  for (const auto& d : deltas) {
    const Eigen::Vector3d e = d - mean;
    C.noalias() += e * e.transpose();
  }
  C /= static_cast<double>(M - 1);
  return C;
}

static Eigen::Matrix3d sampleSecondMomentRRt(const std::vector<Eigen::Matrix3d>& dR_list)
{
  const std::size_t M = dR_list.size();
  if (M < 2) return Eigen::Matrix3d::Zero();

  Eigen::Matrix3d mean = Eigen::Matrix3d::Zero();
  for (const auto& dR : dR_list) mean += dR;
  mean /= static_cast<double>(M);

  Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
  for (const auto& dR : dR_list) {
    const Eigen::Matrix3d E = dR - mean;
    S.noalias() += E * E.transpose();
  }
  S /= static_cast<double>(M - 1);
  return S;
}

static Eigen::Matrix<double, 6, 6> sampleCov6(const std::vector<Eigen::Matrix<double, 6, 1>>& deltas)
{
  const std::size_t M = deltas.size();
  if (M < 2) return Eigen::Matrix<double, 6, 6>::Zero();

  Eigen::Matrix<double, 6, 1> mean;
  mean.setZero();
  for (const auto& d : deltas) mean += d;
  mean /= static_cast<double>(M);

  Eigen::Matrix<double, 6, 6> C;
  C.setZero();
  for (const auto& d : deltas) {
    const Eigen::Matrix<double, 6, 1> e = d - mean;
    C.noalias() += e * e.transpose();
  }
  C /= static_cast<double>(M - 1);
  return C;
}

static void printUsage(const char* prog)
{
  pcl::console::print_info(
      "Usage:\n"
      "  %s [options]\n\n"
      "Monte-Carlo covariance validation for GLnR with *fixed correspondences* (index-aligned).\n"
      "This follows the paper's simulation model b_i = R_true r_i + t_true + eps (Eq. (42)),\n"
      "repeats registration with different Gaussian noise realizations, computes empirical\n"
      "covariances, and compares them with the theoretical covariances from the GLnR derivation\n"
      "(paper Section II-E, Eq. (32)-(41)).\n\n"
      "Options:\n"
      "  --num_points N         number of point pairs (default 2000)\n"
      "  --trials M             Monte-Carlo trials (default 200)\n"
      "  --seed S               RNG seed (default 0)\n"
      "  --sigma_tgt S          target noise std-dev (meters, default 0.01)\n"
      "  --sigma_src S          source noise std-dev (meters, default 0.0)\n"
      "  --noise_both           set sigma_src = sigma_tgt\n"
      "  --scale L              source point coordinate range ~ U[-L, L] (default 5.0)\n"
      "  --t_scale L            translation range ~ U[-L, L] (default 1.0)\n"
      "  --max_angle_deg A      rotation angle range ~ U[-A, A] (deg, default 90)\n"
      "  --output_prefix P      write CSV outputs with this prefix (default: glnr_mc)\n"
      "  --enforce_so3          project R to SO(3) in each estimate (default: on)\n"
      "\n"
      "Outputs (CSV):\n"
      "  <P>_theory_cov_g.csv, <P>_theory_cov_R.csv, <P>_theory_cov_t.csv\n"
      "  <P>_theory_cov_gt.csv  6x6 block-diagonal theory covariance of [g; t]\n"
      "  <P>_sample_cov_g.csv, <P>_sample_cov_R.csv, <P>_sample_cov_t.csv\n"
      "  <P>_sample_cov_gt.csv  6x6 sample covariance of [delta_g; delta_t]\n",
      prog);
}

}  // namespace

int main(int argc, char** argv)
{
  if (pcl::console::find_switch(argc, argv, "--help") ||
      pcl::console::find_switch(argc, argv, "-h")) {
    printUsage(argv[0]);
    return 0;
  }

  int num_points = 2000;
  int trials = 200;
  int seed = 0;
  double sigma_tgt = 0.01;
  double sigma_src = 0.0;
  double scale = 5.0;
  double t_scale = 1.0;
  double max_angle_deg = 90.0;
  std::string out_prefix = "glnr_mc";
  bool enforce_so3 = true;

  pcl::console::parse_argument(argc, argv, "--num_points", num_points);
  pcl::console::parse_argument(argc, argv, "--trials", trials);
  pcl::console::parse_argument(argc, argv, "--seed", seed);
  pcl::console::parse_argument(argc, argv, "--sigma_tgt", sigma_tgt);
  pcl::console::parse_argument(argc, argv, "--sigma_src", sigma_src);
  pcl::console::parse_argument(argc, argv, "--sigma", sigma_tgt);
  pcl::console::parse_argument(argc, argv, "--scale", scale);
  pcl::console::parse_argument(argc, argv, "--t_scale", t_scale);
  pcl::console::parse_argument(argc, argv, "--max_angle_deg", max_angle_deg);
  pcl::console::parse_argument(argc, argv, "--output_prefix", out_prefix);

  if (pcl::console::find_switch(argc, argv, "--noise_both")) {
    sigma_src = sigma_tgt;
  }
  if (pcl::console::find_switch(argc, argv, "--no_enforce_so3")) {
    enforce_so3 = false;
  }
  if (pcl::console::find_switch(argc, argv, "--enforce_so3")) {
    enforce_so3 = true;
  }

  if (num_points < 3) {
    pcl::console::print_error("--num_points must be >= 3\n");
    return 1;
  }
  if (trials < 2) {
    pcl::console::print_error("--trials must be >= 2\n");
    return 1;
  }
  if (!(sigma_tgt >= 0.0) || !(sigma_src >= 0.0)) {
    pcl::console::print_error("Sigma values must be non-negative\n");
    return 1;
  }

  std::mt19937 gen(static_cast<std::uint32_t>(seed));

  // 1) Generate source points.
  std::vector<Eigen::Vector3d> r_true = generateUniformPoints(gen, num_points, scale);

  // 2) Generate a ground-truth transform (avoid Cayley singularities near 180 deg).
  const double max_angle_rad = max_angle_deg * M_PI / 180.0;
  Eigen::Matrix3d R_true = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_true = Eigen::Vector3d::Zero();
  for (int k = 0; k < 100; ++k) {
    R_true = randomRotationAngleAxis(gen, max_angle_rad);
    t_true = randomTranslation(gen, t_scale);
    const double det_IpR = (Eigen::Matrix3d::Identity() + R_true).determinant();
    if (std::abs(det_IpR) > 1e-6) {
      break;
    }
  }

  // 3) Generate noise-free target points.
  std::vector<Eigen::Vector3d> b_true = applyTransform(r_true, R_true, t_true);

  pcl::console::print_info("Ground-truth transform (target <- source):\n");
  Eigen::Matrix4d T_true = Eigen::Matrix4d::Identity();
  T_true.block<3, 3>(0, 0) = R_true;
  T_true.block<3, 1>(0, 3) = t_true;
  glnr::printMatrix4d(std::cout, T_true, "  ");

  // 4) Compute theoretical covariance at the nominal (noise-free) configuration.
  const glnr::GLnRNoiseModel nm = glnr::GLnRNoiseModel::isotropic(sigma_src, sigma_tgt);
  const glnr::GLnRResult nominal = glnr::estimateRigidTransformGLnR(r_true, b_true, nullptr, enforce_so3, &nm);
  if (!nominal.success || !nominal.has_covariance) {
    pcl::console::print_error("Failed to compute nominal GLnR solution / covariance: %s\n", nominal.message.c_str());
    return 1;
  }

  const Eigen::Vector3d g_true = cayleyGFromRotation(R_true);

  pcl::console::print_info("Nominal (noise-free) GLnR estimate (should match GT):\n");
  glnr::printMatrix4d(std::cout, nominal.T, "  ");
  pcl::console::print_info("Nominal RMSE: %.6g\n\n", nominal.rmse);

  // Store empirical deltas.
  std::vector<Eigen::Vector3d> dg_list;
  std::vector<Eigen::Vector3d> dt_list;
  std::vector<Eigen::Matrix3d> dR_list;
  std::vector<Eigen::Matrix<double, 6, 1>> dgt_list;
  dg_list.reserve(static_cast<std::size_t>(trials));
  dt_list.reserve(static_cast<std::size_t>(trials));
  dR_list.reserve(static_cast<std::size_t>(trials));
  dgt_list.reserve(static_cast<std::size_t>(trials));

  // 5) Monte-Carlo trials with new noise each run.
  for (int k = 0; k < trials; ++k) {
    std::vector<Eigen::Vector3d> r = r_true;
    std::vector<Eigen::Vector3d> b = b_true;

    addGaussianNoiseInPlace(gen, r, sigma_src);
    addGaussianNoiseInPlace(gen, b, sigma_tgt);

    const glnr::GLnRResult est = glnr::estimateRigidTransformGLnR(r, b, nullptr, enforce_so3, nullptr);
    if (!est.success) {
      pcl::console::print_warn("Trial %d failed: %s\n", k, est.message.c_str());
      continue;
    }

    const Eigen::Vector3d dg = est.g - g_true;
    const Eigen::Vector3d dt = est.t - t_true;
    const Eigen::Matrix3d dR = est.R - R_true;

    dg_list.push_back(dg);
    dt_list.push_back(dt);
    dR_list.push_back(dR);

    Eigen::Matrix<double, 6, 1> dgt;
    dgt.head<3>() = dg;
    dgt.tail<3>() = dt;
    dgt_list.push_back(dgt);
  }

  if (dg_list.size() < 2) {
    pcl::console::print_error("Too few successful trials to compute statistics.\n");
    return 1;
  }

  // 6) Empirical sample covariances.
  const Eigen::Matrix3d cov_g_s = sampleCov3(dg_list);
  const Eigen::Matrix3d cov_t_s = sampleCov3(dt_list);
  const Eigen::Matrix3d cov_R_s = sampleSecondMomentRRt(dR_list);
  const Eigen::Matrix<double, 6, 6> cov_gt_s = sampleCov6(dgt_list);

  pcl::console::print_info("\n=== Theoretical covariances (paper Section II-E) ===\n");
  pcl::console::print_info("Sigma_g (3x3):\n");
  glnr::printMatrix3d(std::cout, nominal.cov_g, "  ");
  pcl::console::print_info("Sigma_R = <dR dR^T> (3x3):\n");
  glnr::printMatrix3d(std::cout, nominal.cov_R, "  ");
  pcl::console::print_info("Sigma_t (3x3):\n");
  glnr::printMatrix3d(std::cout, nominal.cov_t, "  ");

  pcl::console::print_info("\n=== Empirical sample covariances (Monte-Carlo) ===\n");
  pcl::console::print_info("Sample cov_g (3x3):\n");
  glnr::printMatrix3d(std::cout, cov_g_s, "  ");
  pcl::console::print_info("Sample cov_R (3x3):\n");
  glnr::printMatrix3d(std::cout, cov_R_s, "  ");
  pcl::console::print_info("Sample cov_t (3x3):\n");
  glnr::printMatrix3d(std::cout, cov_t_s, "  ");

  auto frob = [](const Eigen::Matrix3d& A) { return std::sqrt((A.array() * A.array()).sum()); };
  const double err_g = frob(cov_g_s - nominal.cov_g) / std::max(1e-12, frob(nominal.cov_g));
  const double err_R = frob(cov_R_s - nominal.cov_R) / std::max(1e-12, frob(nominal.cov_R));
  const double err_t = frob(cov_t_s - nominal.cov_t) / std::max(1e-12, frob(nominal.cov_t));

  pcl::console::print_info("\nRelative Frobenius errors (sample vs theory):\n");
  pcl::console::print_info("  g: %.6g\n", err_g);
  pcl::console::print_info("  R: %.6g\n", err_R);
  pcl::console::print_info("  t: %.6g\n", err_t);

  // Write CSV outputs for plotting.
  Eigen::Matrix<double, 6, 6> cov_gt_th = Eigen::Matrix<double, 6, 6>::Zero();
  cov_gt_th.block<3, 3>(0, 0) = nominal.cov_g;
  cov_gt_th.block<3, 3>(3, 3) = nominal.cov_t;

  writeMatrixCSV(out_prefix + "_theory_cov_g.csv", nominal.cov_g);
  writeMatrixCSV(out_prefix + "_theory_cov_R.csv", nominal.cov_R);
  writeMatrixCSV(out_prefix + "_theory_cov_t.csv", nominal.cov_t);
  writeMatrixCSV(out_prefix + "_theory_cov_gt.csv", cov_gt_th);
  writeMatrixCSV(out_prefix + "_sample_cov_g.csv", cov_g_s);
  writeMatrixCSV(out_prefix + "_sample_cov_R.csv", cov_R_s);
  writeMatrixCSV(out_prefix + "_sample_cov_t.csv", cov_t_s);
  writeMatrixCSV(out_prefix + "_sample_cov_gt.csv", cov_gt_s);

  pcl::console::print_info("\nWrote CSV files with prefix '%s'\n", out_prefix.c_str());
  pcl::console::print_info("You can plot them with tools/plot_cov_uppertri.py (included in repo).\n");

  return 0;
}
