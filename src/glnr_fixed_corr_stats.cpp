#include <glnr/glnr_solver.h>

#include <Eigen/Dense>

#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/conversions.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// VTK loader support (for .vtk meshes/point sets). Same strategy as glnr_register.cpp.
#if defined(__has_include)
#  if __has_include(<pcl/io/vtk_lib_io.h>)
#    include <pcl/io/vtk_lib_io.h>
#    define GLNR_HAS_VTK_IO 1
#  elif __has_include(<pcl/io/vtk_io.h>)
#    include <pcl/io/vtk_io.h>
#    define GLNR_HAS_VTK_IO 1
#  else
#    define GLNR_HAS_VTK_IO 0
#  endif
#else
#  define GLNR_HAS_VTK_IO 0
#endif

namespace {

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;
using CloudPtr = CloudT::Ptr;

static std::string toLower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

static std::string fileExtensionLower(const std::string& filename)
{
  const auto pos = filename.find_last_of('.');
  if (pos == std::string::npos) return "";
  return toLower(filename.substr(pos + 1));
}

static bool loadPointCloudAny(const std::string& filename, CloudPtr cloud)
{
  if (!cloud) return false;

  const std::string ext = fileExtensionLower(filename);

  // Many users informally use .pcl for PCD files. We'll treat it as PCD first.
  if (ext == "pcd" || ext == "pcl") {
    if (pcl::io::loadPCDFile<PointT>(filename, *cloud) == 0) {
      return true;
    }
    pcl::console::print_error("Failed to read as PCD: %s\n", filename.c_str());
    return false;
  }

  if (ext == "ply") {
    if (pcl::io::loadPLYFile<PointT>(filename, *cloud) == 0) {
      return true;
    }
    pcl::console::print_error("Failed to read as PLY: %s\n", filename.c_str());
    return false;
  }

  if (ext == "vtk") {
#if GLNR_HAS_VTK_IO
    pcl::PolygonMesh mesh;
    const int ret = pcl::io::loadPolygonFileVTK(filename, mesh);
    if (ret <= 0) {
      pcl::console::print_error("Failed to read as VTK PolygonMesh: %s\n", filename.c_str());
      return false;
    }
    pcl::fromPCLPointCloud2(mesh.cloud, *cloud);
    return true;
#else
    pcl::console::print_error(
      "This build does not include PCL VTK IO headers. Rebuild PCL with VTK support to load .vtk files.\n");
    return false;
#endif
  }

  // Unknown extension: try PCD then PLY, then VTK (if available).
  if (pcl::io::loadPCDFile<PointT>(filename, *cloud) == 0) {
    return true;
  }
  if (pcl::io::loadPLYFile<PointT>(filename, *cloud) == 0) {
    return true;
  }

#if GLNR_HAS_VTK_IO
  {
    pcl::PolygonMesh mesh;
    const int ret = pcl::io::loadPolygonFileVTK(filename, mesh);
    if (ret > 0) {
      pcl::fromPCLPointCloud2(mesh.cloud, *cloud);
      return true;
    }
  }
#endif

  pcl::console::print_error("Unsupported or unreadable input file: %s\n", filename.c_str());
  return false;
}

static bool loadMatrix4dFromFile(const std::string& path, Eigen::Matrix4d& T)
{
  std::ifstream ifs(path.c_str());
  if (!ifs.is_open()) return false;

  std::vector<double> vals;
  vals.reserve(16);
  double v;
  while (ifs >> v) {
    vals.push_back(v);
  }
  if (vals.size() != 16) return false;

  T = Eigen::Matrix4d::Identity();
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      T(r, c) = vals[static_cast<std::size_t>(r * 4 + c)];
    }
  }
  return true;
}

static Eigen::Matrix3d randomRotation(std::mt19937& gen)
{
  // Uniform random unit quaternion -> SO(3)
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  const double u1 = unif(gen);
  const double u2 = unif(gen);
  const double u3 = unif(gen);

  const double kTwoPi = 6.28318530717958647692;

  const double q1 = std::sqrt(1.0 - u1) * std::sin(kTwoPi * u2);
  const double q2 = std::sqrt(1.0 - u1) * std::cos(kTwoPi * u2);
  const double q3 = std::sqrt(u1)       * std::sin(kTwoPi * u3);
  const double q4 = std::sqrt(u1)       * std::cos(kTwoPi * u3);

  Eigen::Quaterniond q(q4, q1, q2, q3);
  q.normalize();
  return q.toRotationMatrix();
}

static std::vector<Eigen::Vector3d> cloudToEigen(const CloudT& cloud)
{
  std::vector<Eigen::Vector3d> pts;
  pts.reserve(cloud.size());
  for (const auto& p : cloud) {
    pts.emplace_back(static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z));
  }
  return pts;
}

static CloudT eigenToCloud(const std::vector<Eigen::Vector3d>& pts)
{
  CloudT cloud;
  cloud.reserve(pts.size());
  for (const auto& v : pts) {
    PointT p;
    p.x = static_cast<float>(v(0));
    p.y = static_cast<float>(v(1));
    p.z = static_cast<float>(v(2));
    cloud.push_back(p);
  }
  return cloud;
}

}  // namespace

static void printUsage(const char* prog)
{
  pcl::console::print_info(
    "Usage:\n"
    "  %s --source src.pcd [--target tgt.pcd] [--gt_file gt.txt] [options]\n\n"
    "Fixed-correspondence (index-aligned) GLnR registration statistics.\n"
    "If --target is omitted, a synthetic target is generated from --source using --gt_file\n"
    "(or a random GT if --random_gt is set).\n\n"
    "Options:\n"
    "  --trials N                 Monte-Carlo trials (default 200)\n"
    "  --seed S                   RNG seed (default 0)\n"
    "  --sigma T                  std-dev for target noise (meters, default 0.01)\n"
    "  --sigma_src S              std-dev for source noise (meters, default 0.0)\n"
    "  --noise_both               shortcut: sigma_src = sigma_tgt (overrides --sigma_src)\n"
    "  --random_gt                generate a random GT (ignored if --gt_file provided)\n"
    "  --gt_tx X --gt_ty Y --gt_tz Z  translation for random/identity GT (default 0,0,0)\n"
    "  --enforce_so3              project R to SO(3) before error calc (default: off)\n"
    "\n"
    "GT file format (gt.txt): 16 whitespace-separated numbers (row-major 4x4).\n",
    prog);
}

int main(int argc, char** argv)
{
  if (pcl::console::find_switch(argc, argv, "--help") ||
      pcl::console::find_switch(argc, argv, "-h")) {
    printUsage(argv[0]);
    return 0;
  }

  std::string src_file;
  std::string tgt_file;
  std::string gt_file;

  pcl::console::parse_argument(argc, argv, "--source", src_file);
  pcl::console::parse_argument(argc, argv, "--target", tgt_file);
  pcl::console::parse_argument(argc, argv, "--gt_file", gt_file);

  if (src_file.empty()) {
    pcl::console::print_error("Missing --source\n");
    printUsage(argv[0]);
    return 1;
  }

  int trials = 200;
  int seed = 0;
  double sigma_tgt = 0.01;
  double sigma_src = 0.0;

  pcl::console::parse_argument(argc, argv, "--trials", trials);
  pcl::console::parse_argument(argc, argv, "--seed", seed);
  pcl::console::parse_argument(argc, argv, "--sigma", sigma_tgt);
  pcl::console::parse_argument(argc, argv, "--sigma_tgt", sigma_tgt);
  pcl::console::parse_argument(argc, argv, "--sigma_src", sigma_src);

  const bool noise_both = pcl::console::find_switch(argc, argv, "--noise_both");
  if (noise_both) sigma_src = sigma_tgt;

  const bool random_gt = pcl::console::find_switch(argc, argv, "--random_gt");
  const bool enforce_so3 = pcl::console::find_switch(argc, argv, "--enforce_so3");

  double gt_tx = 0.0, gt_ty = 0.0, gt_tz = 0.0;
  pcl::console::parse_argument(argc, argv, "--gt_tx", gt_tx);
  pcl::console::parse_argument(argc, argv, "--gt_ty", gt_ty);
  pcl::console::parse_argument(argc, argv, "--gt_tz", gt_tz);

  // Load source
  CloudPtr src_cloud(new CloudT);
  if (!loadPointCloudAny(src_file, src_cloud)) {
    return 1;
  }

  // Load or synthesize target
  CloudPtr tgt_cloud(new CloudT);
  bool has_target = !tgt_file.empty();
  if (has_target) {
    if (!loadPointCloudAny(tgt_file, tgt_cloud)) {
      return 1;
    }
  }

  if (src_cloud->empty()) {
    pcl::console::print_error("Source cloud is empty.\n");
    return 1;
  }

  Eigen::Matrix4d T_gt = Eigen::Matrix4d::Identity();
  bool has_gt = false;

  // RNG (used for Monte-Carlo noise and optional random GT).
  std::mt19937 gen(static_cast<std::mt19937::result_type>(seed));

  if (!gt_file.empty()) {
    if (!loadMatrix4dFromFile(gt_file, T_gt)) {
      pcl::console::print_error("Failed to load --gt_file. Expect 16 doubles.\n");
      return 1;
    }
    has_gt = true;
  } else if (!has_target) {
    // If we're synthesizing and no GT file is provided:
    // either random GT or identity (with optional translation).
    T_gt.setIdentity();
    T_gt.block<3,3>(0,0) = random_gt ? randomRotation(gen) : Eigen::Matrix3d::Identity();
    T_gt(0,3) = gt_tx;
    T_gt(1,3) = gt_ty;
    T_gt(2,3) = gt_tz;
    has_gt = true;
  }

  if (has_target && !has_gt) {
    pcl::console::print_error("When providing --target, you must also provide --gt_file for ground-truth comparison.\n");
    return 1;
  }


  const Eigen::Matrix3d R_gt = T_gt.block<3,3>(0,0);
  const Eigen::Vector3d t_gt = T_gt.block<3,1>(0,3);

  std::vector<Eigen::Vector3d> src_clean = cloudToEigen(*src_cloud);
  std::vector<Eigen::Vector3d> tgt_clean;

  if (has_target) {
    tgt_clean = cloudToEigen(*tgt_cloud);
  } else {
    // Synthetic target: b_i = R_gt r_i + t_gt (paper Eq. (42), without noise here)
    tgt_clean.resize(src_clean.size());
    for (std::size_t i = 0; i < src_clean.size(); ++i) {
      tgt_clean[i] = R_gt * src_clean[i] + t_gt;
    }
  }

  if (tgt_clean.size() != src_clean.size()) {
    pcl::console::print_error("Source and target must have same number of points for fixed correspondences. "
                              "Got %zu vs %zu.\n",
                              src_clean.size(), tgt_clean.size());
    return 1;
  }

  const std::size_t N = src_clean.size();
  pcl::console::print_info("N=%zu, trials=%d, sigma_src=%.6f, sigma_tgt=%.6f\n", N, trials, sigma_src, sigma_tgt);

  pcl::console::print_info("Ground-truth T (row-major):\n");
  glnr::printMatrix4d(std::cout, T_gt, "  ");

  const glnr::GLnRNoiseModel noise_model = glnr::GLnRNoiseModel::isotropic(sigma_src, sigma_tgt);
  std::normal_distribution<double> ndist(0.0, 1.0);

  Eigen::Matrix3d Sigma_R_sample = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d Sigma_t_sample = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d Sigma_R_pred = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d Sigma_t_pred = Eigen::Matrix3d::Zero();

  double mean_rmse = 0.0;

  for (int k = 0; k < trials; ++k) {
    std::vector<Eigen::Vector3d> src_noisy = src_clean;
    std::vector<Eigen::Vector3d> tgt_noisy = tgt_clean;

    if (sigma_src > 0.0) {
      for (auto& p : src_noisy) {
        p(0) += sigma_src * ndist(gen);
        p(1) += sigma_src * ndist(gen);
        p(2) += sigma_src * ndist(gen);
      }
    }
    if (sigma_tgt > 0.0) {
      for (auto& p : tgt_noisy) {
        p(0) += sigma_tgt * ndist(gen);
        p(1) += sigma_tgt * ndist(gen);
        p(2) += sigma_tgt * ndist(gen);
      }
    }

    // Fixed-correspondence GLnR (single shot), with covariance
    const glnr::GLnRResult res = glnr::estimateRigidTransformGLnR(
      src_noisy, tgt_noisy, nullptr, enforce_so3, &noise_model);

    if (!res.success || !res.has_covariance) {
      pcl::console::print_error("GLnR failed at trial %d: %s\n", k, res.message.c_str());
      return 1;
    }

    const Eigen::Matrix3d dR = res.R - R_gt;
    const Eigen::Vector3d dt = res.t - t_gt;

    Sigma_R_sample.noalias() += dR * dR.transpose();
    Sigma_t_sample.noalias() += dt * dt.transpose();

    Sigma_R_pred.noalias() += res.cov_R;
    Sigma_t_pred.noalias() += res.cov_t;

    mean_rmse += res.rmse;
  }

  Sigma_R_sample /= static_cast<double>(trials);
  Sigma_t_sample /= static_cast<double>(trials);
  Sigma_R_pred /= static_cast<double>(trials);
  Sigma_t_pred /= static_cast<double>(trials);
  mean_rmse /= static_cast<double>(trials);

  pcl::console::print_info("\nMean RMSE over trials: %.8f\n", mean_rmse);

  pcl::console::print_info("\nSample covariance Σ_R = E[(R_est-R_gt)(R_est-R_gt)^T]:\n");
  glnr::printMatrix3d(std::cout, Sigma_R_sample, "  ");

  pcl::console::print_info("\nPredicted covariance Σ_R (paper Eq. (40), averaged over trials):\n");
  glnr::printMatrix3d(std::cout, Sigma_R_pred, "  ");

  pcl::console::print_info("\nSample covariance Σ_t = E[(t_est-t_gt)(t_est-t_gt)^T]:\n");
  glnr::printMatrix3d(std::cout, Sigma_t_sample, "  ");

  pcl::console::print_info("\nPredicted covariance Σ_t (paper Eq. (41), averaged over trials):\n");
  glnr::printMatrix3d(std::cout, Sigma_t_pred, "  ");

  const double diff_R = (Sigma_R_sample - Sigma_R_pred).norm();
  const double diff_t = (Sigma_t_sample - Sigma_t_pred).norm();
  pcl::console::print_info("\nFrobenius ||Σ_R_sample - Σ_R_pred|| = %.8e\n", diff_R);
  pcl::console::print_info("Frobenius ||Σ_t_sample - Σ_t_pred|| = %.8e\n", diff_t);

  return 0;
}
