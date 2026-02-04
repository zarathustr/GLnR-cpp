#include <glnr/glnr_solver.h>

#include <Eigen/Dense>

#include <pcl/point_types.h>
#include <pcl/common/point_tests.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/conversions.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// VTK loader support (for .vtk meshes/point sets).
// We use __has_include so the same code can build across PCL 1.8 ~ 1.12
// with different header names.
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
using CloudConstPtr = CloudT::ConstPtr;

struct Options
{
  int max_iterations = 50;
  double max_correspondence_distance = 0.05;  // meters
  double transformation_epsilon = 1e-6;      // meters and radians

  double ransac_inlier_threshold = 0.0;  // 0 disables RANSAC rejection
  int ransac_max_iterations = 2000;

  double voxel_leaf_size = 0.0;  // 0 disables downsampling

  bool visualize = false;
  bool verbose = true;
};

static std::string toLower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

static std::string fileExtensionLower(const std::string& filename)
{
  const auto pos = filename.find_last_of('.');
  if (pos == std::string::npos) return std::string();
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
    pcl::console::print_error("This build does not include PCL VTK IO headers. Rebuild PCL with VTK support to load .vtk files.\n");
    return false;
#endif
  }

  // Unknown extension: try PCD then PLY.
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

static CloudPtr downsampleIfNeeded(const CloudConstPtr& in, double leaf)
{
  if (!in) return CloudPtr(new CloudT);
  if (!(leaf > 0.0)) {
    return CloudPtr(new CloudT(*in));
  }

  pcl::VoxelGrid<PointT> vg;
  vg.setInputCloud(in);
  vg.setLeafSize(static_cast<float>(leaf), static_cast<float>(leaf), static_cast<float>(leaf));

  CloudPtr out(new CloudT);
  vg.filter(*out);
  return out;
}

static bool savePointCloudAny(const std::string& filename, const CloudConstPtr& cloud)
{
  if (!cloud) return false;
  const std::string ext = fileExtensionLower(filename);

  if (ext == "pcd" || ext == "pcl" || ext.empty()) {
    // If user gives no extension, we still write PCD.
    if (pcl::io::savePCDFileBinary(filename, *cloud) == 0) {
      return true;
    }
    pcl::console::print_error("Failed to write PCD: %s\n", filename.c_str());
    return false;
  }

  if (ext == "ply") {
    if (pcl::io::savePLYFileBinary(filename, *cloud) == 0) {
      return true;
    }
    pcl::console::print_error("Failed to write PLY: %s\n", filename.c_str());
    return false;
  }

  pcl::console::print_error("Unsupported output extension (use .pcd or .ply): %s\n", filename.c_str());
  return false;
}

static bool writeMatrixToTextFile(const std::string& filename, const Eigen::Matrix4d& T)
{
  std::ofstream ofs(filename.c_str());
  if (!ofs.is_open()) {
    return false;
  }
  ofs.setf(std::ios::fixed);
  ofs.precision(12);
  glnr::printMatrix4d(ofs, T);
  ofs.close();
  return true;
}

static double rotationAngleRadians(const Eigen::Matrix3d& R)
{
  Eigen::AngleAxisd aa(R);
  return std::abs(aa.angle());
}

struct ICPResult
{
  bool success = false;
  std::string message;

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  int iterations = 0;
  std::size_t num_correspondences = 0;
  double last_rmse = std::numeric_limits<double>::quiet_NaN();
};

static ICPResult glnrIcp(const CloudConstPtr& source_in,
                         const CloudConstPtr& target_in,
                         const Options& opt)
{
  ICPResult result;

  if (!source_in || !target_in) {
    result.success = false;
    result.message = "Null input cloud.";
    return result;
  }
  if (source_in->empty() || target_in->empty()) {
    result.success = false;
    result.message = "Empty input cloud.";
    return result;
  }

  // KD-tree for target (fixed).
  pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);
  kdtree->setInputCloud(target_in);

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();

  CloudPtr src_trans(new CloudT);
  src_trans->reserve(source_in->size());

  const double max_corr2 = opt.max_correspondence_distance * opt.max_correspondence_distance;

  for (int iter = 0; iter < opt.max_iterations; ++iter) {
    // Transform source by current estimate.
    pcl::transformPointCloud(*source_in, *src_trans, T.cast<float>());

    // Build correspondences by NN search.
    pcl::Correspondences corrs;
    corrs.reserve(src_trans->size());

    std::vector<int> nn_idx(1);
    std::vector<float> nn_dist2(1);

    for (std::size_t i = 0; i < src_trans->size(); ++i) {
      const PointT& p = src_trans->points[i];
      if (!pcl::isFinite(p)) {
        continue;
      }

      if (kdtree->nearestKSearch(p, 1, nn_idx, nn_dist2) > 0) {
        if (static_cast<double>(nn_dist2[0]) <= max_corr2) {
          corrs.emplace_back(static_cast<int>(i), nn_idx[0], nn_dist2[0]);
        }
      }
    }

    if (corrs.size() < 3) {
      result.success = false;
      result.message = "Not enough correspondences (<3). Try increasing --corr_dist, using a better initial guess, or downsampling.";
      return result;
    }

    // Optional: RANSAC rejector for correspondences.
    if (opt.ransac_inlier_threshold > 0.0) {
      pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> rejector;
      rejector.setInputSource(src_trans);
      rejector.setInputTarget(target_in);
      rejector.setInlierThreshold(opt.ransac_inlier_threshold);
      rejector.setMaximumIterations(opt.ransac_max_iterations);

      pcl::Correspondences corrs_inliers;
      rejector.getRemainingCorrespondences(corrs, corrs_inliers);
      if (corrs_inliers.size() >= 3) {
        corrs.swap(corrs_inliers);
      }
      // else keep original correspondences (better than failing immediately).
    }

    // Extract paired points for GLnR.
    std::vector<Eigen::Vector3d> src_pts;
    std::vector<Eigen::Vector3d> tgt_pts;
    src_pts.reserve(corrs.size());
    tgt_pts.reserve(corrs.size());

    for (const auto& c : corrs) {
      const PointT& ps = src_trans->points[static_cast<std::size_t>(c.index_query)];
      const PointT& pt = target_in->points[static_cast<std::size_t>(c.index_match)];
      src_pts.emplace_back(static_cast<double>(ps.x), static_cast<double>(ps.y), static_cast<double>(ps.z));
      tgt_pts.emplace_back(static_cast<double>(pt.x), static_cast<double>(pt.y), static_cast<double>(pt.z));
    }

    const glnr::GLnRResult delta = glnr::estimateRigidTransformGLnR(src_pts, tgt_pts, nullptr, true);
    if (!delta.success) {
      result.success = false;
      result.message = std::string("GLnR solve failed: ") + delta.message;
      return result;
    }

    const Eigen::Matrix4d dT = delta.T;

    // Update global transform.
    T = dT * T;

    // Convergence check on incremental transform.
    const double d_trans = dT.block<3, 1>(0, 3).norm();
    const double d_rot = rotationAngleRadians(dT.block<3, 3>(0, 0));

    if (opt.verbose) {
      pcl::console::print_info(
          "[Iter %d] corrs=%zu  delta_rmse=%.6g  |dT|: trans=%.3g m rot=%.3g rad\n",
          iter, corrs.size(), delta.rmse, d_trans, d_rot);
    }

    result.iterations = iter + 1;
    result.num_correspondences = corrs.size();
    result.last_rmse = delta.rmse;

    if (d_trans < opt.transformation_epsilon && d_rot < opt.transformation_epsilon) {
      result.success = true;
      result.message = "Converged.";
      result.T = T;
      return result;
    }
  }

  result.success = true;
  result.message = "Reached max iterations.";
  result.T = T;
  return result;
}

static void visualizeClouds(const CloudConstPtr& target,
                            const CloudConstPtr& source,
                            const CloudConstPtr& aligned)
{
  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("GLnR Registration"));
  viewer->setBackgroundColor(0.05, 0.05, 0.05);

  pcl::visualization::PointCloudColorHandlerCustom<PointT> tgt_color(target, 0, 255, 0);
  viewer->addPointCloud<PointT>(target, tgt_color, "target");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target");

  if (source && !source->empty()) {
    pcl::visualization::PointCloudColorHandlerCustom<PointT> src_color(source, 255, 0, 0);
    viewer->addPointCloud<PointT>(source, src_color, "source");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "source");
  }

  if (aligned && !aligned->empty()) {
    pcl::visualization::PointCloudColorHandlerCustom<PointT> aligned_color(aligned, 0, 128, 255);
    viewer->addPointCloud<PointT>(aligned, aligned_color, "aligned");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "aligned");
  }

  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();

  while (!viewer->wasStopped()) {
    viewer->spinOnce(50);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

static void printUsage(const char* prog)
{
  pcl::console::print_info(
      "Usage:\n"
      "  %s --source <src.{pcd|ply|vtk|pcl}> --target <tgt.{pcd|ply|vtk|pcl}> [options]\n\n"
      "Options:\n"
      "  --output <file.pcd|file.ply>     Output aligned cloud (default: aligned.pcd)\n"
      "  --transform <file.txt>           Save 4x4 transform matrix (default: transform.txt)\n"
      "  --max_iter <int>                 Max ICP iterations (default: 50)\n"
      "  --corr_dist <float>              Max correspondence distance (m) (default: 0.05)\n"
      "  --trans_eps <float>              Convergence threshold (m & rad) (default: 1e-6)\n"
      "  --ransac <float>                 RANSAC inlier threshold (m); 0 disables (default: 0)\n"
      "  --ransac_iter <int>              RANSAC max iterations (default: 2000)\n"
      "  --voxel <float>                  Voxel leaf size (m); 0 disables (default: 0)\n"
      "  --visualize                      Show target/source/aligned in a viewer\n"
      "  --quiet                          Less console output\n\n",
      prog);
}

}  // namespace

int main(int argc, char** argv)
{
  std::string source_file;
  std::string target_file;
  std::string output_file = "aligned.pcd";
  std::string transform_file = "transform.txt";

  if (pcl::console::find_switch(argc, argv, "--help") || pcl::console::find_switch(argc, argv, "-h")) {
    printUsage(argv[0]);
    return 0;
  }

  pcl::console::parse_argument(argc, argv, "--source", source_file);
  pcl::console::parse_argument(argc, argv, "-s", source_file);
  pcl::console::parse_argument(argc, argv, "--target", target_file);
  pcl::console::parse_argument(argc, argv, "-t", target_file);

  if (source_file.empty() || target_file.empty()) {
    pcl::console::print_error("--source and --target are required.\n\n");
    printUsage(argv[0]);
    return 1;
  }

  pcl::console::parse_argument(argc, argv, "--output", output_file);
  pcl::console::parse_argument(argc, argv, "-o", output_file);
  pcl::console::parse_argument(argc, argv, "--transform", transform_file);

  Options opt;
  pcl::console::parse_argument(argc, argv, "--max_iter", opt.max_iterations);
  pcl::console::parse_argument(argc, argv, "--corr_dist", opt.max_correspondence_distance);
  pcl::console::parse_argument(argc, argv, "--trans_eps", opt.transformation_epsilon);
  pcl::console::parse_argument(argc, argv, "--ransac", opt.ransac_inlier_threshold);
  pcl::console::parse_argument(argc, argv, "--ransac_iter", opt.ransac_max_iterations);
  pcl::console::parse_argument(argc, argv, "--voxel", opt.voxel_leaf_size);

  opt.visualize = pcl::console::find_switch(argc, argv, "--visualize");
  opt.verbose = !pcl::console::find_switch(argc, argv, "--quiet");

  // Load clouds.
  CloudPtr source_raw(new CloudT);
  CloudPtr target_raw(new CloudT);

  if (!loadPointCloudAny(source_file, source_raw)) {
    pcl::console::print_error("Failed to load source cloud: %s\n", source_file.c_str());
    return 1;
  }
  if (!loadPointCloudAny(target_file, target_raw)) {
    pcl::console::print_error("Failed to load target cloud: %s\n", target_file.c_str());
    return 1;
  }

  pcl::console::print_info("Loaded source: %zu points\n", source_raw->size());
  pcl::console::print_info("Loaded target: %zu points\n", target_raw->size());

  // Optional downsampling.
  CloudPtr source = downsampleIfNeeded(source_raw, opt.voxel_leaf_size);
  CloudPtr target = downsampleIfNeeded(target_raw, opt.voxel_leaf_size);
  if (opt.voxel_leaf_size > 0.0) {
    pcl::console::print_info("Downsampled source: %zu points\n", source->size());
    pcl::console::print_info("Downsampled target: %zu points\n", target->size());
  }

  // Run GLnR-based ICP.
  const ICPResult icp = glnrIcp(source, target, opt);
  if (!icp.success) {
    pcl::console::print_error("Registration failed: %s\n", icp.message.c_str());
    return 2;
  }

  pcl::console::print_info("Registration finished: %s\n", icp.message.c_str());
  pcl::console::print_info("Iterations: %d\n", icp.iterations);
  pcl::console::print_info("Final correspondences: %zu\n", icp.num_correspondences);

  pcl::console::print_info("Final transform (target <- source):\n");
  std::cout.setf(std::ios::fixed);
  std::cout.precision(6);
  glnr::printMatrix4d(std::cout, icp.T, "  ");

  if (!writeMatrixToTextFile(transform_file, icp.T)) {
    pcl::console::print_warn("Could not write transform to %s\n", transform_file.c_str());
  } else {
    pcl::console::print_info("Saved transform to %s\n", transform_file.c_str());
  }

  // Apply transform to ORIGINAL (non-downsampled) source for output.
  CloudPtr aligned_raw(new CloudT);
  pcl::transformPointCloud(*source_raw, *aligned_raw, icp.T.cast<float>());

  if (!savePointCloudAny(output_file, aligned_raw)) {
    pcl::console::print_error("Failed to save aligned cloud to %s\n", output_file.c_str());
    return 3;
  }
  pcl::console::print_info("Saved aligned cloud: %s (%zu points)\n", output_file.c_str(), aligned_raw->size());

  if (opt.visualize) {
    visualizeClouds(target_raw, source_raw, aligned_raw);
  }

  return 0;
}
