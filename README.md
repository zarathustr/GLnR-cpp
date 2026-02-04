# GLnR registration (PCL C++)

This is a small, self-contained reference implementation of the **GLnR (Generalized Linear n-Dimensional Rigid Registration)** method from:

- Jin Wu et al., *Generalized n-Dimensional Rigid Registration: Theory and Applications*, IEEE Transactions on Cybernetics, 2023.

The original paper is *n*-dimensional; this implementation targets the practical PCL use case: **3D rigid registration** (SO(3) rotation + 3D translation) for point clouds.

The program runs an ICP-like loop:
1. Find correspondences by nearest neighbor search (optional RANSAC rejection).
2. Estimate the incremental rigid transform via the **GLnR closed-form linear solve** using the Cayley parameterization.
3. Repeat until convergence.

## Build

Tested API usage is compatible with **PCL 1.8.0 ~ 1.12.0**.

```bash
mkdir -p build
cd build
cmake ..
make -j
```

## Run

```bash
./glnr_register --source source.pcd --target target.pcd --output aligned.pcd --transform transform.txt --visualize
```

### Common options

- `--corr_dist 0.05` : max correspondence distance (meters).
- `--ransac 0.05` : enable RANSAC correspondence rejection with inlier threshold (meters).
- `--voxel 0.02` : optional voxel downsampling (meters).

## Supported input formats

- `.pcd`
- `.ply`
- `.vtk` *(requires PCL built with VTK)*
- `.pcl` *(treated like `.pcd` — many users use this extension for PCD files)*

Outputs:
- `.pcd` or `.ply` aligned point cloud
- `transform.txt` 4x4 homogeneous transform matrix (target <- source)
