#!/usr/bin/env python3
"""Upper-triangular covariance plot helper.

This is a lightweight utility to visualize the 6x6 covariance matrix produced by
`glnr_monte_carlo_cov`:

  <prefix>_theory_cov_gt.csv  (block-diagonal theory for [g; t])
  <prefix>_sample_cov_gt.csv  (Monte-Carlo sample covariance for [delta_g; delta_t])

Example:
  python3 tools/plot_cov_uppertri.py --prefix glnr_mc

The plot is saved as:
  <prefix>_cov_gt_uppertri.png
"""

import argparse
import csv
import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt


def read_csv_matrix(path: str) -> np.ndarray:
    rows: List[List[float]] = []
    with open(path, "r", newline="") as f:
        for r in csv.reader(f):
            if not r:
                continue
            rows.append([float(x) for x in r])
    return np.array(rows, dtype=float)


def uppertri_mask(n: int) -> np.ndarray:
    # True for entries we want to hide.
    return np.tril(np.ones((n, n), dtype=bool), k=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="glnr_mc", help="CSV prefix (default: glnr_mc)")
    ap.add_argument(
        "--theory_csv",
        default=None,
        help="Override theory CSV path (default: <prefix>_theory_cov_gt.csv)",
    )
    ap.add_argument(
        "--sample_csv",
        default=None,
        help="Override sample CSV path (default: <prefix>_sample_cov_gt.csv)",
    )
    ap.add_argument("--out", default=None, help="Output PNG path (default: <prefix>_cov_gt_uppertri.png)")
    args = ap.parse_args()

    theory_csv = args.theory_csv or f"{args.prefix}_theory_cov_gt.csv"
    sample_csv = args.sample_csv or f"{args.prefix}_sample_cov_gt.csv"
    out_png = args.out or f"{args.prefix}_cov_gt_uppertri.png"

    if not os.path.exists(sample_csv):
        raise FileNotFoundError(sample_csv)

    S = read_csv_matrix(sample_csv)
    if S.shape[0] != S.shape[1]:
        raise ValueError(f"Sample covariance must be square, got {S.shape}")

    T = None
    if os.path.exists(theory_csv):
        T = read_csv_matrix(theory_csv)

    n = S.shape[0]
    mask = uppertri_mask(n)

    fig = plt.figure(figsize=(10, 4 if T is None else 8))

    def plot_one(ax, M: np.ndarray, title: str):
        Mm = M.copy()
        Mm[mask] = np.nan
        im = ax.imshow(Mm, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([f"g{i+1}" for i in range(3)] + [f"t{i+1}" for i in range(3)])
        ax.set_yticklabels([f"g{i+1}" for i in range(3)] + [f"t{i+1}" for i in range(3)])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if T is None:
        ax = fig.add_subplot(1, 1, 1)
        plot_one(ax, S, "Sample cov([delta_g; delta_t]) upper-triangular")
    else:
        ax1 = fig.add_subplot(2, 1, 1)
        plot_one(ax1, T, "Theory cov([g; t]) (block-diagonal) upper-triangular")
        ax2 = fig.add_subplot(2, 1, 2)
        plot_one(ax2, S, "Sample cov([delta_g; delta_t]) upper-triangular")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
