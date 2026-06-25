#!/usr/bin/env python3
"""LESYMAP benchmark runner.

This script is intentionally separate from pytest. It records wall time and
lightweight checksums for reproducible performance snapshots without making
the unit test suite slower.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy
import sklearn
import numba

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lesymap.core.image_utils import images_to_matrix, mask_from_average
from lesymap.core.io import check_binary_values, load_lesions
from lesymap.core.patch import get_unique_lesion_patches
from lesymap.methods.correction import correct_pvalues
from lesymap.stats_compiled.bm import brunner_munzel_fast
from lesymap.stats_compiled.regression import regression_fast
from lesymap.stats_compiled.ttest import ttest_fast, welch_fast


def _time_call(func: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start
    return result, elapsed


def _checksum(value: Any) -> float:
    if isinstance(value, tuple):
        return float(sum(_checksum(item) for item in value))
    if isinstance(value, dict):
        return float(sum(_checksum(value[key]) for key in sorted(value)))
    arr = np.asarray(value)
    if arr.dtype == np.bool_:
        arr = arr.astype(np.int64)
    if not np.issubdtype(arr.dtype, np.number):
        return float(arr.size)
    return float(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).sum())


def _summarize_array(value: Any) -> dict[str, Any]:
    arr = np.asarray(value)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "checksum": _checksum(arr),
    }


def _measure_repeated(repeat: int, func: Callable[[], Any]) -> tuple[Any, list[float]]:
    timings = []
    result = None
    for _ in range(repeat):
        result, elapsed = _time_call(func)
        timings.append(elapsed)
    return result, timings


def _metric(name: str, result: Any, timings: list[float]) -> dict[str, Any]:
    return {
        "name": name,
        "wall_time_seconds": timings,
        "min_wall_time_seconds": min(timings),
        "mean_wall_time_seconds": float(np.mean(timings)),
        "checksum": _checksum(result),
    }


def _make_synthetic_lesmat(seed: int = 20260626) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_subjects = 48
    n_voxels = 4096
    subject_risk = rng.beta(1.7, 6.0, size=(n_subjects, 1))
    voxel_risk = rng.beta(1.2, 8.0, size=(1, n_voxels))
    probability = np.clip(subject_risk + voxel_risk, 0.02, 0.85)
    lesmat = (rng.random((n_subjects, n_voxels)) < probability).astype(np.float64)
    behavior = rng.normal(size=n_subjects).astype(np.float64)
    behavior += 0.35 * lesmat[:, :128].mean(axis=1)
    return lesmat, behavior


def run_kernels(repeat: int) -> dict[str, Any]:
    lesmat, behavior = _make_synthetic_lesmat()
    pvals = np.linspace(0.0001, 0.9999, lesmat.shape[1], dtype=np.float64)

    # Warm up Numba compilation outside measured runs.
    warm_x = lesmat[:12, :16].copy()
    warm_y = behavior[:12].copy()
    brunner_munzel_fast(warm_x, warm_y, True)
    ttest_fast(warm_x, warm_y, True)
    welch_fast(warm_x, warm_y, True)
    regression_fast(warm_x, warm_y, None)

    benchmarks: list[dict[str, Any]] = []
    cases: list[tuple[str, Callable[[], Any]]] = [
        ("bm", lambda: brunner_munzel_fast(lesmat, behavior, True)),
        ("ttest", lambda: ttest_fast(lesmat, behavior, True)),
        ("welch", lambda: welch_fast(lesmat, behavior, True)),
        ("regression", lambda: regression_fast(lesmat, behavior, None)),
        ("patch", lambda: get_unique_lesion_patches(lesmat, return_patch_matrix=True)),
        ("fdr", lambda: correct_pvalues(pvals, method="fdr")),
    ]

    for name, func in cases:
        result, timings = _measure_repeated(repeat, func)
        benchmarks.append(_metric(name, result, timings))

    return {
        "suite": "kernels",
        "input": {
            "lesmat_shape": list(lesmat.shape),
            "lesmat_dtype": str(lesmat.dtype),
            "behavior_shape": list(behavior.shape),
        },
        "benchmarks": benchmarks,
    }


def run_testdata_io(repeat: int, test_data_dir: Path, data_kind: str) -> dict[str, Any]:
    data_subdir = {
        "masks": "normalized_masks",
        "images": "normalized_images",
    }[data_kind]
    image_paths = sorted((test_data_dir / data_subdir).glob("*.nii.gz"))
    if not image_paths:
        raise FileNotFoundError(f"No NIfTI files found under {test_data_dir / data_subdir}")

    image_path_strings = [str(path) for path in image_paths]
    images, load_timings = _measure_repeated(
        repeat,
        lambda: load_lesions(image_path_strings, check_headers=True),
    )
    binary_info, binary_timings = _measure_repeated(
        repeat,
        lambda: check_binary_values(images, verbose=False),
    )
    mask, mask_timings = _measure_repeated(
        repeat,
        lambda: mask_from_average(images, threshold=0.0, min_voxels=0),
    )
    matrix_dtype = np.uint8 if data_kind == "masks" and binary_info["is_binary"] else np.float64
    matrix, matrix_timings = _measure_repeated(
        repeat,
        lambda: images_to_matrix(images, mask, dtype=matrix_dtype),
    )

    return {
        "suite": "testdata-io",
        "test_data_dir": str(test_data_dir),
        "data_kind": data_kind,
        "data_subdir": data_subdir,
        "n_images": len(image_paths),
        "image_shape": list(images[0].shape),
        "segments": [
            {
                "name": "load_lesions",
                "wall_time_seconds": load_timings,
                "min_wall_time_seconds": min(load_timings),
                "mean_wall_time_seconds": float(np.mean(load_timings)),
                "n_images": len(images),
                "image_shape": list(images[0].shape),
                "image_dtype": str(images[0].get_data_dtype()),
            },
            {
                "name": "check_binary_values",
                "wall_time_seconds": binary_timings,
                "min_wall_time_seconds": min(binary_timings),
                "mean_wall_time_seconds": float(np.mean(binary_timings)),
                "is_binary": bool(binary_info["is_binary"]),
                "is_255_format": bool(binary_info["is_255_format"]),
                "unique_values": np.asarray(binary_info["unique_values"]).tolist(),
            },
            {
                "name": "mask_from_average",
                "wall_time_seconds": mask_timings,
                "min_wall_time_seconds": min(mask_timings),
                "mean_wall_time_seconds": float(np.mean(mask_timings)),
                "mask_shape": list(mask.shape),
                "mask_dtype": str(mask.get_fdata().dtype),
                "n_mask_voxels": int(np.count_nonzero(mask.get_fdata() > 0)),
            },
            {
                "name": "images_to_matrix",
                "wall_time_seconds": matrix_timings,
                "min_wall_time_seconds": min(matrix_timings),
                "mean_wall_time_seconds": float(np.mean(matrix_timings)),
                **_summarize_array(matrix),
            },
        ],
    }


def collect_metadata() -> dict[str, Any]:
    def module_version(module: Any) -> str:
        return getattr(module, "__version__", "unknown")

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = "unknown"

    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "numba": numba.__version__,
            "sklearn": module_version(sklearn),
        },
        "env": {
            "NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "git_commit": git_commit,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("kernels", "testdata-io"),
        required=True,
        help="Benchmark suite to run.",
    )
    parser.add_argument("--repeat", type=int, default=3, help="Number of measured repeats.")
    parser.add_argument("--output", type=Path, required=True, help="JSON output path.")
    parser.add_argument(
        "--test-data-dir",
        type=Path,
        default=REPO_ROOT / "test_data",
        help="Directory containing normalized_masks/*.nii.gz or normalized_images/*.nii.gz.",
    )
    parser.add_argument(
        "--test-data-kind",
        choices=("masks", "images"),
        default="masks",
        help="Which test_data subdirectory to benchmark for the testdata-io suite.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")

    if args.suite == "kernels":
        suite_result = run_kernels(args.repeat)
    else:
        suite_result = run_testdata_io(args.repeat, args.test_data_dir, args.test_data_kind)

    payload = {
        "metadata": collect_metadata(),
        "result": suite_result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.suite} benchmark results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
