from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import (
    betti_curve,
    load_npz_pair,
    pd_bounds,
    run_pair_1d,
    run_ripser_1d,
    sample_rows,
    save_betti_curve_png,
    save_json,
)


def _parse_int_list(x: str) -> List[int]:
    parts = [p.strip() for p in x.split(",") if p.strip()]
    return [int(p) for p in parts]


def _parse_float_list(x: str) -> List[float]:
    parts = [p.strip() for p in x.split(",") if p.strip()]
    return [float(p) for p in parts]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--src-key", default=None)
    p.add_argument("--tgt-key", default=None)
    p.add_argument("--outdir", default="out")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sizes", default="250,500,1000", help="Comma-separated sample sizes.")
    p.add_argument("--pca-dim", type=int, default=64, help="0 disables PCA.")
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--min-persistence", type=float, default=0.01)
    p.add_argument("--thresh-grid", default="", help="Optional: comma-separated ripser thresh values.")
    p.add_argument("--fixed-size", type=int, default=500, help="Sample size for thresh sweep.")
    p.add_argument("--betti-grid", type=int, default=200, help="Number of radii points for Betti-1 curve.")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    src0, tgt0 = load_npz_pair(args.npz, src_key=args.src_key, tgt_key=args.tgt_key)
    nn = min(len(src0), len(tgt0))
    sizes = [s for s in _parse_int_list(args.sizes) if s > 0 and s <= nn]
    if not sizes:
        sizes = [min(nn, 100)]

    sweep: Dict[str, Any] = {
        "npz": str(Path(args.npz)),
        "sizes": sizes,
        "seed": int(args.seed),
        "normalize": not bool(args.no_normalize),
        "pca_dim": int(args.pca_dim),
        "min_persistence": float(args.min_persistence),
        "results_by_size": {},
        "thresh_sweep": {},
    }

    for s in sizes:
        src, tgt, _ = sample_rows(src0, tgt0, n=int(s), seed=int(args.seed))
        res = run_pair_1d(
            src,
            tgt,
            normalize=not bool(args.no_normalize),
            pca_dim=int(args.pca_dim),
            thresh=None,
            min_persistence=float(args.min_persistence),
        )

        h1_src = np.asarray(res["diagrams"]["src"]["h1"], dtype=float)
        h1_tgt = np.asarray(res["diagrams"]["tgt"]["h1"], dtype=float)
        rmin, rmax = pd_bounds(h1_src, h1_tgt, fallback=(0.0, 1.0))
        if not np.isfinite(rmin) or not np.isfinite(rmax) or rmax <= rmin:
            rmin, rmax = 0.0, 1.0
        radii = np.linspace(rmin, rmax, int(args.betti_grid))
        curve_src = betti_curve(h1_src, radii)
        curve_tgt = betti_curve(h1_tgt, radii)

        sweep["results_by_size"][str(s)] = {
            "pe_h1": res["pe_h1"],
            "h1_count": res["h1_count"],
            "betti1_curve": {
                "radii_min": float(rmin),
                "radii_max": float(rmax),
                "radii": radii.tolist(),
                "src": curve_src.tolist(),
                "tgt": curve_tgt.tolist(),
            },
        }

        if not args.no_plots:
            save_betti_curve_png(
                outdir / f"betti1_curve_size_{s}.png",
                radii,
                curve_src,
                curve_tgt,
                title=f"Betti-1 Curves (Sample Size={s})",
            )

    thr_vals: List[float] = []
    if args.thresh_grid.strip():
        thr_vals = [t for t in _parse_float_list(args.thresh_grid) if t > 0]
    if thr_vals:
        sfix = min(int(args.fixed_size), nn)
        src_fix, tgt_fix, _ = sample_rows(src0, tgt0, n=sfix, seed=int(args.seed))
        if not bool(args.no_normalize):
            from .core import l2_normalize

            src_fix = l2_normalize(src_fix)
            tgt_fix = l2_normalize(tgt_fix)
        if int(args.pca_dim) > 0:
            from .core import pca_reduce

            src_fix = pca_reduce(src_fix, int(args.pca_dim))
            tgt_fix = pca_reduce(tgt_fix, int(args.pca_dim))

        ent_src: List[float] = []
        ent_tgt: List[float] = []
        delta: List[float] = []
        for thr in thr_vals:
            _, h1s = run_ripser_1d(src_fix, thresh=float(thr))
            _, h1t = run_ripser_1d(tgt_fix, thresh=float(thr))
            from .core import persistence_entropy

            es = persistence_entropy(h1s)
            et = persistence_entropy(h1t)
            ent_src.append(es)
            ent_tgt.append(et)
            delta.append(et - es)

        sweep["thresh_sweep"] = {
            "fixed_size": int(sfix),
            "thresh": thr_vals,
            "pe_h1_src": ent_src,
            "pe_h1_tgt": ent_tgt,
            "delta_tgt_minus_src": delta,
        }

    save_json(outdir / "sweep_1d.json", sweep)


if __name__ == "__main__":
    main()
