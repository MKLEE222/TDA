from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from .core import (
    load_npz_pair,
    pd_bounds,
    run_pair_1d,
    sample_rows,
    save_json,
    save_pd_png,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="NPZ file containing src/tgt embeddings.")
    p.add_argument("--src-key", default=None, help="Optional: override src embedding key in NPZ.")
    p.add_argument("--tgt-key", default=None, help="Optional: override tgt embedding key in NPZ.")
    p.add_argument("--outdir", default="out", help="Output directory.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-size", type=int, default=0, help="0 means use all pairs; otherwise sample N pairs.")
    p.add_argument("--pca-dim", type=int, default=64, help="0 disables PCA.")
    p.add_argument("--no-normalize", action="store_true", help="Disable L2 normalization.")
    p.add_argument("--thresh", type=float, default=None, help="Optional ripser thresh.")
    p.add_argument("--min-persistence", type=float, default=0.01, help="Threshold for counting H1 bars.")
    p.add_argument("--no-plots", action="store_true", help="Disable PD plots.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    src, tgt = load_npz_pair(args.npz, src_key=args.src_key, tgt_key=args.tgt_key)
    nn = min(len(src), len(tgt))
    if args.sample_size and args.sample_size > 0:
        src, tgt, idx = sample_rows(src, tgt, n=int(args.sample_size), seed=int(args.seed))
    else:
        src = src[:nn]
        tgt = tgt[:nn]
        idx = np.arange(nn, dtype=int)

    res = run_pair_1d(
        src,
        tgt,
        normalize=not bool(args.no_normalize),
        pca_dim=int(args.pca_dim),
        thresh=args.thresh,
        min_persistence=float(args.min_persistence),
    )
    res["npz"] = str(Path(args.npz))
    res["n_pairs"] = int(len(idx))
    res["seed"] = int(args.seed)

    save_json(outdir / "metrics_1d.json", res)

    if not args.no_plots:
        dgms_src = [np.asarray(res["diagrams"]["src"]["h0"]), np.asarray(res["diagrams"]["src"]["h1"])]
        dgms_tgt = [np.asarray(res["diagrams"]["tgt"]["h0"]), np.asarray(res["diagrams"]["tgt"]["h1"])]
        save_pd_png(outdir / "pd_src.png", dgms_src, title="Persistence Diagram (Source)")
        save_pd_png(outdir / "pd_tgt.png", dgms_tgt, title="Persistence Diagram (Translation)")


if __name__ == "__main__":
    main()
