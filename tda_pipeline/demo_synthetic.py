from __future__ import annotations

from pathlib import Path

import numpy as np

from .core import save_json
from .run_1d import main as run_1d_main


def make_synthetic_pair(n: int = 800, d: int = 64, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=0.0, scale=1.0, size=(n, d))
    b = rng.normal(loc=0.0, scale=1.0, size=(n, d))
    rot = rng.normal(size=(d, d))
    u, _, vt = np.linalg.svd(rot, full_matrices=False)
    r = u @ vt
    b = b @ r
    b[:, :4] += 0.15
    return a, b


def main() -> None:
    outdir = Path("out_synth")
    outdir.mkdir(parents=True, exist_ok=True)
    src, tgt = make_synthetic_pair()
    npz_path = outdir / "synthetic_pair.npz"
    np.savez_compressed(npz_path, src=src, tgt=tgt)
    save_json(outdir / "synthetic_meta.json", {"n": int(src.shape[0]), "d": int(src.shape[1])})

    import sys

    sys.argv = [
        sys.argv[0],
        "--npz",
        str(npz_path),
        "--outdir",
        str(outdir),
        "--sample-size",
        "500",
        "--pca-dim",
        "32",
    ]
    run_1d_main()


if __name__ == "__main__":
    main()
