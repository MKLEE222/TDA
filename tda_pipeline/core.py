from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from ripser import ripser
except Exception:
    ripser = None


def load_npz_pair(
    npz_path: str | Path,
    src_key: Optional[str] = None,
    tgt_key: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    p = Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    data = np.load(str(p))

    if src_key is not None and tgt_key is not None:
        if src_key not in data.files or tgt_key not in data.files:
            raise ValueError(f"NPZ keys not found: {src_key}, {tgt_key}")
        src = data[src_key]
        tgt = data[tgt_key]
        return src, tgt

    key_pairs: List[Tuple[str, str]] = [
        ("src_emb", "tgt_emb"),
        ("src", "tgt"),
        ("src_embeddings", "tgt_embeddings"),
        ("source", "target"),
        ("zh", "en"),
        ("X_src", "X_tgt"),
    ]
    for ks, kt in key_pairs:
        if ks in data.files and kt in data.files:
            return data[ks], data[kt]

    if "pairs" in data.files:
        arr = data["pairs"]
        if arr.ndim == 3 and arr.shape[0] == 2:
            return arr[0], arr[1]

    raise ValueError(f"Cannot infer src/tgt keys from NPZ. Available keys: {data.files}")


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def pca_reduce(X: np.ndarray, k: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if k <= 0 or X.shape[1] <= k:
        return X
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    kk = min(k, Vt.shape[0])
    return Xc @ Vt[:kk].T


def sample_rows(
    src: np.ndarray,
    tgt: np.ndarray,
    n: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nn = min(len(src), len(tgt))
    if n > nn:
        raise ValueError(f"sample size {n} exceeds available pairs {nn}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(nn, size=n, replace=False)
    return src[idx], tgt[idx], idx


def run_ripser_1d(
    X: np.ndarray,
    thresh: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if ripser is None:
        raise RuntimeError("ripser is not available. Install ripser to run persistent homology.")
    kwargs: Dict[str, Any] = {"maxdim": 1}
    if thresh is not None:
        kwargs["thresh"] = float(thresh)
    res = ripser(np.asarray(X, dtype=float), **kwargs)
    dgms = res.get("dgms", [])
    h0 = dgms[0] if len(dgms) > 0 else np.zeros((0, 2), dtype=float)
    h1 = dgms[1] if len(dgms) > 1 else np.zeros((0, 2), dtype=float)
    return np.asarray(h0, dtype=float), np.asarray(h1, dtype=float)


def persistence_entropy(diagram: np.ndarray, eps: float = 1e-12) -> float:
    diagram = np.asarray(diagram, dtype=float)
    if diagram.size == 0:
        return 0.0
    life = diagram[:, 1] - diagram[:, 0]
    life = life[np.isfinite(life)]
    life = life[life > 0]
    if life.size == 0:
        return 0.0
    p = life / float(life.sum())
    return float(-(p * np.log(p + eps)).sum())


def count_bars(diagram: np.ndarray, min_persistence: float = 0.01) -> int:
    diagram = np.asarray(diagram, dtype=float)
    if diagram.size == 0:
        return 0
    life = diagram[:, 1] - diagram[:, 0]
    return int(np.sum(life >= float(min_persistence)))


def betti_curve(diagram: np.ndarray, radii: np.ndarray) -> np.ndarray:
    diagram = np.asarray(diagram, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if diagram.size == 0:
        return np.zeros_like(radii, dtype=int)
    births = diagram[:, 0]
    deaths = diagram[:, 1]
    out = np.zeros_like(radii, dtype=int)
    for i, r in enumerate(radii):
        out[i] = int(np.sum((births <= r) & (r < deaths)))
    return out


def pd_bounds(h1a: np.ndarray, h1b: np.ndarray, fallback: Tuple[float, float] = (0.0, 1.0)) -> Tuple[float, float]:
    vals: List[float] = []
    for h1 in (h1a, h1b):
        h1 = np.asarray(h1, dtype=float)
        if h1.size:
            vals.extend([float(np.min(h1[:, 0])), float(np.max(h1[:, 1]))])
    if not vals:
        return float(fallback[0]), float(fallback[1])
    return float(min(vals)), float(max(vals))


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_pd_png(path: str | Path, dgms: Sequence[np.ndarray], title: str) -> None:
    import matplotlib.pyplot as plt

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 4), dpi=200)
    colors = {0: "#e67e22", 1: "#3498db"}
    labels = {0: "H0", 1: "H1"}
    maxv = 1.0
    for idx in range(min(2, len(dgms))):
        arr = np.asarray(dgms[idx], dtype=float)
        if arr.size:
            plt.scatter(arr[:, 0], arr[:, 1], s=10, c=colors[idx], label=labels[idx], alpha=0.8)
            maxv = max(maxv, float(np.max(arr)))
    plt.plot([0, maxv], [0, maxv], "k--", linewidth=0.8)
    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.legend(loc="best")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(str(p))
    plt.close()


def save_betti_curve_png(
    path: str | Path,
    radii: np.ndarray,
    curve_src: np.ndarray,
    curve_tgt: np.ndarray,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 3), dpi=200)
    plt.plot(radii, curve_src, label="Source", color="#2c3e50", linewidth=2)
    plt.plot(radii, curve_tgt, label="Translation", color="#e74c3c", linewidth=2)
    plt.xlabel("Filtration radius")
    plt.ylabel("Betti-1 Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(p))
    plt.close()


def run_pair_1d(
    src: np.ndarray,
    tgt: np.ndarray,
    *,
    normalize: bool = True,
    pca_dim: int = 64,
    thresh: Optional[float] = None,
    min_persistence: float = 0.01,
) -> Dict[str, Any]:
    if normalize:
        src = l2_normalize(src)
        tgt = l2_normalize(tgt)
    if pca_dim > 0:
        src = pca_reduce(src, pca_dim)
        tgt = pca_reduce(tgt, pca_dim)

    h0_src, h1_src = run_ripser_1d(src, thresh=thresh)
    h0_tgt, h1_tgt = run_ripser_1d(tgt, thresh=thresh)

    pe_src = persistence_entropy(h1_src)
    pe_tgt = persistence_entropy(h1_tgt)
    c_src = count_bars(h1_src, min_persistence=min_persistence)
    c_tgt = count_bars(h1_tgt, min_persistence=min_persistence)

    return {
        "pe_h1": {"src": pe_src, "tgt": pe_tgt, "delta_tgt_minus_src": pe_tgt - pe_src},
        "h1_count": {"src": c_src, "tgt": c_tgt},
        "thresh": None if thresh is None else float(thresh),
        "pca_dim": int(pca_dim),
        "normalize": bool(normalize),
        "diagrams": {
            "src": {"h0": h0_src.tolist(), "h1": h1_src.tolist()},
            "tgt": {"h0": h0_tgt.tolist(), "h1": h1_tgt.tolist()},
        },
    }
