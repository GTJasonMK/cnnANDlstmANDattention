from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

# Optional plotting libs (graceful fallback if not installed)
try:
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

try:
    import plotly.express as px  # type: ignore
    import plotly.io as pio  # type: ignore
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False


# -----------------------------
# Helpers
# -----------------------------

PREFERRED_METRICS_ASC = [
    ("mse", True), ("test_mse", True), ("val_mse", True),
    ("rmse", True), ("test_rmse", True), ("val_rmse", True),
    ("mae", True), ("test_mae", True), ("val_mae", True),
    ("r2", False), ("test_r2", False), ("val_r2", False),
]

GROUP_COLS_ARCH = [
    "cnn_variant", "rnn_type", "attn_variant", "pos", "ca", "direction", "heads",
]
GROUP_COLS_DATA = [
    "dataset", "seq_len", "horizon", "normalize", "wav_on", "wavelet_base",
]


def _lower(x: Optional[str]) -> Optional[str]:
    return None if x is None else str(x).strip().lower()


def _resolve_metric(df: pd.DataFrame, prefer: Optional[str]) -> Tuple[str, bool]:
    if prefer and prefer in df.columns:
        # ascending for most, r2 is higher-better
        asc = prefer.lower() != "r2"
        return prefer, asc
    for name, asc in PREFERRED_METRICS_ASC:
        if name in df.columns:
            return name, asc
    # fallback: any numeric column except obvious ids
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c, True
    raise ValueError("No numeric metric column found in CSV")


def _col(df: pd.DataFrame, candidates: List[str], default: Optional[str] = None) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return default


def _normalize_key_cols(df: pd.DataFrame) -> pd.DataFrame:
    renames = {}
    # unify names that may vary across versions
    if "attn_variant" not in df.columns and "attn" in df.columns:
        renames["attn"] = "attn_variant"
    if "pos" not in df.columns and "positional_mode" in df.columns:
        renames["positional_mode"] = "pos"
    if "cnn_variant" not in df.columns and "cnn" in df.columns:
        renames["cnn"] = "cnn_variant"
    if "rnn_type" not in df.columns and "rnn" in df.columns:
        renames["rnn"] = "rnn_type"
    df = df.rename(columns=renames)
    # lowercase string categorical values
    for c in [
        "cnn_variant", "rnn_type", "attn_variant", "pos", "ca", "dataset",
        "normalize", "wavelet_base",
    ]:
        if c in df.columns:
            df[c] = df[c].astype(str).map(lambda x: x.strip().lower())
    # boolean-like to bool
    for c in ["wav_on", "wavelet_on"]:
        if c in df.columns:
            df[c] = df[c].map(lambda v: str(v).strip().lower() in ("1","true","yes","on"))
    # integer-like
    for c in ["seq_len", "horizon", "heads"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Plotting (Matplotlib/Seaborn)
# -----------------------------

def bar_and_box(df: pd.DataFrame, group_col: str, metric: str, ascending: bool, outdir: Path) -> None:
    if not _HAS_MPL:
        return
    gdf = df[[group_col, metric]].dropna()
    if gdf.empty:
        return
    # Bar: mean +/- std
    plt.figure(figsize=(10, 5))
    order = (
        gdf.groupby(group_col)[metric]
        .mean()
        .sort_values(ascending=ascending)
        .index
        .tolist()
    )
    sns.barplot(data=gdf, x=group_col, y=metric, order=order, errorbar='sd')
    plt.title(f"{group_col} vs {metric} (lower is better)" if ascending else f"{group_col} vs {metric} (higher is better)")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(outdir / f"bar_{group_col}_{metric}.png", dpi=200)
    plt.close()

    # Box plot
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=gdf, x=group_col, y=metric, order=order)
    plt.title(f"{group_col} distribution of {metric}")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(outdir / f"box_{group_col}_{metric}.png", dpi=200)
    plt.close()


def scatter_and_reg(df: pd.DataFrame, x: str, y: str, hue: Optional[str], outdir: Path) -> None:
    if not _HAS_MPL:
        return
    if x not in df.columns or y not in df.columns:
        return
    sdf = df[[x, y] + ([hue] if hue and hue in df.columns else [])].dropna()
    if sdf.empty:
        return
    plt.figure(figsize=(7, 5))
    if hue and hue in sdf.columns:
        sns.scatterplot(data=sdf, x=x, y=y, hue=hue, alpha=0.8)
    else:
        sns.scatterplot(data=sdf, x=x, y=y, alpha=0.8)
    try:
        sns.regplot(data=sdf, x=x, y=y, scatter=False, color='k')
    except Exception:
        pass
    plt.tight_layout()
    plt.savefig(outdir / f"scatter_{x}_vs_{y}.png", dpi=200)
    plt.close()


def heatmap_pivot(df: pd.DataFrame, x: str, y: str, metric: str, outdir: Path) -> None:
    if not _HAS_MPL:
        return
    if x not in df.columns or y not in df.columns or metric not in df.columns:
        return
    pdf = df[[x, y, metric]].dropna()
    if pdf.empty:
        return
    pivot = pdf.pivot_table(index=y, columns=x, values=metric, aggfunc='mean')
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='.3g', cmap='viridis')
    plt.title(f"Mean {metric} over {x} x {y}")
    plt.tight_layout()
    plt.savefig(outdir / f"heatmap_{x}_by_{y}_{metric}.png", dpi=200)
    plt.close()


# -----------------------------
# Plotly (optional interactive)
# -----------------------------

def interactive_group_bar(df: pd.DataFrame, group_col: str, metric: str, ascending: bool, outdir: Path) -> None:
    if not _HAS_PLOTLY:
        return
    g = df[[group_col, metric]].dropna()
    if g.empty:
        return
    ord_idx = g.groupby(group_col)[metric].mean().sort_values(ascending=ascending).index
    fig = px.bar(g, x=group_col, y=metric, category_orders={group_col: list(ord_idx)})
    fig.update_layout(title=f"{group_col}: {metric}")
    pio.write_html(fig, file=str(outdir / f"interactive_bar_{group_col}_{metric}.html"), auto_open=False)


def interactive_scatter(df: pd.DataFrame, x: str, y: str, color: Optional[str], outdir: Path) -> None:
    if not _HAS_PLOTLY:
        return
    if x not in df.columns or y not in df.columns:
        return
    sdf = df[[x, y] + ([color] if color and color in df.columns else [])].dropna()
    if sdf.empty:
        return
    fig = px.scatter(sdf, x=x, y=y, color=color)
    fig.update_layout(title=f"{x} vs {y}")
    pio.write_html(fig, file=str(outdir / f"interactive_scatter_{x}_vs_{y}.html"), auto_open=False)


# -----------------------------
# Top-K best models table
# -----------------------------

def make_top_k_table(df: pd.DataFrame, metric: str, ascending: bool, outdir: Path, k: int = 20) -> pd.DataFrame:
    cols = [c for c in [
        "model", "checkpoint", "dataset", "seq_len", "horizon", "normalize",
        "cnn_variant", "rnn_type", "attn_variant", "pos", "ca", "heads", "params",
        metric,
    ] if c in df.columns]
    tdf = df.sort_values(by=metric, ascending=ascending).head(k)[cols]
    tdf.to_csv(outdir / f"top_{k}_{metric}.csv", index=False, encoding='utf-8')
    return tdf


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Visualize evaluation CSV from batch_eval.py")
    ap.add_argument("--csv", default=str(Path("getmodel/batch_eval/batch_metrics.csv")), help="Path to evaluation CSV")
    ap.add_argument("--output-dir", default=str(Path("getmodel/rank/plots")), help="Directory to save plots")
    ap.add_argument("--metric", default=None, help="Metric column to use (auto if omitted)")
    ap.add_argument("--top-k", type=int, default=20, help="Top-K best models table size")
    ap.add_argument("--no-static", action="store_true", help="Disable static matplotlib/seaborn plots")
    ap.add_argument("--interactive", action="store_true", help="Also output Plotly interactive HTML charts if available")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    outdir = Path(args.output_dir)
    _ensure_dir(outdir)

    df = pd.read_csv(csv_path)
    df = _normalize_key_cols(df)

    metric, ascending = _resolve_metric(df, args.metric)
    print(f"[INFO] Using metric: {metric} (ascending={ascending})")

    # Save merged overview columns for convenience
    df.to_csv(outdir / "eval_with_arch.csv", index=False, encoding='utf-8')

    # Static plots
    if _HAS_MPL and not args.no_static:
        sns.set_theme(style="whitegrid")
        # 1) Architecture components
        for col in GROUP_COLS_ARCH:
            if col in df.columns:
                bar_and_box(df, col, metric, ascending, outdir)
        # 2) Data config
        for col in GROUP_COLS_DATA:
            if col in df.columns:
                bar_and_box(df, col, metric, ascending, outdir)
        # 3) Scatter: heads/params/seq_len/horizon vs metric
        for x in ["heads", "params", "seq_len", "horizon"]:
            if x in df.columns:
                scatter_and_reg(df, x, metric, hue=_col(df, ["attn_variant", "attn"]), outdir=outdir)
        # 4) Heatmap: seq_len x horizon
        if "seq_len" in df.columns and "horizon" in df.columns:
            heatmap_pivot(df, x="seq_len", y="horizon", metric=metric, outdir=outdir)

    # Interactive plots
    if args.interactive and _HAS_PLOTLY:
        for col in GROUP_COLS_ARCH + GROUP_COLS_DATA:
            if col in df.columns:
                interactive_group_bar(df, col, metric, ascending, outdir)
        for x in ["heads", "params", "seq_len", "horizon"]:
            if x in df.columns:
                interactive_scatter(df, x, metric, color=_col(df, ["attn_variant", "attn"]), outdir=outdir)

    # Top-K best models table
    topk = make_top_k_table(df, metric, ascending, outdir, k=int(args.top_k))
    print(f"[INFO] Saved top-{args.top_k} table with columns: {list(topk.columns)}")
    print(f"[INFO] Plots saved under: {outdir}")


if __name__ == "__main__":
    main()

