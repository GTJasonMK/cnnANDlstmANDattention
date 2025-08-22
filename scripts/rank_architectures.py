from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("pandas is required. Please install with: pip install pandas")

# Optional plotting
try:
    import matplotlib.pyplot as plt  # type: ignore
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False


# -----------------------------
# Parsing helpers
# -----------------------------

SLUG_RE = re.compile(
    r"(?P<cnn>standard|depthwise|dilated|inception|tcn)-(?P<rnn>lstm|gru|ssm)"
    r"-(?P<attn>standard|multiscale|local|conformer|spatiotemporal)"
    r"-pos(?P<pos>none|absolute|alibi|rope)"
    r"-ca(?P<ca>on|off)",
    re.IGNORECASE,
)

# ctrl yaml names like: ctrl_cnn-tcn__tcn-standard-lstm-posnone-caoff[_patched].yaml
CTRL_NAME_RE = re.compile(
    r"ctrl_(?P<factor>[^_]+)__(?P<slug>[^.]+)", re.IGNORECASE
)


def _lower(x: Optional[str]) -> Optional[str]:
    return None if x is None else str(x).strip().lower()


def parse_slug_from_text(text: str) -> Dict[str, Optional[str]]:
    """Try to parse architecture fields from a slug embedded in text.
    Supports both export filename slugs and ctrl yaml names.
    Returns dict with keys: cnn,rnn,attn,pos,ca.
    """
    text_l = text.lower()
    # ctrl pattern first
    m = CTRL_NAME_RE.search(text_l)
    if m:
        slug = m.group("slug")
        # strip suffixes like _patched, .yaml left out earlier
        slug = slug.replace("_patched", "")
        m2 = SLUG_RE.search(slug)
        if m2:
            d = m2.groupdict()
            d = {k: (v.lower() if isinstance(v, str) else v) for k, v in d.items()}
            return d
    # direct slug
    m = SLUG_RE.search(text_l)
    if m:
        d = m.groupdict()
        return {k: (v.lower() if isinstance(v, str) else v) for k, v in d.items()}
    return {"cnn": None, "rnn": None, "attn": None, "pos": None, "ca": None}


def extract_arch_from_row(row: pd.Series) -> Dict[str, Any]:
    # Prefer explicit columns if present
    fields = {
        "cnn": _lower(row.get("cnn_variant")) or _lower(row.get("cnn")),
        "rnn": _lower(row.get("rnn_type")) or _lower(row.get("rnn")),
        "attn": _lower(row.get("attention_variant")) or _lower(row.get("attn")) or _lower(row.get("attention")),
        "pos": _lower(row.get("positional")) or _lower(row.get("positional_mode")) or _lower(row.get("pos")),
        "ca": _lower(row.get("channel_attention")) or _lower(row.get("ca")),
    }
    # Normalize boolean-like CA
    if fields["ca"] in ("true", "1", "on", "yes"):
        fields["ca"] = "on"
    elif fields["ca"] in ("false", "0", "off", "no"):
        fields["ca"] = "off"

    # Wavelet
    wavelet_enabled = None
    wavelet_base = None
    for key in ["data.wavelet.enabled", "wavelet_enabled", "wavelet.enable", "wavelet_on"]:
        v = row.get(key)
        if v is not None:
            s = str(v).strip().lower()
            wavelet_enabled = (s in ("true", "1", "yes", "on"))
            break
    for key in ["data.wavelet.wavelet", "wavelet", "wavelet_base", "wavelet.name"]:
        v = row.get(key)
        if v is not None and str(v).strip().lower() not in ("nan", "none", ""):
            wavelet_base = str(v).strip().lower()
            break

    # If missing, attempt parsing from checkpoint/yaml name or run_name
    text_sources = [
        row.get("checkpoint"), row.get("checkpoint_name"), row.get("model_name"), row.get("yaml"),
        row.get("yaml_name"), row.get("config_name"), row.get("run_name"), row.get("file"), row.get("path")
    ]
    for t in text_sources:
        if t and any(v is None for v in fields.values()):
            d = parse_slug_from_text(str(t))
            for k, v in d.items():
                if fields.get(k) is None and v is not None:
                    fields[k] = v

    # Default values
    fields["cnn"] = fields["cnn"] or "standard"
    fields["rnn"] = fields["rnn"] or "lstm"
    fields["attn"] = fields["attn"] or "standard"
    fields["pos"] = fields["pos"] or "none"
    fields["ca"] = fields["ca"] or "off"

    # Wavelet defaults
    if wavelet_enabled is None:
        wavelet_enabled = False
    fields["wavelet_on"] = bool(wavelet_enabled)
    fields["wavelet_base"] = wavelet_base if wavelet_enabled else "none"

    return fields


# -----------------------------
# Ranking
# -----------------------------

def resolve_metric(df: pd.DataFrame, metric: str) -> str:
    """Resolve the actual metric column present in df.
    Accepts names like 'loss', 'val_loss', 'test_loss'. If the requested name is
    missing, tries common prefixed/unprefixed alternatives.
    """
    cols = set(df.columns)
    if metric in cols:
        return metric
    # derive base (strip common prefixes)
    base = metric
    for p in ("val_", "valid_", "test_"):
        if metric.startswith(p):
            base = metric[len(p):]
            break
    candidates = [
        metric,
        f"val_{base}", f"valid_{base}", f"test_{base}", base,
    ]
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"Metric '{metric}' not found. Tried candidates: {candidates}. Available: {list(df.columns)[:30]} ...")


def group_rank(df: pd.DataFrame, group_col: str, metric: str, ascending: bool) -> pd.DataFrame:
    if metric not in df.columns:
        metric = resolve_metric(df, metric)
    g = (
        df.groupby(group_col)[metric]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .rename(columns={"count": "n", "min": "best" if ascending else "worst", "max": "worst" if ascending else "best"})
    )
    g = g.sort_values(by=["mean", "best" if ascending else "worst"], ascending=[ascending, ascending])
    g.index.name = group_col
    return g.reset_index()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def make_plots(df: pd.DataFrame, group_col: str, metric: str, out_dir: Path, topk: Optional[int], ascending: bool):
    if not _HAS_PLT:
        return
    # Bar plot of mean metric
    rank = group_rank(df, group_col, metric, ascending)
    if topk and topk > 0:
        rank = rank.head(topk)
    plt.figure(figsize=(8, 4))
    plt.bar(rank[group_col].astype(str), rank["mean"], color="#4C72B0")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(metric)
    plt.title(f"Mean {metric} by {group_col}")
    plt.tight_layout()
    plt.savefig(out_dir / f"rank_{group_col}_bar.png", dpi=150)
    plt.close()

    # Boxplot of distribution
    # Keep only rows for the levels we plotted
    levels = list(rank[group_col].astype(str).values)
    sub = df[df[group_col].astype(str).isin(levels)]
    plt.figure(figsize=(9, 4))
    sub.boxplot(column=metric, by=group_col, grid=False, rot=30)
    plt.suptitle("")
    plt.title(f"{metric} distribution by {group_col}")
    plt.tight_layout()
    plt.savefig(out_dir / f"rank_{group_col}_box.png", dpi=150)
    plt.close()


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Rank architecture variants from batch_eval CSV results")
    ap.add_argument("--input", required=True, help="Path to batch_eval CSV file")
    ap.add_argument("--metric", default="val_loss", help="Metric column name to rank by (default: val_loss)")
    ap.add_argument("--output-dir", default=".", help="Directory to save ranking reports and plots")
    ap.add_argument("--ascending", action="store_true", help="Sort ascending (loss metrics). Default True.")
    ap.add_argument("--descending", action="store_true", help="Force descending (score metrics like R2)")
    ap.add_argument("--top-k", type=int, default=0, help="Show top-K rows per dimension (0=all)")
    ap.add_argument("--plots", action="store_true", help="Generate bar/box plots if matplotlib is available")
    args = ap.parse_args()

    ascending = True
    if args.descending:
        ascending = False
    elif args.ascending:
        ascending = True

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    plots_dir = out_dir / "plots"
    ensure_dir(plots_dir)

    # Load
    df = pd.read_csv(args.input)
    used_metric = resolve_metric(df, args.metric)
    if used_metric != args.metric:
        print(f"[INFO] Using metric column '{used_metric}' (requested '{args.metric}')")

    # Enrich with architecture fields
    arch_rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        arch_rows.append(extract_arch_from_row(row))
    arch_df = pd.DataFrame(arch_rows)
    mdf = pd.concat([df.reset_index(drop=True), arch_df.reset_index(drop=True)], axis=1)

    # Dimensions to rank
    dims = [
        ("cnn", "CNN Backbones"),
        ("rnn", "RNN Types"),
        ("attn", "Attention Variants"),
        ("pos", "Positional Encoding"),
        ("ca", "Channel Attention (on/off)"),
        ("wavelet_on", "Wavelet Enabled"),
        ("wavelet_base", "Wavelet Base"),
    ]

    # Write summary per dimension
    summary_paths = []
    for col, title in dims:
        try:
            ranked = group_rank(mdf, col, used_metric, ascending)
        except KeyError as e:
            print(f"[WARN] Skip {col}: {e}")
            continue
        if args.top_k and args.top_k > 0:
            disp = ranked.head(args.top_k)
        else:
            disp = ranked
        print("\n" + "=" * 80)
        print(f"{title} ranked by {used_metric} ({'asc' if ascending else 'desc'})")
        print(disp.to_string(index=False))
        # Save CSV
        path = out_dir / f"rank_{col}.csv"
        ranked.to_csv(path, index=False)
        summary_paths.append(path)
        # Plots (optional)
        if args.plots:
            make_plots(mdf, col, used_metric, plots_dir, args.top_k, ascending)

    # Save merged overview with selected columns
    cols_to_keep = [
        "cnn", "rnn", "attn", "pos", "ca", "wavelet_on", "wavelet_base", used_metric,
    ]
    cols_to_keep += [c for c in ["checkpoint", "yaml", "config_name", "run_name"] if c in mdf.columns]
    mdf[cols_to_keep].to_csv(out_dir / "eval_with_arch.csv", index=False)

    print("\nDone. Reports saved to:")
    for p in summary_paths:
        print(" -", p)
    if args.plots and _HAS_PLT:
        print(" -", plots_dir)
    elif args.plots:
        print("[WARN] matplotlib not available; plots were skipped.")


if __name__ == "__main__":
    main()

