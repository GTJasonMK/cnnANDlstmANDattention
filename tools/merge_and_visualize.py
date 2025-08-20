
import os, re, json, math, argparse
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

def _read_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    # 尝试多种编码
    for enc in ["utf-8-sig", "utf-8", "gbk", "utf-16", "utf-16le", "utf-16be"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, engine="python", error_bad_lines=False)

def _std_cols(df):
    # 统一列名到小写
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    # 将可能出现的变体列名做一次映射
    rename = {"rnn": "rnn_type", "attention": "attn_variant", "attn": "attn_variant"}
    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
    return df

def _arch_from_row(row):
    # 优先使用显式字段
    cnn = str(row.get("cnn_variant", "") or "").lower()
    rnn = str(row.get("rnn_type", "") or "").lower()
    attn = str(row.get("attn_variant", "") or "").lower()
    if not cnn or not rnn or not attn:
        # 尝试从 model 名/简写推断（如 sgs -> standard+gru+standard）
        m = str(row.get("model", "") or "").lower()
        if re.fullmatch(r"[sdlt][lg][sm]", m):
            mp_cnn = {"s":"standard","d":"dilated","l":"depthwise","t":"tcn"}
            mp_rnn = {"l":"lstm","g":"gru"}
            mp_att = {"s":"standard","m":"multiscale"}
            cnn = mp_cnn.get(m[0], cnn)
            rnn = mp_rnn.get(m[1], rnn)
            attn = mp_att.get(m[2], attn)
    zh = {
        "standard":"标准CNN", "depthwise":"深度可分离CNN", "dilated":"空洞CNN", "tcn":"TCN",
        "lstm":"LSTM", "gru":"GRU",
        "standard_attention":"标准注意力", "standard":"标准注意力", "multiscale":"多尺度注意力"
    }
    cnn_zh = zh.get(cnn, cnn or "未知CNN")
    rnn_zh = zh.get(rnn, rnn or "未知RNN")
    attn_zh = zh.get(attn, attn or "未知注意力")
    return f"{cnn_zh}+{rnn_zh}+{attn_zh}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output", required=True, help="输出目录，用于保存合并CSV与图像")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # 读取与合并
    rows = []
    for fn in os.listdir(args.input_dir):
        if not fn.lower().endswith((".csv",".xlsx",".xls")): continue
        path = os.path.join(args.input_dir, fn)
        df = _std_cols(_read_table(path))
        dataset = os.path.splitext(fn)[0].lower()
        if "status" in df.columns:
            df = df[(~df["status"].astype(str).str.startswith("error", na=False))]
        df["dataset"] = dataset
        if "arch" not in df.columns:
            df["arch"] = df.apply(_arch_from_row, axis=1)
        rows.append(df)
    if not rows:
        raise RuntimeError("未在目录中找到可读取的对比表格")
    all_df = pd.concat(rows, ignore_index=True)

    # 指标列（存在即用）
    metric_candidates = ["mse","mae","rmse","mape","r2"]
    metrics = [m for m in metric_candidates if m in all_df.columns]
    for m in metrics:
        all_df[m] = pd.to_numeric(all_df[m], errors="coerce")
    all_df.to_csv(os.path.join(args.output, "merged_results.csv"), index=False, encoding="utf-8-sig")

    # 排序：按各指标在三个数据集的平均进行，对每个指标独立排序
    datasets = sorted(all_df["dataset"].unique())
    arches = sorted(all_df["arch"].unique())
    colors = ["#4C78A8", "#F58518", "#54A24B"]  # 三数据集配色

    # 中文字体（若系统无黑体则回退）
    plt.rcParams["font.sans-serif"] = ["SimHei","Microsoft YaHei","Arial Unicode MS","DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(max(8, 1.2*len(arches)), 3.6*n_metrics), constrained_layout=True)
    if n_metrics == 1: axes = [axes]

    for idx, m in enumerate(metrics):
        ax = axes[idx]
        # 计算每个架构-数据集的均值
        pv = all_df.pivot_table(index="arch", columns="dataset", values=m, aggfunc="mean")
        # 对当前指标按跨数据集的均值排序，便于阅读
        pv["__avg__"] = pv.mean(axis=1)
        pv = pv.sort_values("__avg__")
        pv = pv.drop(columns="__avg__")
        x_names = list(pv.index)
        x = np.arange(len(x_names))
        w = 0.8 / max(1, len(datasets))

        for di, ds in enumerate(datasets):
            y = pv[ds].values if ds in pv.columns else np.array([np.nan]*len(x_names))
            ax.bar(x + di*w - 0.4 + w/2, y, width=w, label=ds, color=colors[di % len(colors)], edgecolor="k", alpha=0.85)

        ax.set_title(f"{m.upper()} 指标对比", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(x_names, rotation=25, ha="right")
        ax.set_ylabel(m.upper())
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(title="数据集", fontsize=9)

    fig.suptitle("三数据集模型架构性能对比", fontsize=14)
    out_png = os.path.join(args.output, "summary_all_datasets.png")
    out_pdf = os.path.join(args.output, "summary_all_datasets.pdf")
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=args.dpi, bbox_inches="tight")
    print(f"已生成: {out_png}\n已生成: {out_pdf}\n合并表: {os.path.join(args.output,'merged_results.csv')}")
if __name__ == "__main__":
    main()

