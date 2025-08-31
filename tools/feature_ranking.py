import argparse
import os
from typing import List, Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "对模型结果CSV按控制变量法进行特征价值排名：\n"
            "- 仅允许某一特征不同、其他非评估列完全一致的样本组成一组；\n"
            "- 在每组内按指定评估指标(默认mse)升序排名；\n"
            "- 为每个特征分别输出一个CSV，列为：[特征名, 特征值, 排名位次, 参与对比的模型序号, 参与对比的模型mse]。"
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="输入的batch_metrics.csv路径",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        type=str,
        required=True,
        help="输出目录（每个特征一个CSV）",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=None,
        help="可选：若提供，将为每个特征输出mse柱状图PNG到该目录",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="mse",
        choices=["mse", "mae", "rmse", "mape", "r2"],
        help="用于排名的评估指标(默认mse，升序更好)",
    )
    parser.add_argument(
        "--drop_admin_cols",
        type=str,
        default="model,status,checkpoint,params,arch",
        help=(
            "参与控制(必须相同)时需要忽略的非评估行政/派生列，逗号分隔。"
            "默认: model,status,checkpoint,params,arch"
        ),
    )
    parser.add_argument(
        "--min_group_size",
        type=int,
        default=2,
        help="仅当一组内可对比的不同取值数不少于该值时才纳入排名",
    )
    parser.add_argument(
        "--aggregate_duplicates",
        type=str,
        default="best",
        choices=["best", "mean"],
        help=(
            "当同一控制组内存在相同特征值的重复样本时的聚合方式：\n"
            "best: 选择指标最优(最小)的那个样本代表；\n"
            "mean: 取该特征值下样本指标的均值；"
        ),
    )
    return parser.parse_args()


EVAL_COLS_DEFAULT = ["mse", "mae", "rmse", "mape", "r2"]


def make_model_order_column(df: pd.DataFrame) -> pd.DataFrame:
    # 以CSV中的原始顺序定义1-based的模型序号，用于输出引用
    df = df.reset_index(drop=True).copy()
    df["model_index"] = df.index + 1
    return df


def get_feature_columns(df: pd.DataFrame, eval_cols: List[str], admin_cols: List[str]) -> List[str]:
    columns = [c for c in df.columns if c not in set(eval_cols) | set(admin_cols) | {"model_index"}]
    return columns


def control_groups_ranking(
    df: pd.DataFrame,
    feature: str,
    all_feature_cols: List[str],
    metric: str,
    min_group_size: int,
    aggregate_duplicates: str,
) -> List[Dict[str, object]]:
    other_cols = [c for c in all_feature_cols if c != feature]

    # 分组键：除当前特征之外的所有特征列
    if not other_cols:
        group_keys = None
    else:
        group_keys = other_cols

    results: List[Dict[str, object]] = []

    if group_keys is None:
        grouped = [((), df)]
    else:
        grouped = df.groupby(group_keys, dropna=False, as_index=False)

    # 为每个控制组进行排名
    if group_keys is None:
        iter_groups = grouped
    else:
        iter_groups = grouped.__iter__()

    # 收集所有有效控制组（仅当前特征不同）
    candidate_groups: List[Dict[str, object]] = []

    for _, group_df in iter_groups:
        # 过滤无效/缺失的当前特征或指标
        g = group_df.dropna(subset=[feature, metric]).copy()
        if g.empty:
            continue

        # 额外稳健性校验：确保除当前特征外的其他列在本组内都恒定（控制变量原则）
        is_valid_control = True
        for col in other_cols:
            if g[col].nunique(dropna=False) != 1:
                is_valid_control = False
                break
        if not is_valid_control:
            continue

        # 对同一特征值可能存在多条：按策略聚合到代表项
        representatives: List[Tuple[object, int, float]] = []  # (特征值, model_index, metric_value)
        for val, val_df in g.groupby(feature, dropna=False):
            if val_df.empty:
                continue
            if aggregate_duplicates == "mean":
                # 平均指标，选择一个代表model_index（用指标最优的行代表索引）
                mean_metric = float(val_df[metric].mean())
                best_row = val_df.iloc[val_df[metric].astype(float).values.argmin()]
                representatives.append((val, int(best_row["model_index"]), mean_metric))
            else:  # best
                best_row = val_df.iloc[val_df[metric].astype(float).values.argmin()]
                representatives.append((val, int(best_row["model_index"]), float(best_row[metric])))

        # 需要至少min_group_size个不同取值才能排名
        if len(representatives) < min_group_size:
            continue

        # 保存候选组信息（暂不输出，稍后择优选一组）
        representatives.sort(key=lambda x: x[2])
        compared_indices = [repr_item[1] for repr_item in representatives]
        compared_indices_sorted = sorted(compared_indices)
        mi_to_metric = {mi: m for (_, mi, m) in representatives}
        compared_mses_in_sorted_order = [mi_to_metric[mi] for mi in compared_indices_sorted]

        candidate_groups.append(
            {
                "representatives": representatives,  # 已按指标升序
                "values_set": set(val for (val, _, __) in representatives),
                "compared_indices_sorted": compared_indices_sorted,
                "compared_mses_sorted": compared_mses_in_sorted_order,
                "mean_metric": float(sum(m for (_, __, m) in representatives) / len(representatives)),
            }
        )

    # 若无候选组，返回空
    if not candidate_groups:
        return results

    # 选择覆盖最多取值的组；若有并列，选择平均指标最优的组
    max_coverage = max(len(cg["values_set"]) for cg in candidate_groups)
    best_groups = [cg for cg in candidate_groups if len(cg["values_set"]) == max_coverage]
    best_group = min(best_groups, key=lambda x: x["mean_metric"])  # 均值越小越好

    representatives = best_group["representatives"]

    # 输出单一最佳组的排名，确保每个特征值唯一且无重复
    for rank_pos, (val, mi, mval) in enumerate(representatives, start=1):
        results.append(
            {
                "feature_name": feature,
                "feature_value": val,
                "rank": rank_pos,
                # 仅该特征值对应的代表模型序号（与该值排名一致）
                "compared_model_indices": str(mi),
                # 与该代表模型对应的指标值（默认mse）
                "compared_model_mse": float(mval),
            }
        )

    return results


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    if df.empty:
        raise SystemExit("输入CSV为空")

    df = make_model_order_column(df)

    eval_cols = [c for c in EVAL_COLS_DEFAULT if c in df.columns]
    if args.metric not in eval_cols:
        raise SystemExit(f"指标{args.metric}不在CSV列中，可选: {eval_cols}")

    admin_cols = [c.strip() for c in args.drop_admin_cols.split(",") if c.strip()]
    admin_cols = [c for c in admin_cols if c in df.columns]

    feature_cols = get_feature_columns(df, eval_cols, admin_cols)
    if not feature_cols:
        raise SystemExit("未找到可用于对比的特征列")

    # 对每个特征分别输出一个CSV
    for feature in feature_cols:
        rows = control_groups_ranking(
            df=df,
            feature=feature,
            all_feature_cols=feature_cols,
            metric=args.metric,
            min_group_size=args.min_group_size,
            aggregate_duplicates=args.aggregate_duplicates,
        )

        out_df = pd.DataFrame(rows, columns=[
            "feature_name",
            "feature_value",
            "rank",
            "compared_model_indices",
            "compared_model_mse",
        ])

        # 去重，避免重复值干扰分析
        if not out_df.empty:
            out_df = out_df.drop_duplicates(
                subset=[
                    "feature_name",
                    "feature_value",
                    "rank",
                    "compared_model_indices",
                    "compared_model_mse",
                ],
                keep="first",
            )

        # 若某特征在全表中没有形成任何有效对比组，则也输出空CSV，便于排查
        out_path = os.path.join(args.outdir, f"{feature}_ranking.csv")
        out_df.to_csv(out_path, index=False)

        # 可选：绘制柱状图
        if args.plot_dir is not None:
            os.makedirs(args.plot_dir, exist_ok=True)
            if not out_df.empty:
                # x轴：特征取值(字符串化)；y轴：对应代表样本的指标值
                x_labels = out_df.sort_values("rank")["feature_value"].astype(str).tolist()
                y_values = out_df.sort_values("rank")["compared_model_mse"].astype(float).tolist()

                plt.figure(figsize=(max(6, len(x_labels) * 0.9), 4.5))
                plt.bar(range(len(x_labels)), y_values, color="#4C78A8")
                plt.xticks(range(len(x_labels)), x_labels, rotation=30, ha="right")
                plt.ylabel(args.metric)
                plt.title(f"{feature} ranking by {args.metric}")
                for i, v in enumerate(y_values):
                    plt.text(i, v, f"{v:.4g}", ha="center", va="bottom", fontsize=9)
                plt.tight_layout()
                plot_path = os.path.join(args.plot_dir, f"{feature}_mse_bar.png")
                plt.savefig(plot_path, dpi=150)
                plt.close()

    print(f"完成。输出目录: {args.outdir}")


if __name__ == "__main__":
    main()


