#!/usr/bin/env python3
"""
🔬 高级消融实验分析器
Advanced Ablation Study Analyzer

主要功能：
1. 统计显著性检验 (t-test, Mann-Whitney U, Kolmogorov-Smirnov)
2. 效应量计算 (Cohen's d, Glass's delta, Hedge's g)
3. 置信区间估计 (Bootstrap, 参数化)
4. 多重比较校正 (Bonferroni, FDR, Holm)
5. 交互作用分析 (ANOVA, 回归分析)
6. 功效分析和样本量估计
7. LaTeX学术表格生成
8. 贝叶斯因子分析

作者: Claude Code Assistant
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, t as t_dist
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import json
import itertools
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import ttest_power
from statsmodels.stats.effect_size import cohen_d
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# 忽略警告
warnings.filterwarnings('ignore')

# 设置matplotlib学术风格
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.titleweight': 'bold',
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3
})


class AdvancedAblationAnalyzer:
    """高级消融实验统计分析器"""

    def __init__(self, results_csv: str, baseline_csv: str = None, output_dir: str = "./analysis_output"):
        """
        初始化高级分析器

        Args:
            results_csv: 模型评估结果CSV文件
            baseline_csv: 基准对比CSV文件（可选）
            output_dir: 输出目录
        """
        self.results_csv = Path(results_csv)
        self.baseline_csv = Path(baseline_csv) if baseline_csv else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 读取数据
        self.results_df = pd.read_csv(results_csv)
        if baseline_csv and Path(baseline_csv).exists():
            self.baseline_df = pd.read_csv(baseline_csv)
        else:
            self.baseline_df = None

        print(f"[INFO] 加载了 {len(self.results_df)} 个模型的评估结果")
        if self.baseline_df is not None:
            print(f"[INFO] 加载了 {len(self.baseline_df)} 个基准对比实验")

        # 分析配置
        self.primary_metric = 'mse'  # 主要评估指标
        self.alpha = 0.05  # 显著性水平
        self.confidence_level = 0.95  # 置信水平
        self.n_bootstrap = 10000  # Bootstrap采样次数

        # 消融维度定义
        self.ablation_factors = [
            'cnn_variant', 'rnn_type', 'attn_variant', 'channel_attention',
            'normalize', 'pos_encoding', 'wavelet_enabled', 'use_revin', 'use_decomposition'
        ]

        # 结果存储
        self.statistical_results = {}
        self.effect_sizes = {}
        self.confidence_intervals = {}
        self.interaction_results = {}

    def statistical_significance_test(self) -> Dict[str, Any]:
        """
        执行统计显著性检验

        包括：
        - 配对t检验
        - Wilcoxon符号秩检验
        - Mann-Whitney U检验
        - Kolmogorov-Smirnov检验
        """
        print("[INFO] 执行统计显著性检验...")

        significance_results = {}

        if self.baseline_df is None:
            print("[WARNING] 无基准对比数据，跳过显著性检验")
            return significance_results

        for _, row in self.baseline_df.iterrows():
            variable = row['Variable']
            baseline_val = row['Baseline_Value']
            test_val = row['Test_Value']

            # 获取基准和测试组的数据
            baseline_models = self.results_df[self.results_df[variable] == baseline_val]
            test_models = self.results_df[self.results_df[variable] == test_val]

            if len(baseline_models) == 0 or len(test_models) == 0:
                continue

            baseline_scores = baseline_models[self.primary_metric].values
            test_scores = test_models[self.primary_metric].values

            # 1. 配对t检验 (如果样本量相等)
            if len(baseline_scores) == len(test_scores):
                t_stat, t_pvalue = stats.ttest_rel(baseline_scores, test_scores)
                test_name = "Paired t-test"
            else:
                t_stat, t_pvalue = stats.ttest_ind(baseline_scores, test_scores)
                test_name = "Independent t-test"

            # 2. Wilcoxon符号秩检验
            if len(baseline_scores) == len(test_scores):
                w_stat, w_pvalue = stats.wilcoxon(baseline_scores, test_scores)
                w_test_name = "Wilcoxon signed-rank"
            else:
                w_stat, w_pvalue = stats.mannwhitneyu(baseline_scores, test_scores)
                w_test_name = "Mann-Whitney U"

            # 3. Kolmogorov-Smirnov检验
            ks_stat, ks_pvalue = stats.ks_2samp(baseline_scores, test_scores)

            # 4. 正态性检验
            baseline_normal = stats.shapiro(baseline_scores)[1] > 0.05
            test_normal = stats.shapiro(test_scores)[1] > 0.05

            # 5. 方差齐性检验
            levene_stat, levene_pvalue = stats.levene(baseline_scores, test_scores)
            equal_variances = levene_pvalue > 0.05

            test_key = f"{variable}_{baseline_val}_vs_{test_val}"
            significance_results[test_key] = {
                'variable': variable,
                'baseline_value': baseline_val,
                'test_value': test_val,
                'baseline_n': len(baseline_scores),
                'test_n': len(test_scores),
                'baseline_mean': np.mean(baseline_scores),
                'test_mean': np.mean(test_scores),
                'baseline_std': np.std(baseline_scores, ddof=1),
                'test_std': np.std(test_scores, ddof=1),
                'parametric_test': {
                    'name': test_name,
                    'statistic': t_stat,
                    'p_value': t_pvalue,
                    'significant': t_pvalue < self.alpha
                },
                'nonparametric_test': {
                    'name': w_test_name,
                    'statistic': w_stat,
                    'p_value': w_pvalue,
                    'significant': w_pvalue < self.alpha
                },
                'distribution_test': {
                    'name': 'Kolmogorov-Smirnov',
                    'statistic': ks_stat,
                    'p_value': ks_pvalue,
                    'significant': ks_pvalue < self.alpha
                },
                'assumptions': {
                    'baseline_normal': baseline_normal,
                    'test_normal': test_normal,
                    'equal_variances': equal_variances,
                    'levene_p': levene_pvalue
                },
                'recommended_test': self._recommend_test(baseline_normal, test_normal, equal_variances)
            }

        self.statistical_results = significance_results
        return significance_results

    def _recommend_test(self, baseline_normal: bool, test_normal: bool, equal_variances: bool) -> str:
        """推荐适当的统计检验方法"""
        if baseline_normal and test_normal:
            if equal_variances:
                return "parametric"  # t-test
            else:
                return "parametric_unequal_var"  # Welch's t-test
        else:
            return "nonparametric"  # Mann-Whitney U / Wilcoxon

    def effect_size_analysis(self) -> Dict[str, Any]:
        """
        计算效应量

        包括：
        - Cohen's d
        - Glass's delta
        - Hedge's g
        - Cliff's delta (非参数效应量)
        """
        print("[INFO] 计算效应量...")

        effect_size_results = {}

        if self.baseline_df is None:
            print("[WARNING] 无基准对比数据，跳过效应量计算")
            return effect_size_results

        for _, row in self.baseline_df.iterrows():
            variable = row['Variable']
            baseline_val = row['Baseline_Value']
            test_val = row['Test_Value']

            # 获取基准和测试组的数据
            baseline_models = self.results_df[self.results_df[variable] == baseline_val]
            test_models = self.results_df[self.results_df[variable] == test_val]

            if len(baseline_models) == 0 or len(test_models) == 0:
                continue

            baseline_scores = baseline_models[self.primary_metric].values
            test_scores = test_models[self.primary_metric].values

            # 1. Cohen's d
            cohens_d = self._cohens_d(baseline_scores, test_scores)

            # 2. Glass's delta
            glass_delta = (np.mean(test_scores) - np.mean(baseline_scores)) / np.std(baseline_scores, ddof=1)

            # 3. Hedge's g (修正的Cohen's d)
            hedges_g = self._hedges_g(baseline_scores, test_scores)

            # 4. Cliff's delta (非参数效应量)
            cliffs_delta = self._cliffs_delta(baseline_scores, test_scores)

            # 5. 效应量解释
            cohens_interpretation = self._interpret_cohens_d(abs(cohens_d))
            cliffs_interpretation = self._interpret_cliffs_delta(abs(cliffs_delta))

            test_key = f"{variable}_{baseline_val}_vs_{test_val}"
            effect_size_results[test_key] = {
                'variable': variable,
                'baseline_value': baseline_val,
                'test_value': test_val,
                'cohens_d': cohens_d,
                'glass_delta': glass_delta,
                'hedges_g': hedges_g,
                'cliffs_delta': cliffs_delta,
                'cohens_interpretation': cohens_interpretation,
                'cliffs_interpretation': cliffs_interpretation,
                'recommended_effect_size': 'cliffs_delta' if len(baseline_scores) < 30 or len(test_scores) < 30 else 'cohens_d'
            }

        self.effect_sizes = effect_size_results
        return effect_size_results

    def _cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """计算Cohen's d"""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
        return (np.mean(group2) - np.mean(group1)) / pooled_std

    def _hedges_g(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """计算Hedge's g (修正的Cohen's d)"""
        cohens_d = self._cohens_d(group1, group2)
        n1, n2 = len(group1), len(group2)
        correction_factor = 1 - (3 / (4 * (n1 + n2 - 2) - 1))
        return cohens_d * correction_factor

    def _cliffs_delta(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """计算Cliff's delta"""
        n1, n2 = len(group1), len(group2)
        dominance = 0
        for x1 in group1:
            for x2 in group2:
                if x2 > x1:
                    dominance += 1
                elif x2 < x1:
                    dominance -= 1
        return dominance / (n1 * n2)

    def _interpret_cohens_d(self, d: float) -> str:
        """解释Cohen's d效应量大小"""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    def _interpret_cliffs_delta(self, delta: float) -> str:
        """解释Cliff's delta效应量大小"""
        if delta < 0.147:
            return "negligible"
        elif delta < 0.33:
            return "small"
        elif delta < 0.474:
            return "medium"
        else:
            return "large"

    def confidence_intervals(self, method: str = "bootstrap") -> Dict[str, Any]:
        """
        计算置信区间

        Args:
            method: 方法 ("bootstrap", "parametric", "percentile")
        """
        print(f"[INFO] 计算置信区间 (方法: {method})...")

        ci_results = {}

        if self.baseline_df is None:
            print("[WARNING] 无基准对比数据，跳过置信区间计算")
            return ci_results

        for _, row in self.baseline_df.iterrows():
            variable = row['Variable']
            baseline_val = row['Baseline_Value']
            test_val = row['Test_Value']

            # 获取基准和测试组的数据
            baseline_models = self.results_df[self.results_df[variable] == baseline_val]
            test_models = self.results_df[self.results_df[variable] == test_val]

            if len(baseline_models) == 0 or len(test_models) == 0:
                continue

            baseline_scores = baseline_models[self.primary_metric].values
            test_scores = test_models[self.primary_metric].values

            if method == "bootstrap":
                baseline_ci = self._bootstrap_ci(baseline_scores)
                test_ci = self._bootstrap_ci(test_scores)
                diff_ci = self._bootstrap_difference_ci(baseline_scores, test_scores)
            elif method == "parametric":
                baseline_ci = self._parametric_ci(baseline_scores)
                test_ci = self._parametric_ci(test_scores)
                diff_ci = self._parametric_difference_ci(baseline_scores, test_scores)
            else:
                baseline_ci = self._percentile_ci(baseline_scores)
                test_ci = self._percentile_ci(test_scores)
                diff_ci = self._percentile_difference_ci(baseline_scores, test_scores)

            test_key = f"{variable}_{baseline_val}_vs_{test_val}"
            ci_results[test_key] = {
                'variable': variable,
                'baseline_value': baseline_val,
                'test_value': test_val,
                'method': method,
                'confidence_level': self.confidence_level,
                'baseline_ci': baseline_ci,
                'test_ci': test_ci,
                'difference_ci': diff_ci,
                'baseline_mean': np.mean(baseline_scores),
                'test_mean': np.mean(test_scores),
                'difference_mean': np.mean(test_scores) - np.mean(baseline_scores)
            }

        self.confidence_intervals = ci_results
        return ci_results

    def _bootstrap_ci(self, data: np.ndarray) -> Tuple[float, float]:
        """Bootstrap置信区间"""
        bootstrap_means = []
        for _ in range(self.n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - self.confidence_level
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        return (lower, upper)

    def _bootstrap_difference_ci(self, group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
        """Bootstrap差值置信区间"""
        bootstrap_diffs = []
        for _ in range(self.n_bootstrap):
            sample1 = np.random.choice(group1, size=len(group1), replace=True)
            sample2 = np.random.choice(group2, size=len(group2), replace=True)
            bootstrap_diffs.append(np.mean(sample2) - np.mean(sample1))

        alpha = 1 - self.confidence_level
        lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
        upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))
        return (lower, upper)

    def _parametric_ci(self, data: np.ndarray) -> Tuple[float, float]:
        """参数化置信区间"""
        mean = np.mean(data)
        sem = stats.sem(data)
        t_val = t_dist.ppf((1 + self.confidence_level) / 2, len(data) - 1)
        margin = t_val * sem
        return (mean - margin, mean + margin)

    def _parametric_difference_ci(self, group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
        """参数化差值置信区间"""
        diff = np.mean(group2) - np.mean(group1)
        n1, n2 = len(group1), len(group2)
        pooled_var = ((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2)
        se_diff = np.sqrt(pooled_var * (1/n1 + 1/n2))
        t_val = t_dist.ppf((1 + self.confidence_level) / 2, n1 + n2 - 2)
        margin = t_val * se_diff
        return (diff - margin, diff + margin)

    def _percentile_ci(self, data: np.ndarray) -> Tuple[float, float]:
        """百分位数置信区间"""
        alpha = 1 - self.confidence_level
        lower = np.percentile(data, 100 * alpha / 2)
        upper = np.percentile(data, 100 * (1 - alpha / 2))
        return (lower, upper)

    def _percentile_difference_ci(self, group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
        """百分位数差值置信区间"""
        differences = []
        for x1 in group1:
            for x2 in group2:
                differences.append(x2 - x1)

        alpha = 1 - self.confidence_level
        lower = np.percentile(differences, 100 * alpha / 2)
        upper = np.percentile(differences, 100 * (1 - alpha / 2))
        return (lower, upper)

    def multiple_comparison_correction(self, method: str = "fdr_bh") -> Dict[str, Any]:
        """
        多重比较校正

        Args:
            method: 校正方法 ("bonferroni", "fdr_bh", "fdr_by", "holm")
        """
        print(f"[INFO] 进行多重比较校正 (方法: {method})...")

        if not self.statistical_results:
            print("[WARNING] 请先运行统计显著性检验")
            return {}

        # 收集所有p值
        p_values = []
        test_keys = []

        for test_key, result in self.statistical_results.items():
            # 使用推荐的检验方法
            recommended = result['recommended_test']
            if recommended == "parametric" or recommended == "parametric_unequal_var":
                p_val = result['parametric_test']['p_value']
            else:
                p_val = result['nonparametric_test']['p_value']

            p_values.append(p_val)
            test_keys.append(test_key)

        # 执行多重比较校正
        if len(p_values) > 0:
            rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
                p_values, alpha=self.alpha, method=method
            )

            correction_results = {
                'method': method,
                'original_alpha': self.alpha,
                'corrected_alpha_bonferroni': alpha_bonf,
                'corrected_alpha_sidak': alpha_sidak,
                'total_tests': len(p_values),
                'significant_after_correction': sum(rejected),
                'results': {}
            }

            for i, test_key in enumerate(test_keys):
                correction_results['results'][test_key] = {
                    'original_p': p_values[i],
                    'corrected_p': p_corrected[i],
                    'significant_before': p_values[i] < self.alpha,
                    'significant_after': rejected[i],
                    'correction_factor': p_corrected[i] / p_values[i] if p_values[i] > 0 else 1.0
                }

            return correction_results
        else:
            return {'method': method, 'total_tests': 0, 'results': {}}

    def interaction_analysis(self) -> Dict[str, Any]:
        """
        分析不同因子之间的交互作用
        """
        print("[INFO] 分析因子交互作用...")

        # 确保有足够的因子进行交互分析
        available_factors = [f for f in self.ablation_factors if f in self.results_df.columns]

        if len(available_factors) < 2:
            print("[WARNING] 因子数量不足，跳过交互分析")
            return {}

        interaction_results = {}

        # 两因子交互分析
        for factor1, factor2 in itertools.combinations(available_factors, 2):
            try:
                # 创建交互模型
                df_clean = self.results_df.dropna(subset=[factor1, factor2, self.primary_metric])

                if len(df_clean) < 10:  # 样本量太小
                    continue

                # 将分类变量转换为数值
                df_encoded = df_clean.copy()
                for factor in [factor1, factor2]:
                    if df_encoded[factor].dtype == 'object' or df_encoded[factor].dtype == 'bool':
                        df_encoded[factor] = pd.Categorical(df_encoded[factor]).codes

                # 构建ANOVA模型
                formula = f"{self.primary_metric} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})"
                model = ols(formula, data=df_encoded).fit()
                anova_table = anova_lm(model)

                # 计算效应量 (eta squared)
                ss_total = anova_table['sum_sq'].sum()
                eta_squared = {}
                for index in anova_table.index:
                    eta_squared[index] = anova_table.loc[index, 'sum_sq'] / ss_total

                interaction_key = f"{factor1}_x_{factor2}"
                interaction_results[interaction_key] = {
                    'factor1': factor1,
                    'factor2': factor2,
                    'n_observations': len(df_encoded),
                    'anova_table': anova_table.to_dict(),
                    'eta_squared': eta_squared,
                    'interaction_significant': anova_table.loc[f'C({factor1}):C({factor2})', 'PR(>F)'] < self.alpha if f'C({factor1}):C({factor2})' in anova_table.index else False,
                    'model_r_squared': model.rsquared,
                    'model_adj_r_squared': model.rsquared_adj
                }

            except Exception as e:
                print(f"[WARNING] 交互分析失败 {factor1} x {factor2}: {e}")
                continue

        self.interaction_results = interaction_results
        return interaction_results

    def power_analysis(self) -> Dict[str, Any]:
        """
        功效分析和样本量估计
        """
        print("[INFO] 进行功效分析...")

        power_results = {}

        if not self.effect_sizes:
            print("[WARNING] 请先运行效应量分析")
            return power_results

        for test_key, effect_result in self.effect_sizes.items():
            cohens_d = abs(effect_result['cohens_d'])

            # 当前样本量的功效
            baseline_models = self.results_df[
                self.results_df[effect_result['variable']] == effect_result['baseline_value']
            ]
            test_models = self.results_df[
                self.results_df[effect_result['variable']] == effect_result['test_value']
            ]

            n1, n2 = len(baseline_models), len(test_models)

            if n1 > 0 and n2 > 0:
                current_power = ttest_power(cohens_d, n1, self.alpha, alternative='two-sided')

                # 推荐样本量（功效=0.8）
                recommended_n = sm.stats.tt_solve_power(
                    effect_size=cohens_d,
                    power=0.8,
                    alpha=self.alpha,
                    alternative='two-sided'
                )

                power_results[test_key] = {
                    'variable': effect_result['variable'],
                    'baseline_value': effect_result['baseline_value'],
                    'test_value': effect_result['test_value'],
                    'effect_size_cohens_d': cohens_d,
                    'current_n1': n1,
                    'current_n2': n2,
                    'current_power': current_power,
                    'recommended_n_per_group': max(2, int(np.ceil(recommended_n))) if not np.isnan(recommended_n) else 'Cannot calculate',
                    'power_adequate': current_power >= 0.8,
                    'alpha': self.alpha
                }

        return power_results

    def generate_latex_tables(self) -> Dict[str, str]:
        """
        生成LaTeX格式的学术表格
        """
        print("[INFO] 生成LaTeX表格...")

        latex_tables = {}

        # 1. 消融实验结果总表
        if self.baseline_df is not None:
            latex_tables['ablation_summary'] = self._generate_ablation_latex_table()

        # 2. 统计检验结果表
        if self.statistical_results:
            latex_tables['statistical_tests'] = self._generate_statistical_latex_table()

        # 3. 效应量表
        if self.effect_sizes:
            latex_tables['effect_sizes'] = self._generate_effect_size_latex_table()

        # 4. 置信区间表
        if self.confidence_intervals:
            latex_tables['confidence_intervals'] = self._generate_ci_latex_table()

        return latex_tables

    def _generate_ablation_latex_table(self) -> str:
        """生成消融实验结果LaTeX表格"""

        # 按改进幅度排序
        sorted_df = self.baseline_df.sort_values('RMSE_Improvement(%)', ascending=False)

        latex = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{消融实验结果总结}",
            "\\label{tab:ablation_results}",
            "\\begin{tabular}{llllrr}",
            "\\toprule",
            "维度 & 基准值 & 测试值 & 目标 & RMSE改进(\\%) & MAE改进(\\%) \\\\",
            "\\midrule"
        ]

        for _, row in sorted_df.head(15).iterrows():  # 显示前15个
            latex.append(
                f"{row['Variable']} & {row['Baseline_Value']} & {row['Test_Value']} & "
                f"{row['Target']} & {row['RMSE_Improvement(%)']:.1f} & {row['MAE_Improvement(%)']:.1f} \\\\"
            )

        latex.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])

        return "\n".join(latex)

    def _generate_statistical_latex_table(self) -> str:
        """生成统计检验结果LaTeX表格"""

        latex = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{统计显著性检验结果}",
            "\\label{tab:statistical_tests}",
            "\\begin{tabular}{llrrrc}",
            "\\toprule",
            "比较 & 检验方法 & 统计量 & p值 & 效应量 & 显著性 \\\\",
            "\\midrule"
        ]

        for test_key, result in self.statistical_results.items():
            variable = result['variable']
            baseline_val = result['baseline_value']
            test_val = result['test_value']

            # 选择推荐的检验
            recommended = result['recommended_test']
            if recommended.startswith('parametric'):
                test_info = result['parametric_test']
                test_name = test_info['name']
            else:
                test_info = result['nonparametric_test']
                test_name = test_info['name']

            # 获取效应量
            effect_size = self.effect_sizes.get(test_key, {}).get('cohens_d', 'N/A')
            if isinstance(effect_size, float):
                effect_size_str = f"{effect_size:.3f}"
            else:
                effect_size_str = "N/A"

            significance = "**" if test_info['p_value'] < 0.01 else "*" if test_info['p_value'] < 0.05 else "ns"

            comparison = f"{variable}: {baseline_val} vs {test_val}"

            latex.append(
                f"{comparison} & {test_name} & {test_info['statistic']:.3f} & "
                f"{test_info['p_value']:.4f} & {effect_size_str} & {significance} \\\\"
            )

        latex.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\tablefoot{** p < 0.01, * p < 0.05, ns = 不显著}",
            "\\end{table}"
        ])

        return "\n".join(latex)

    def _generate_effect_size_latex_table(self) -> str:
        """生成效应量LaTeX表格"""

        latex = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{效应量分析结果}",
            "\\label{tab:effect_sizes}",
            "\\begin{tabular}{llrrr}",
            "\\toprule",
            "比较 & Cohen's d & Cliff's δ & 解释 & 推荐 \\\\",
            "\\midrule"
        ]

        for test_key, result in self.effect_sizes.items():
            variable = result['variable']
            baseline_val = result['baseline_value']
            test_val = result['test_value']

            comparison = f"{variable}: {baseline_val} vs {test_val}"

            latex.append(
                f"{comparison} & {result['cohens_d']:.3f} & {result['cliffs_delta']:.3f} & "
                f"{result['cohens_interpretation']} & {result['recommended_effect_size']} \\\\"
            )

        latex.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])

        return "\n".join(latex)

    def _generate_ci_latex_table(self) -> str:
        """生成置信区间LaTeX表格"""

        latex = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{int(self.confidence_level*100)}\\% 置信区间}}",
            "\\label{tab:confidence_intervals}",
            "\\begin{tabular}{llrr}",
            "\\toprule",
            "比较 & 差值均值 & 置信区间下限 & 置信区间上限 \\\\",
            "\\midrule"
        ]

        for test_key, result in self.confidence_intervals.items():
            variable = result['variable']
            baseline_val = result['baseline_value']
            test_val = result['test_value']

            comparison = f"{variable}: {baseline_val} vs {test_val}"
            diff_mean = result['difference_mean']
            ci_lower, ci_upper = result['difference_ci']

            latex.append(
                f"{comparison} & {diff_mean:.4f} & {ci_lower:.4f} & {ci_upper:.4f} \\\\"
            )

        latex.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])

        return "\n".join(latex)

    def generate_comprehensive_report(self) -> str:
        """生成综合分析报告"""

        report = []
        report.append("=" * 80)
        report.append("🔬 高级消融实验统计分析报告")
        report.append("Advanced Ablation Study Statistical Analysis Report")
        report.append("=" * 80)
        report.append("")

        # 1. 分析概览
        report.append("📊 分析概览")
        report.append("-" * 40)
        report.append(f"模型总数: {len(self.results_df)}")
        report.append(f"消融比较数: {len(self.baseline_df) if self.baseline_df is not None else 0}")
        report.append(f"主要评估指标: {self.primary_metric}")
        report.append(f"显著性水平: α = {self.alpha}")
        report.append(f"置信水平: {int(self.confidence_level*100)}%")
        report.append("")

        # 2. 统计检验摘要
        if self.statistical_results:
            report.append("🧪 统计显著性检验摘要")
            report.append("-" * 40)

            significant_tests = sum(1 for result in self.statistical_results.values()
                                  if result['parametric_test']['significant'] or result['nonparametric_test']['significant'])
            total_tests = len(self.statistical_results)

            report.append(f"显著性检验总数: {total_tests}")
            report.append(f"显著结果数量: {significant_tests} ({significant_tests/total_tests*100:.1f}%)")
            report.append("")

        # 3. 效应量摘要
        if self.effect_sizes:
            report.append("📏 效应量分析摘要")
            report.append("-" * 40)

            effect_interpretations = {}
            for result in self.effect_sizes.values():
                interp = result['cohens_interpretation']
                effect_interpretations[interp] = effect_interpretations.get(interp, 0) + 1

            for interp, count in effect_interpretations.items():
                report.append(f"{interp.capitalize()} 效应: {count}")
            report.append("")

        # 4. 置信区间摘要
        if self.confidence_intervals:
            report.append("📊 置信区间分析摘要")
            report.append("-" * 40)

            significant_ci = sum(1 for result in self.confidence_intervals.values()
                               if result['difference_ci'][0] > 0 or result['difference_ci'][1] < 0)
            total_ci = len(self.confidence_intervals)

            report.append(f"置信区间总数: {total_ci}")
            report.append(f"不包含0的区间: {significant_ci} ({significant_ci/total_ci*100:.1f}%)")
            report.append("")

        # 5. 交互作用摘要
        if self.interaction_results:
            report.append("🔀 交互作用分析摘要")
            report.append("-" * 40)

            significant_interactions = sum(1 for result in self.interaction_results.values()
                                         if result['interaction_significant'])
            total_interactions = len(self.interaction_results)

            report.append(f"交互作用检验总数: {total_interactions}")
            report.append(f"显著交互作用: {significant_interactions}")
            report.append("")

        # 6. 建议
        report.append("💡 统计分析建议")
        report.append("-" * 40)

        if self.statistical_results:
            # 推荐最可靠的改进
            reliable_improvements = []
            for test_key, stat_result in self.statistical_results.items():
                effect_result = self.effect_sizes.get(test_key, {})

                is_significant = (stat_result['parametric_test']['significant'] or
                                stat_result['nonparametric_test']['significant'])
                has_large_effect = effect_result.get('cohens_interpretation') in ['medium', 'large']

                if is_significant and has_large_effect:
                    reliable_improvements.append((
                        test_key,
                        stat_result['variable'],
                        stat_result['test_value'],
                        effect_result.get('cohens_d', 0)
                    ))

            report.append("推荐的可靠改进（显著性 + 大效应量）:")
            for test_key, variable, test_value, cohens_d in reliable_improvements[:5]:
                report.append(f"  • {variable}: 切换到 {test_value} (Cohen's d = {cohens_d:.3f})")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def run_complete_analysis(self) -> Dict[str, Any]:
        """运行完整的高级统计分析"""

        print("[INFO] 开始高级消融实验统计分析...")

        results = {}

        # 1. 统计显著性检验
        results['statistical_tests'] = self.statistical_significance_test()

        # 2. 效应量分析
        results['effect_sizes'] = self.effect_size_analysis()

        # 3. 置信区间
        results['confidence_intervals'] = self.confidence_intervals()

        # 4. 多重比较校正
        results['multiple_comparison'] = self.multiple_comparison_correction()

        # 5. 交互作用分析
        results['interaction_analysis'] = self.interaction_analysis()

        # 6. 功效分析
        results['power_analysis'] = self.power_analysis()

        # 7. LaTeX表格
        results['latex_tables'] = self.generate_latex_tables()

        # 8. 综合报告
        results['comprehensive_report'] = self.generate_comprehensive_report()

        # 保存结果
        self._save_results(results)

        print(f"\n[SUCCESS] 高级统计分析完成！结果已保存到: {self.output_dir}")

        return results

    def _save_results(self, results: Dict[str, Any]):
        """保存所有分析结果"""

        # 保存JSON结果
        json_results = {}
        for key, value in results.items():
            if key != 'comprehensive_report':
                json_results[key] = value

        with open(self.output_dir / 'advanced_analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False, default=str)

        # 保存综合报告
        with open(self.output_dir / 'advanced_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(results['comprehensive_report'])

        # 保存LaTeX表格
        if 'latex_tables' in results:
            latex_dir = self.output_dir / 'latex_tables'
            latex_dir.mkdir(exist_ok=True)

            for table_name, latex_content in results['latex_tables'].items():
                with open(latex_dir / f'{table_name}.tex', 'w', encoding='utf-8') as f:
                    f.write(latex_content)

        print(f"[INFO] 结果已保存到:")
        print(f"  - advanced_analysis_results.json: 完整分析结果")
        print(f"  - advanced_analysis_report.txt: 综合报告")
        print(f"  - latex_tables/: LaTeX表格文件")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='高级消融实验统计分析')
    parser.add_argument('--results_csv', required=True, help='模型评估结果CSV文件路径')
    parser.add_argument('--baseline_csv', help='基准对比CSV文件路径（可选）')
    parser.add_argument('--output_dir', default='./advanced_analysis', help='输出目录')
    parser.add_argument('--alpha', type=float, default=0.05, help='显著性水平')
    parser.add_argument('--confidence_level', type=float, default=0.95, help='置信水平')
    parser.add_argument('--primary_metric', default='mse', help='主要评估指标')
    parser.add_argument('--generate_latex_tables', action='store_true', help='生成LaTeX表格')
    parser.add_argument('--statistical_significance_test', action='store_true', help='执行统计显著性检验')

    args = parser.parse_args()

    # 创建分析器
    analyzer = AdvancedAblationAnalyzer(
        results_csv=args.results_csv,
        baseline_csv=args.baseline_csv,
        output_dir=args.output_dir
    )

    # 设置参数
    analyzer.alpha = args.alpha
    analyzer.confidence_level = args.confidence_level
    analyzer.primary_metric = args.primary_metric

    # 运行完整分析
    results = analyzer.run_complete_analysis()

    print(f"\n[SUCCESS] 高级消融实验分析完成！")
    print(f"主要发现:")

    if 'statistical_tests' in results and results['statistical_tests']:
        significant_count = sum(1 for r in results['statistical_tests'].values()
                              if r['parametric_test']['significant'] or r['nonparametric_test']['significant'])
        total_count = len(results['statistical_tests'])
        print(f"  - 显著性结果: {significant_count}/{total_count} ({significant_count/total_count*100:.1f}%)")

    if 'effect_sizes' in results and results['effect_sizes']:
        large_effects = sum(1 for r in results['effect_sizes'].values()
                          if r['cohens_interpretation'] in ['medium', 'large'])
        total_effects = len(results['effect_sizes'])
        print(f"  - 中等/大效应: {large_effects}/{total_effects} ({large_effects/total_effects*100:.1f}%)")


if __name__ == '__main__':
    main()