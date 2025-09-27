#!/usr/bin/env python3
"""
最优集合体配置消融验证评估器
Optimal Ensemble Configuration Ablation Validation Evaluator

目的：评估最优配置 vs 消融配置的性能差异，验证每个最优组件的重要性
"""

from __future__ import annotations
import sys
import os
# 🔥 修复：添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
import yaml
import json  # 🔥 修复：添加缺失的json导入

# Set matplotlib English support and academic style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.titleweight': 'bold',
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

class OptimalEnsembleEvaluator:
    """最优集合体配置评估器"""

    def __init__(self, optimal_model_path: str, ablation_models_dir: str,
                 data_path: str, output_dir: str):
        """
        初始化评估器

        Args:
            optimal_model_path: 最优配置训练的模型路径
            ablation_models_dir: 消融配置训练的模型目录
            data_path: 测试数据路径
            output_dir: 输出目录
        """
        self.optimal_model_path = Path(optimal_model_path)
        self.ablation_models_dir = Path(ablation_models_dir)
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {}
        self.component_importance = {}

    def load_and_evaluate_models(self) -> Dict[str, Any]:
        """加载并评估所有模型"""

        print("="*80)
        print("最优集合体配置消融验证评估")
        print("="*80)

        # 1. 评估最优配置模型
        print("\n[STEP 1] 评估最优配置模型...")
        print(f"[DEBUG] 最优模型路径: {self.optimal_model_path}")
        print(f"[DEBUG] 路径是否存在: {self.optimal_model_path.exists()}")

        if not self.optimal_model_path.exists():
            print(f"[ERROR] 最优模型文件不存在: {self.optimal_model_path}")
            raise ValueError(f"最优模型文件不存在: {self.optimal_model_path}")

        optimal_result = self._evaluate_single_model(
            self.optimal_model_path,
            model_name="OptimalEnsemble"
        )

        if optimal_result is None:
            print(f"[ERROR] 最优配置模型评估返回None，请检查上述错误信息")
            print(f"[DEBUG] 尝试手动检查checkpoint文件...")
            try:
                checkpoint = torch.load(str(self.optimal_model_path), map_location='cpu')
                print(f"[DEBUG] Checkpoint文件可以加载，键: {list(checkpoint.keys())}")
                if 'cfg' in checkpoint:
                    cfg = checkpoint['cfg']
                    print(f"[DEBUG] 配置文件可用，键: {list(cfg.keys())}")
                else:
                    print(f"[ERROR] Checkpoint中没有'cfg'键")
            except Exception as e:
                print(f"[ERROR] 无法加载checkpoint文件: {e}")

            raise ValueError("无法评估最优配置模型")

        self.results['optimal'] = optimal_result
        print(f"[INFO] 最优配置性能: MSE={optimal_result['mse']:.6f}, R²={optimal_result['r2']:.4f}")

        # 2. 评估所有消融配置模型
        print("\n[STEP 2] 评估消融配置模型...")
        ablation_model_files = list(self.ablation_models_dir.glob("*.pt"))

        if not ablation_model_files:
            print(f"[WARNING] 未找到消融模型文件在: {self.ablation_models_dir}")
            ablation_model_files = []

        ablation_results = {}
        for model_file in ablation_model_files:
            model_name = model_file.stem
            print(f"[INFO] 评估消融模型: {model_name}")

            result = self._evaluate_single_model(model_file, model_name)
            if result is not None:
                ablation_results[model_name] = result

        self.results['ablations'] = ablation_results
        print(f"[INFO] 成功评估 {len(ablation_results)} 个消融模型")

        # 3. 计算组件重要性
        print("\n[STEP 3] 分析组件重要性...")
        self.component_importance = self._analyze_component_importance()

        return {
            'optimal': optimal_result,
            'ablations': ablation_results,
            'component_importance': self.component_importance
        }

    def _evaluate_single_model(self, model_path: Path, model_name: str) -> Optional[Dict[str, Any]]:
        """🔥 复用batch_well_log_eval.py中的评估逻辑"""

        try:
            print(f"[DEBUG] 开始评估模型: {model_path}")

            # 🔥 关键修复：直接复用batch_well_log_eval.py的评估器，并正确初始化
            from scripts.batch_well_log_eval import WellLogBatchEvaluator

            # 创建临时的批量评估器
            temp_evaluator = WellLogBatchEvaluator(
                checkpoints_dir=model_path.parent,  # 模型所在目录
                data_path=str(self.data_path),
                output_dir="./temp_eval",
                max_models=1
            )

            # 🔥 重要：确保评估器先加载数据
            temp_evaluator.load_test_data()
            print(f"[DEBUG] batch_well_log_eval评估器数据初始化完成")

            # 直接调用其评估方法
            result = temp_evaluator.evaluate_single_model(model_path)

            if result is None:
                print(f"[ERROR] batch_well_log_eval评估返回None")
                return None

            # 转换为我们需要的格式
            metrics = result.get('metrics', {})
            converted_result = {
                'model_name': model_name,
                'model_path': str(model_path),
                'mse': metrics.get('mse', metrics.get('test_mse', 0)),
                'mae': metrics.get('mae', metrics.get('test_mae', 0)),
                'r2': metrics.get('r2', metrics.get('test_r2', 0)),
                'rmse': metrics.get('rmse', metrics.get('test_rmse', 0)),
                'config_info': self._extract_config_from_result(result),
                'evaluation_success': True,
                'original_result': result  # 保存原始结果
            }

            print(f"[SUCCESS] 模型 {model_name} 评估完成: MSE={converted_result['mse']:.6f}, R²={converted_result['r2']:.4f}")
            return converted_result

        except Exception as e:
            print(f"[ERROR] 评估模型 {model_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_config_from_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """从batch_well_log_eval结果中提取配置信息"""

        try:
            # 从ablation_factors或filename_architecture提取配置
            ablation_factors = result.get('ablation_factors', {})
            if ablation_factors:
                return {
                    'cnn_variant': ablation_factors.get('cnn_variant', 'standard'),
                    'rnn_type': ablation_factors.get('rnn_type', 'lstm'),
                    'attn_variant': ablation_factors.get('attn_variant', 'standard'),
                    'channel_attention': ablation_factors.get('channel_attention', 'off'),
                    'pos_encoding': ablation_factors.get('pos_encoding', 'none'),
                    'use_revin': ablation_factors.get('use_revin', False),
                    'use_decomposition': ablation_factors.get('use_decomposition', False),
                    'wavelet': ablation_factors.get('wavelet', False),
                    'wavelet_base': ablation_factors.get('wavelet_base', 'db4'),
                    'wavelet_level': ablation_factors.get('wavelet_level', 3),
                }

            # 回退到默认配置
            return {
                'cnn_variant': 'standard',
                'rnn_type': 'lstm',
                'attn_variant': 'standard',
                'channel_attention': 'off',
                'pos_encoding': 'none',
                'use_revin': False,
                'use_decomposition': False,
                'wavelet': False,
                'wavelet_base': 'db4',
                'wavelet_level': 3,
            }

        except Exception as e:
            print(f"[WARNING] 配置提取失败: {e}")
            return {}

    def _extract_config_components(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """从配置中提取组件信息（简化版本）"""

        model_config = cfg.get('model', {})
        data_config = cfg.get('data', {})
        train_config = cfg.get('train', {})

        return {
            'cnn_variant': model_config.get('cnn', {}).get('variant', 'standard'),
            'rnn_type': model_config.get('lstm', {}).get('rnn_type', 'lstm'),
            'attn_variant': model_config.get('attention', {}).get('variant', 'standard'),
            'channel_attention': (
                model_config.get('cnn', {}).get('channel_attention_type', 'off')
                if model_config.get('cnn', {}).get('use_channel_attention', False)
                else 'off'
            ),
            'pos_encoding': model_config.get('attention', {}).get('positional_mode', 'none'),
            'use_revin': model_config.get('normalization', {}).get('revin', {}).get('enabled', False),
            'use_decomposition': model_config.get('decomposition', {}).get('enabled', False),
            'wavelet': data_config.get('wavelet', {}).get('enabled', False),
            'wavelet_base': data_config.get('wavelet', {}).get('wavelet', 'db4'),
            'wavelet_level': data_config.get('wavelet', {}).get('level', 3),
        }

    def _analyze_component_importance(self) -> Dict[str, Any]:
        """Analyze component importance"""

        optimal_mse = self.results['optimal']['mse']
        component_impacts = {}

        print("\n[INFO] Component Importance Analysis:")
        print("-" * 60)

        for model_name, result in self.results['ablations'].items():
            # Extract ablated component from model name
            if 'ablation_' in model_name and '_optimal_to_baseline' in model_name:
                component = model_name.replace('ablation_', '').replace('_optimal_to_baseline', '')

                # Calculate performance drop
                ablation_mse = result['mse']
                performance_drop = (ablation_mse - optimal_mse) / optimal_mse * 100

                component_impacts[component] = {
                    'performance_drop_pct': performance_drop,
                    'optimal_mse': optimal_mse,
                    'ablation_mse': ablation_mse,
                    'importance_score': max(0, performance_drop),  # Higher drop = more important
                    'optimal_config': result['config_info'].get(component, 'unknown'),
                    'ablation_config': 'baseline'
                }

                print(f"{component:<20}: Performance drop {performance_drop:+.1f}% "
                      f"(MSE: {optimal_mse:.6f} → {ablation_mse:.6f})")

        # Sort by importance
        sorted_importance = sorted(
            component_impacts.items(),
            key=lambda x: x[1]['importance_score'],
            reverse=True
        )

        return {
            'component_impacts': component_impacts,
            'importance_ranking': sorted_importance,
            'total_components': len(component_impacts),
            'most_important': sorted_importance[0] if sorted_importance else None,
            'least_important': sorted_importance[-1] if sorted_importance else None
        }

    def generate_comparison_report(self) -> str:
        """Generate comparison analysis report"""

        report_lines = []
        report_lines.append("# Optimal Ensemble Configuration Ablation Validation Report")
        report_lines.append("=" * 60)
        report_lines.append("")

        # 1. Experiment Overview
        report_lines.append("## Experiment Overview")
        report_lines.append(f"- Optimal Configuration Model: {self.results['optimal']['model_name']}")
        report_lines.append(f"- Number of Ablation Configurations: {len(self.results['ablations'])}")
        report_lines.append(f"- Evaluation Metrics: MSE, MAE, R², RMSE")
        report_lines.append("")

        # 2. Optimal Configuration Performance
        optimal = self.results['optimal']
        report_lines.append("## Optimal Configuration Performance")
        report_lines.append(f"- MSE: {optimal['mse']:.6f}")
        report_lines.append(f"- MAE: {optimal['mae']:.6f}")
        report_lines.append(f"- R²: {optimal['r2']:.4f}")
        report_lines.append(f"- RMSE: {optimal['rmse']:.6f}")
        report_lines.append("")

        # 3. Component Importance Analysis
        report_lines.append("## Component Importance Ranking")
        report_lines.append("| Rank | Component | Performance Drop(%) | Importance Rating |")
        report_lines.append("|------|-----------|-------------------|------------------|")

        for i, (component, impact) in enumerate(self.component_importance['importance_ranking'], 1):
            drop = impact['performance_drop_pct']
            if drop > 20:
                rating = "Critical"
            elif drop > 10:
                rating = "Very Important"
            elif drop > 5:
                rating = "Important"
            elif drop > 0:
                rating = "Useful"
            else:
                rating = "No Impact"

            report_lines.append(f"| {i} | {component} | {drop:+.1f}% | {rating} |")

        report_lines.append("")

        # 4. Detailed Performance Comparison
        report_lines.append("## Detailed Performance Comparison")
        report_lines.append("| Configuration | MSE | MAE | R² | RMSE | vs Optimal(%) |")
        report_lines.append("|---------------|-----|-----|----|------|---------------|")

        # Optimal configuration row
        report_lines.append(f"| Optimal Config | {optimal['mse']:.6f} | {optimal['mae']:.6f} | "
                           f"{optimal['r2']:.4f} | {optimal['rmse']:.6f} | Baseline |")

        # Ablation configuration rows
        for model_name, result in self.results['ablations'].items():
            component = model_name.replace('ablation_', '').replace('_optimal_to_baseline', '')
            change_pct = (result['mse'] - optimal['mse']) / optimal['mse'] * 100

            report_lines.append(f"| Ablation {component} | {result['mse']:.6f} | {result['mae']:.6f} | "
                               f"{result['r2']:.4f} | {result['rmse']:.6f} | {change_pct:+.1f}% |")

        report_lines.append("")

        # 5. Conclusions and Recommendations
        report_lines.append("## Conclusions and Recommendations")

        if self.component_importance['most_important']:
            most_imp = self.component_importance['most_important']
            report_lines.append(f"- **Most Important Component**: {most_imp[0]} (removal causes {most_imp[1]['performance_drop_pct']:.1f}% performance drop)")

        if self.component_importance['least_important']:
            least_imp = self.component_importance['least_important']
            if least_imp[1]['performance_drop_pct'] < 1:
                report_lines.append(f"- **Potentially Over-optimized Component**: {least_imp[0]} (removal has minimal impact: {least_imp[1]['performance_drop_pct']:.1f}%)")

        # Calculate overall effectiveness
        total_components = len(self.component_importance['component_impacts'])
        effective_components = len([
            impact for impact in self.component_importance['component_impacts'].values()
            if impact['performance_drop_pct'] > 1
        ])

        effectiveness = (effective_components / total_components * 100) if total_components > 0 else 0
        report_lines.append(f"- **Configuration Effectiveness**: {effective_components}/{total_components} components effective ({effectiveness:.1f}%)")

        report_content = "\n".join(report_lines)

        # Save report
        report_path = self.output_dir / 'ablation_validation_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"[INFO] Ablation validation report saved: {report_path}")
        return str(report_path)

    def create_comparison_visualizations(self) -> List[str]:
        """Create comparison visualization charts"""

        saved_plots = []

        # 1. Component importance bar chart
        plt.figure(figsize=(12, 8))

        components = []
        importance_scores = []
        colors = []

        for component, impact in self.component_importance['importance_ranking']:
            components.append(component.replace('_', '\n'))  # Line break for display
            importance_scores.append(impact['importance_score'])

            # Set colors based on importance
            if impact['importance_score'] > 20:
                colors.append('#d62728')  # Red - Critical
            elif impact['importance_score'] > 10:
                colors.append('#ff7f0e')  # Orange - Very Important
            elif impact['importance_score'] > 5:
                colors.append('#2ca02c')  # Green - Important
            else:
                colors.append('#1f77b4')  # Blue - Normal

        bars = plt.bar(components, importance_scores, color=colors, alpha=0.8, edgecolor='black')
        plt.title('Optimal Ensemble Configuration Component Importance Analysis', fontweight='bold', fontsize=14)
        plt.xlabel('Configuration Components', fontweight='bold')
        plt.ylabel('Performance Degradation (%)', fontweight='bold')
        plt.xticks(rotation=45, ha='right')

        # Add value labels
        for bar, score in zip(bars, importance_scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')

        # Add importance threshold lines
        plt.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Importance Threshold(10%)')
        plt.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Effectiveness Threshold(5%)')
        plt.legend()

        plt.tight_layout()
        importance_plot = self.output_dir / 'component_importance_analysis.png'
        plt.savefig(importance_plot)
        plt.close()
        saved_plots.append(str(importance_plot))
        print(f"[INFO] Component importance chart: {importance_plot.name}")

        # 2. Performance comparison radar chart
        if len(self.results['ablations']) > 0:
            self._create_performance_radar_chart(saved_plots)

        # 3. Ablation effect heatmap
        self._create_ablation_heatmap(saved_plots)

        return saved_plots

    def _create_performance_radar_chart(self, saved_plots: List[str]) -> None:
        """创建性能对比雷达图"""

        # 准备数据
        models = ['最优配置'] + list(self.results['ablations'].keys())
        metrics = ['MSE', 'MAE', 'R²', 'RMSE']

        # 归一化数据用于雷达图
        data_matrix = []

        # 最优配置数据
        optimal = self.results['optimal']
        optimal_data = [optimal['mse'], optimal['mae'], optimal['r2'], optimal['rmse']]
        data_matrix.append(optimal_data)

        # 消融配置数据
        for result in self.results['ablations'].values():
            ablation_data = [result['mse'], result['mae'], result['r2'], result['rmse']]
            data_matrix.append(ablation_data)

        # 归一化到0-1范围（MSE和MAE反转，R²保持原样）
        data_matrix = np.array(data_matrix)
        normalized_data = np.zeros_like(data_matrix)

        for i, metric in enumerate(metrics):
            if metric in ['MSE', 'MAE', 'RMSE']:  # 越小越好
                min_val, max_val = data_matrix[:, i].min(), data_matrix[:, i].max()
                normalized_data[:, i] = 1 - (data_matrix[:, i] - min_val) / (max_val - min_val + 1e-8)
            else:  # R² 越大越好
                min_val, max_val = data_matrix[:, i].min(), data_matrix[:, i].max()
                normalized_data[:, i] = (data_matrix[:, i] - min_val) / (max_val - min_val + 1e-8)

        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Draw optimal configuration
        optimal_values = normalized_data[0].tolist() + [normalized_data[0][0]]
        ax.plot(angles, optimal_values, 'o-', linewidth=3, label='Optimal Configuration', color='red')
        ax.fill(angles, optimal_values, alpha=0.25, color='red')

        # Draw ablation configurations (select top 5)
        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        for i, (model_name, color) in enumerate(zip(list(self.results['ablations'].keys())[:5], colors)):
            values = normalized_data[i+1].tolist() + [normalized_data[i+1][0]]
            component = model_name.replace('ablation_', '').replace('_optimal_to_baseline', '')
            ax.plot(angles, values, 'o-', linewidth=2, label=f'Ablation {component}', color=color, alpha=0.7)

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Optimal Configuration vs Ablation Configurations Performance Comparison\n(Radar Chart - Further from center is better)', fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()
        radar_plot = self.output_dir / 'performance_radar_comparison.png'
        plt.savefig(radar_plot)
        plt.close()
        saved_plots.append(str(radar_plot))
        print(f"[INFO] 性能雷达图: {radar_plot.name}")

    def _create_ablation_heatmap(self, saved_plots: List[str]) -> None:
        """创建消融效应热力图"""

        # 准备热力图数据
        components = []
        metrics = ['MSE change(%)', 'MAE change(%)', 'R²变化(%)', 'RMSE change(%)']
        heatmap_data = []

        optimal = self.results['optimal']

        for model_name, result in self.results['ablations'].items():
            component = model_name.replace('ablation_', '').replace('_optimal_to_baseline', '')
            components.append(component)

            # 计算各指标的变化百分比
            mse_change = (result['mse'] - optimal['mse']) / optimal['mse'] * 100
            mae_change = (result['mae'] - optimal['mae']) / optimal['mae'] * 100
            r2_change = (result['r2'] - optimal['r2']) / abs(optimal['r2']) * 100 if optimal['r2'] != 0 else 0
            rmse_change = (result['rmse'] - optimal['rmse']) / optimal['rmse'] * 100

            heatmap_data.append([mse_change, mae_change, r2_change, rmse_change])

        if len(heatmap_data) > 0:
            heatmap_df = pd.DataFrame(heatmap_data, index=components, columns=metrics)

            plt.figure(figsize=(10, 8))
            sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='RdYlBu_r',
                       center=0, cbar_kws={'label': 'Performance Change (%)'})
            plt.title('Ablation Experiment Performance Change Heatmap\n(Red=Performance Drop, Blue=Performance Gain)', fontweight='bold')
            plt.xlabel('Evaluation Metrics', fontweight='bold')
            plt.ylabel('Ablated Components', fontweight='bold')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)

            plt.tight_layout()
            heatmap_plot = self.output_dir / 'ablation_effect_heatmap.png'
            plt.savefig(heatmap_plot)
            plt.close()
            saved_plots.append(str(heatmap_plot))
            print(f"[INFO] 消融热力图: {heatmap_plot.name}")

    def save_results(self) -> Dict[str, str]:
        """保存所有结果"""

        # 保存详细结果数据
        results_data = {
            'optimal_performance': self.results['optimal'],
            'ablation_results': self.results['ablations'],
            'component_importance': self.component_importance,
            'summary': {
                'total_ablations': len(self.results['ablations']),
                'most_important_component': (
                    self.component_importance['most_important'][0]
                    if self.component_importance['most_important']
                    else None
                ),
                'average_performance_drop': np.mean([
                    impact['performance_drop_pct']
                    for impact in self.component_importance['component_impacts'].values()
                ]) if self.component_importance['component_impacts'] else 0
            }
        }

        results_json = self.output_dir / 'ablation_validation_results.json'
        with open(results_json, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False, default=str)

        # 生成CSV格式的对比表
        self._generate_comparison_csv()

        # 生成分析报告
        report_path = self.generate_comparison_report()

        # 生成可视化图表
        plot_paths = self.create_comparison_visualizations()

        return {
            'results_json': str(results_json),
            'comparison_csv': str(self.output_dir / 'performance_comparison.csv'),
            'report': report_path,
            'plots': plot_paths
        }

    def _generate_comparison_csv(self) -> None:
        """生成CSV格式的性能对比表"""

        comparison_data = []

        # 最优配置
        optimal = self.results['optimal']
        comparison_data.append({
            'Model_Type': 'Optimal_Ensemble',
            'Component_Ablated': 'None',
            'MSE': optimal['mse'],
            'MAE': optimal['mae'],
            'R2': optimal['r2'],
            'RMSE': optimal['rmse'],
            'MSE_Change_Pct': 0.0,
            'Performance_Rating': 'Baseline'
        })

        # 消融配置
        for model_name, result in self.results['ablations'].items():
            component = model_name.replace('ablation_', '').replace('_optimal_to_baseline', '')
            mse_change = (result['mse'] - optimal['mse']) / optimal['mse'] * 100

            comparison_data.append({
                'Model_Type': 'Ablation',
                'Component_Ablated': component,
                'MSE': result['mse'],
                'MAE': result['mae'],
                'R2': result['r2'],
                'RMSE': result['rmse'],
                'MSE_Change_Pct': mse_change,
                'Performance_Rating': self._rate_performance_change(mse_change)
            })

        # 保存CSV
        df = pd.DataFrame(comparison_data)
        csv_path = self.output_dir / 'performance_comparison.csv'
        df.to_csv(csv_path, index=False)
        print(f"[INFO] 性能对比CSV: {csv_path.name}")

    def _rate_performance_change(self, change_pct: float) -> str:
        """评级性能变化"""
        if change_pct > 20:
            return "Severe_Degradation"
        elif change_pct > 10:
            return "Significant_Degradation"
        elif change_pct > 5:
            return "Moderate_Degradation"
        elif change_pct > 1:
            return "Slight_Degradation"
        else:
            return "No_Impact"

    def run_complete_evaluation(self) -> Dict[str, Any]:
        """运行完整的评估流程"""

        # 1. 加载和评估模型
        evaluation_results = self.load_and_evaluate_models()

        # 2. 保存结果
        output_files = self.save_results()

        # 3. 打印总结
        print("\n" + "="*80)
        print("消融验证评估完成")
        print("="*80)

        if self.component_importance['most_important']:
            most_imp = self.component_importance['most_important']
            print(f"最重要组件: {most_imp[0]} (性能下降: {most_imp[1]['performance_drop_pct']:.1f}%)")

        avg_drop = np.mean([
            impact['performance_drop_pct']
            for impact in self.component_importance['component_impacts'].values()
        ]) if self.component_importance['component_impacts'] else 0

        print(f"平均性能下降: {avg_drop:.1f}%")
        print(f"生成文件: {len(output_files)} 个")

        return {
            'evaluation_results': evaluation_results,
            'output_files': output_files,
            'component_importance': self.component_importance
        }

def main():
    parser = argparse.ArgumentParser(description='最优集合体配置消融验证评估器')
    parser.add_argument('--optimal_model_path', required=True,
                       help='最优配置训练的模型路径 (.pt文件)')
    parser.add_argument('--ablation_models_dir', required=True,
                       help='消融配置训练的模型目录')
    parser.add_argument('--data_path', required=True,
                       help='测试数据路径')
    parser.add_argument('--output_dir', default='./ablation_validation_results',
                       help='输出目录')

    args = parser.parse_args()

    try:
        evaluator = OptimalEnsembleEvaluator(
            optimal_model_path=args.optimal_model_path,
            ablation_models_dir=args.ablation_models_dir,
            data_path=args.data_path,
            output_dir=args.output_dir
        )

        results = evaluator.run_complete_evaluation()

        print(f"\n[SUCCESS] 消融验证评估完成！")
        print(f"[INFO] 查看详细结果: {args.output_dir}")

    except Exception as e:
        print(f"[ERROR] 评估失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()