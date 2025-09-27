#!/usr/bin/env python3
"""
Well Log Prediction Visualization Tool
符合测井行业标准的预测结果可视化工具
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class WellLogVisualizer:
    """测井图可视化器 - 符合行业标准"""

    def __init__(self, figsize=(16, 12), dpi=150):
        """
        初始化测井图可视化器

        Args:
            figsize: 图像尺寸 (宽, 高)
            dpi: 图像分辨率
        """
        self.figsize = figsize
        self.dpi = dpi

        # 测井行业标准颜色配置 - 按用户要求修改
        self.colors = {
            'actual': '#000000',            # 黑色 - 所有实际值
            'pred': '#FF0000',              # 红色 - 所有预测值
            'grid': '#cccccc',              # 网格线颜色
            'depth_line': '#666666'         # 深度轴颜色
        }

        # 线型配置
        self.linestyles = {
            'actual': '-',      # 实线 - 实际值
            'pred': '--',       # 虚线 - 预测值
            'trend': ':'        # 点线 - 趋势线
        }

        # 线宽配置
        self.linewidths = {
            'main': 1.5,       # 主要曲线
            'secondary': 1.0,   # 次要曲线
            'grid': 0.5         # 网格线
        }

    def create_depth_formatter(self, unit: str = "m") -> FuncFormatter:
        """创建深度轴格式化器"""
        def depth_format(x, pos):
            return f"{x:.0f} {unit}"
        return FuncFormatter(depth_format)

    def plot_well_log_predictions(
        self,
        depths: np.ndarray,
        dtcrt_actual: np.ndarray,
        dtcrt_pred: np.ndarray,
        alcdlc_actual: np.ndarray,
        alcdlc_pred: np.ndarray,
        title: str = "Well Log Prediction vs Actual",
        save_path: Optional[str] = None,
        depth_unit: str = "m",
        show_grid: bool = True
    ) -> plt.Figure:
        """
        创建符合测井标准的预测对比图

        Args:
            depths: 深度数组
            dtcrt_actual: DTCRT实际值
            dtcrt_pred: DTCRT预测值
            alcdlc_actual: ALCDLC实际值
            alcdlc_pred: ALCDLC预测值
            title: 图表标题
            save_path: 保存路径
            depth_unit: 深度单位
            show_grid: 是否显示网格

        Returns:
            matplotlib.figure.Figure: 生成的图表
        """
        # 创建图表和子图布局 - 简化为两个子图
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)

        # 使用GridSpec创建专业的测井图布局 - 去掉统计面板
        gs = gridspec.GridSpec(
            nrows=1, ncols=2,
            figure=fig,
            width_ratios=[1, 1],
            hspace=0.05,
            wspace=0.15
        )

        # 子图1: DTCRT
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_single_log_track(
            ax1, depths, dtcrt_actual, dtcrt_pred,
            "DTCRT", depth_unit, show_grid
        )

        # 子图2: ALCDLC_MERGED
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_single_log_track(
            ax2, depths, alcdlc_actual, alcdlc_pred,
            "ALCDLC_MERGED", depth_unit, show_grid
        )

        # 移除中间轴的y轴标签避免重复
        ax2.set_ylabel("")

        # 设置总标题
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)

        # 添加图例
        self._add_comprehensive_legend(fig, ax2)

        plt.tight_layout()

        # 保存图表
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"测井图已保存至: {save_path}")

        return fig

    def _plot_single_log_track(
        self,
        ax: plt.Axes,
        depths: np.ndarray,
        actual: np.ndarray,
        predicted: np.ndarray,
        parameter_name: str,
        depth_unit: str,
        show_grid: bool
    ):
        """绘制单个测井轨道"""

        # 绘制实际值曲线 - 黑色实线
        ax.plot(
            actual, depths,
            color=self.colors['actual'],
            linestyle=self.linestyles['actual'],
            linewidth=self.linewidths['main'],
            label=f'Real Value',
            alpha=0.8
        )

        # 绘制预测值曲线 - 红色虚线
        ax.plot(
            predicted, depths,
            color=self.colors['pred'],
            linestyle=self.linestyles['pred'],
            linewidth=self.linewidths['main'],
            label=f'Prediction Value',
            alpha=0.8
        )

        # 设置深度轴（y轴）- 测井标准：深度从上到下增加
        ax.invert_yaxis()
        ax.set_ylabel(f'深度 ({depth_unit})', fontsize=12, fontweight='bold')
        ax.yaxis.set_major_formatter(self.create_depth_formatter(depth_unit))

        # 设置参数轴（x轴）
        ax.set_xlabel(parameter_name, fontsize=12, fontweight='bold')

        # 网格设置
        if show_grid:
            ax.grid(True, linestyle='-', alpha=0.3, color=self.colors['grid'],
                   linewidth=self.linewidths['grid'])
            ax.set_axisbelow(True)  # 网格线在数据下方

        # 轴的专业化设置
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)

        # 添加轻微的轴线样式
        for spine in ax.spines.values():
            spine.set_color(self.colors['depth_line'])
            spine.set_linewidth(0.8)

    def _add_statistics_panel(
        self,
        ax: plt.Axes,
        dtcrt_actual: np.ndarray,
        dtcrt_pred: np.ndarray,
        alcdlc_actual: np.ndarray,
        alcdlc_pred: np.ndarray
    ):
        """添加统计信息面板"""

        # 计算统计指标
        dtcrt_rmse = np.sqrt(np.mean((dtcrt_actual - dtcrt_pred) ** 2))
        dtcrt_mae = np.mean(np.abs(dtcrt_actual - dtcrt_pred))
        dtcrt_r2 = self._calculate_r2(dtcrt_actual, dtcrt_pred)

        alcdlc_rmse = np.sqrt(np.mean((alcdlc_actual - alcdlc_pred) ** 2))
        alcdlc_mae = np.mean(np.abs(alcdlc_actual - alcdlc_pred))
        alcdlc_r2 = self._calculate_r2(alcdlc_actual, alcdlc_pred)

        # 统计信息文本
        stats_text = f"""预测性能统计

DTCRT:
  RMSE: {dtcrt_rmse:.4f}
  MAE:  {dtcrt_mae:.4f}
  R²:   {dtcrt_r2:.4f}

ALCDLC_MERGED:
  RMSE: {alcdlc_rmse:.4f}
  MAE:  {alcdlc_mae:.4f}
  R²:   {alcdlc_r2:.4f}

双目标综合:
  平均RMSE: {(dtcrt_rmse + alcdlc_rmse)/2:.4f}
  平均R²:   {(dtcrt_r2 + alcdlc_r2)/2:.4f}
"""

        # 清除轴并添加文本
        ax.clear()
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    def _calculate_r2(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """计算R²系数"""
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    def _add_comprehensive_legend(self, fig: plt.Figure, ax: plt.Axes):
        """添加综合图例"""
        # 收集所有图例元素
        handles, labels = ax.get_legend_handles_labels()

        # 在图表外部添加图例
        fig.legend(handles, labels, loc='lower center', ncol=4,
                  bbox_to_anchor=(0.5, 0.02), fontsize=10)

    def _add_professional_annotations(self, fig: plt.Figure):
        """添加专业标注"""
        # 添加时间戳和工具信息
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        annotation_text = f"生成时间: {timestamp} | 工具: CNN-LSTM-Attention深度学习模型"
        fig.text(0.99, 0.01, annotation_text, fontsize=8,
                ha='right', va='bottom', alpha=0.7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    def plot_error_distribution(
        self,
        dtcrt_actual: np.ndarray,
        dtcrt_pred: np.ndarray,
        alcdlc_actual: np.ndarray,
        alcdlc_pred: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """绘制误差分布图"""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), dpi=self.dpi)

        # DTCRT误差分布
        dtcrt_errors = dtcrt_pred - dtcrt_actual
        ax1.hist(dtcrt_errors, bins=50, alpha=0.7, color=self.colors['actual'], edgecolor='black')
        ax1.set_title('DTCRT 预测误差分布')
        ax1.set_xlabel('预测误差')
        ax1.set_ylabel('频次')
        ax1.axvline(0, color='red', linestyle='--', alpha=0.7)

        # ALCDLC误差分布
        alcdlc_errors = alcdlc_pred - alcdlc_actual
        ax2.hist(alcdlc_errors, bins=50, alpha=0.7, color=self.colors['actual'], edgecolor='black')
        ax2.set_title('ALCDLC_MERGED 预测误差分布')
        ax2.set_xlabel('预测误差')
        ax2.set_ylabel('频次')
        ax2.axvline(0, color='red', linestyle='--', alpha=0.7)

        # 散点图 - DTCRT
        ax3.scatter(dtcrt_actual, dtcrt_pred, alpha=0.6, color=self.colors['actual'], s=20)
        ax3.plot([dtcrt_actual.min(), dtcrt_actual.max()],
                [dtcrt_actual.min(), dtcrt_actual.max()], 'r--', alpha=0.8)
        ax3.set_xlabel('DTCRT 实际值')
        ax3.set_ylabel('DTCRT 预测值')
        ax3.set_title('DTCRT 实际vs预测散点图')

        # 散点图 - ALCDLC
        ax4.scatter(alcdlc_actual, alcdlc_pred, alpha=0.6, color=self.colors['actual'], s=20)
        ax4.plot([alcdlc_actual.min(), alcdlc_actual.max()],
                [alcdlc_actual.min(), alcdlc_actual.max()], 'r--', alpha=0.8)
        ax4.set_xlabel('实际值')
        ax4.set_ylabel('预测值')
        ax4.set_title('ALCDLC_MERGED 实际vs预测散点图')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"误差分析图已保存至: {save_path}")

        return fig

    def export_prediction_data_with_visualization(
        self,
        depths: np.ndarray,
        dtcrt_actual: np.ndarray,
        dtcrt_pred: np.ndarray,
        alcdlc_actual: np.ndarray,
        alcdlc_pred: np.ndarray,
        save_dir: Path,
        model_name: str = "model",
        include_plots: bool = True
    ) -> Dict[str, str]:
        """
        🔥 新增：导出预测数据并可选择生成配套可视化

        集成数据导出和可视化功能，一站式导出完整的分析结果

        Args:
            depths: 深度数组
            dtcrt_actual: DTCRT实际值
            dtcrt_pred: DTCRT预测值
            alcdlc_actual: ALCDLC实际值
            alcdlc_pred: ALCDLC预测值
            save_dir: 保存目录
            model_name: 模型名称
            include_plots: 是否同时生成可视化图

        Returns:
            Dict[str, str]: 导出文件的路径信息
        """

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        # 确保数据长度一致
        min_length = min(len(depths), len(dtcrt_actual), len(dtcrt_pred),
                        len(alcdlc_actual), len(alcdlc_pred))

        depths = depths[:min_length]
        dtcrt_actual = dtcrt_actual[:min_length]
        dtcrt_pred = dtcrt_pred[:min_length]
        alcdlc_actual = alcdlc_actual[:min_length]
        alcdlc_pred = alcdlc_pred[:min_length]

        print(f"[INFO] WellLogVisualizer导出: {min_length}个样本")

        # 1. 导出标准格式的测井数据CSV
        well_log_data = pd.DataFrame({
            'DEPTH': depths,
            'DTCRT_MEASURED': dtcrt_actual,
            'DTCRT_PREDICTED': dtcrt_pred,
            'DTCRT_RESIDUAL': dtcrt_pred - dtcrt_actual,
            'ALCDLC_MEASURED': alcdlc_actual,
            'ALCDLC_PREDICTED': alcdlc_pred,
            'ALCDLC_RESIDUAL': alcdlc_pred - alcdlc_actual
        })

        # 添加预测质量评级
        dtcrt_error_std = np.std(dtcrt_pred - dtcrt_actual)
        alcdlc_error_std = np.std(alcdlc_pred - alcdlc_actual)

        def quality_rating(dtcrt_err, alcdlc_err, dt_std, al_std):
            """基于误差大小评定预测质量"""
            if abs(dtcrt_err) <= dt_std and abs(alcdlc_err) <= al_std:
                return 'EXCELLENT'
            elif abs(dtcrt_err) <= 2*dt_std and abs(alcdlc_err) <= 2*al_std:
                return 'GOOD'
            elif abs(dtcrt_err) <= 3*dt_std and abs(alcdlc_err) <= 3*al_std:
                return 'FAIR'
            else:
                return 'POOR'

        well_log_data['PREDICTION_QUALITY'] = [
            quality_rating(dt_err, al_err, dtcrt_error_std, alcdlc_error_std)
            for dt_err, al_err in zip(well_log_data['DTCRT_RESIDUAL'], well_log_data['ALCDLC_RESIDUAL'])
        ]

        # 保存主要数据文件
        csv_file = save_dir / f"{model_name}_well_log_predictions.csv"
        well_log_data.to_csv(csv_file, index=False, encoding='utf-8')
        exported_files['csv'] = str(csv_file)
        print(f"[SUCCESS] 测井预测数据已导出: {csv_file}")

        # 2. 导出深度分段统计Excel
        excel_file = save_dir / f"{model_name}_well_log_analysis.xlsx"

        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # 主数据表
            well_log_data.to_excel(writer, sheet_name='Well_Log_Data', index=False)

            # 深度分段统计
            n_segments = 10
            depth_segments = np.linspace(depths.min(), depths.max(), n_segments + 1)
            segment_stats = []

            for i in range(n_segments):
                mask = (depths >= depth_segments[i]) & (depths < depth_segments[i + 1])
                if np.any(mask):
                    seg_data = {
                        'Segment': f"{depth_segments[i]:.0f}-{depth_segments[i+1]:.0f}m",
                        'Sample_Count': np.sum(mask),
                        'DTCRT_RMSE': np.sqrt(np.mean((dtcrt_pred[mask] - dtcrt_actual[mask]) ** 2)),
                        'DTCRT_MAE': np.mean(np.abs(dtcrt_pred[mask] - dtcrt_actual[mask])),
                        'ALCDLC_RMSE': np.sqrt(np.mean((alcdlc_pred[mask] - alcdlc_actual[mask]) ** 2)),
                        'ALCDLC_MAE': np.mean(np.abs(alcdlc_pred[mask] - alcdlc_actual[mask])),
                        'Depth_Center': (depth_segments[i] + depth_segments[i + 1]) / 2
                    }
                    segment_stats.append(seg_data)

            segment_df = pd.DataFrame(segment_stats)
            segment_df.to_excel(writer, sheet_name='Depth_Segment_Analysis', index=False)

            # 整体统计摘要
            overall_stats = pd.DataFrame({
                'Metric': ['RMSE', 'MAE', 'R2', 'Mean_Error', 'Std_Error', 'Min_Error', 'Max_Error'],
                'DTCRT': [
                    np.sqrt(np.mean((dtcrt_pred - dtcrt_actual) ** 2)),
                    np.mean(np.abs(dtcrt_pred - dtcrt_actual)),
                    self._calculate_r2(dtcrt_actual, dtcrt_pred),
                    np.mean(dtcrt_pred - dtcrt_actual),
                    np.std(dtcrt_pred - dtcrt_actual),
                    np.min(dtcrt_pred - dtcrt_actual),
                    np.max(dtcrt_pred - dtcrt_actual)
                ],
                'ALCDLC': [
                    np.sqrt(np.mean((alcdlc_pred - alcdlc_actual) ** 2)),
                    np.mean(np.abs(alcdlc_pred - alcdlc_actual)),
                    self._calculate_r2(alcdlc_actual, alcdlc_pred),
                    np.mean(alcdlc_pred - alcdlc_actual),
                    np.std(alcdlc_pred - alcdlc_actual),
                    np.min(alcdlc_pred - alcdlc_actual),
                    np.max(alcdlc_pred - alcdlc_actual)
                ]
            })
            overall_stats.to_excel(writer, sheet_name='Overall_Statistics', index=False)

        exported_files['excel'] = str(excel_file)
        print(f"[SUCCESS] 测井分析Excel已导出: {excel_file}")

        # 3. 可选：生成配套的可视化图
        if include_plots:
            # 生成主要对比图
            main_plot = save_dir / f"{model_name}_well_log_comparison.png"
            fig1 = self.plot_well_log_predictions(
                depths=depths,
                dtcrt_actual=dtcrt_actual,
                dtcrt_pred=dtcrt_pred,
                alcdlc_actual=alcdlc_actual,
                alcdlc_pred=alcdlc_pred,
                title=f"{model_name} Well Log Prediction Results",
                save_path=main_plot
            )
            exported_files['main_plot'] = str(main_plot)

            # 生成误差分析图
            error_plot = save_dir / f"{model_name}_error_analysis.png"
            fig2 = self.plot_error_distribution(
                dtcrt_actual=dtcrt_actual,
                dtcrt_pred=dtcrt_pred,
                alcdlc_actual=alcdlc_actual,
                alcdlc_pred=alcdlc_pred,
                save_path=error_plot
            )
            exported_files['error_plot'] = str(error_plot)

            # 生成深度分段性能图
            segment_plot = save_dir / f"{model_name}_depth_segment_performance.png"
            fig3 = self.plot_depth_segment_performance(segment_df, save_path=segment_plot)
            exported_files['segment_plot'] = str(segment_plot)

            plt.close('all')  # 清理图形内存

        # 4. 生成数据导出摘要
        summary_file = save_dir / f"{model_name}_export_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"{model_name} 预测数据导出摘要\n")
            f.write("=" * 60 + "\n")
            f.write(f"导出时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据样本数: {min_length}\n")
            f.write(f"深度范围: {depths.min():.2f} - {depths.max():.2f} m\n\n")

            f.write("导出文件列表:\n")
            for file_type, file_path in exported_files.items():
                f.write(f"- {file_type}: {Path(file_path).name}\n")

            f.write(f"\n预测质量分布:\n")
            quality_counts = well_log_data['PREDICTION_QUALITY'].value_counts()
            for quality, count in quality_counts.items():
                percentage = count / min_length * 100
                f.write(f"- {quality}: {count} samples ({percentage:.1f}%)\n")

        exported_files['summary'] = str(summary_file)
        print(f"[SUCCESS] 导出摘要已保存: {summary_file}")

        return exported_files

    def plot_depth_segment_performance(
        self,
        segment_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """🔥 新增：绘制深度分段性能分析图"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), dpi=self.dpi)

        # 深度vs RMSE图
        ax1.plot(segment_df['Depth_Center'], segment_df['DTCRT_RMSE'],
                 marker='o', color='blue', linewidth=2, label='DTCRT RMSE')
        ax1.plot(segment_df['Depth_Center'], segment_df['ALCDLC_RMSE'],
                 marker='s', color='red', linewidth=2, label='ALCDLC RMSE')
        ax1.set_xlabel('深度中心 (m)')
        ax1.set_ylabel('RMSE')
        ax1.set_title('深度分段RMSE性能')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 深度vs MAE图
        ax2.plot(segment_df['Depth_Center'], segment_df['DTCRT_MAE'],
                 marker='o', color='blue', linewidth=2, label='DTCRT MAE')
        ax2.plot(segment_df['Depth_Center'], segment_df['ALCDLC_MAE'],
                 marker='s', color='red', linewidth=2, label='ALCDLC MAE')
        ax2.set_xlabel('深度中心 (m)')
        ax2.set_ylabel('MAE')
        ax2.set_title('深度分段MAE性能')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"深度分段性能图已保存: {save_path}")

        return fig

def demo_visualization():
    """演示测井图可视化功能"""
    # 生成示例数据
    np.random.seed(42)
    n_samples = 1000
    depths = np.linspace(1000, 2000, n_samples)  # 深度范围: 1000-2000米

    # 模拟DTCRT数据
    dtcrt_actual = 0.1 + 0.05 * np.sin(depths / 100) + 0.02 * np.random.randn(n_samples)
    dtcrt_pred = dtcrt_actual + 0.01 * np.random.randn(n_samples)

    # 模拟ALCDLC数据
    alcdlc_actual = 0.2 + 0.08 * np.cos(depths / 150) + 0.03 * np.random.randn(n_samples)
    alcdlc_pred = alcdlc_actual + 0.015 * np.random.randn(n_samples)

    # 创建可视化器
    visualizer = WellLogVisualizer()

    # 生成主要的测井对比图
    fig1 = visualizer.plot_well_log_predictions(
        depths=depths,
        dtcrt_actual=dtcrt_actual,
        dtcrt_pred=dtcrt_pred,
        alcdlc_actual=alcdlc_actual,
        alcdlc_pred=alcdlc_pred,
        title="模型 Well Log预测结果",
        save_path="well_log_predictions.png"
    )

    # 生成误差分析图
    fig2 = visualizer.plot_error_distribution(
        dtcrt_actual=dtcrt_actual,
        dtcrt_pred=dtcrt_pred,
        alcdlc_actual=alcdlc_actual,
        alcdlc_pred=alcdlc_pred,
        save_path="prediction_error_analysis.png"
    )

    plt.show()

if __name__ == "__main__":
    demo_visualization()