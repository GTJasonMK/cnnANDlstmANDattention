#!/usr/bin/env python3
"""
最优集合体配置消融验证系统
Optimal Ensemble Configuration Ablation Validation System

目的：验证最优集合体配置中每个组件的实际贡献度
方法：反向消融 - 将最优组件逐一替换为baseline组件
输出：消融实验配置文件，用于验证每个最优组件的重要性
"""

from __future__ import annotations
import argparse
import yaml
import json
import copy
from pathlib import Path
from typing import Dict, Any, List, Optional

class OptimalEnsembleAblator:
    """最优集合体配置消融器"""

    def __init__(self, optimal_config_path: str, ablation_comparison_path: str, output_dir: str):
        """
        初始化消融器

        Args:
            optimal_config_path: 最优集合体配置YAML路径
            ablation_comparison_path: 消融对比CSV路径
            output_dir: 输出目录
        """
        self.optimal_config_path = Path(optimal_config_path)
        self.ablation_comparison_path = Path(ablation_comparison_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载最优配置
        self.optimal_config = self._load_optimal_config()
        self.baseline_mappings = self._extract_baseline_mappings()

        print(f"[INFO] 最优集合体配置消融验证系统初始化完成")
        print(f"  输入配置: {self.optimal_config_path}")
        print(f"  消融数据: {self.ablation_comparison_path}")
        print(f"  输出目录: {self.output_dir}")

    def _load_optimal_config(self) -> Dict[str, Any]:
        """加载最优集合体配置"""

        if not self.optimal_config_path.exists():
            raise FileNotFoundError(f"最优配置文件不存在: {self.optimal_config_path}")

        with open(self.optimal_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        print(f"[INFO] 加载最优配置成功")
        return config

    def _extract_baseline_mappings(self) -> Dict[str, Any]:
        """从消融对比数据中提取baseline映射"""

        if not self.ablation_comparison_path.exists():
            print(f"[WARNING] 消融对比文件不存在: {self.ablation_comparison_path}")
            return self._get_default_baselines()

        try:
            import pandas as pd
            df = pd.read_csv(self.ablation_comparison_path, encoding='utf-8-sig')

            # 过滤有效数据行
            data_df = df[
                (df['Variable'].notna()) &
                (df['Variable'] != '') &
                (~df['Variable'].str.contains('===', na=False)) &
                (~df['Variable'].str.contains('---', na=False))
            ].copy()

            # 提取baseline映射
            baseline_mappings = {}
            for _, row in data_df.iterrows():
                variable = row['Variable']
                baseline_value = row['Baseline_Value']
                baseline_mappings[variable] = baseline_value

            print(f"[INFO] 从消融对比数据提取baseline映射: {len(baseline_mappings)} 个")
            return baseline_mappings

        except Exception as e:
            print(f"[WARNING] 提取baseline映射失败: {e}")
            return self._get_default_baselines()

    def _get_default_baselines(self) -> Dict[str, Any]:
        """获取默认的baseline映射"""

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
            'wavelet_take': 'approx',
            'learning_rate': 0.001,
            'batch_size': 64,
            'sequence_length': 64,
            'hidden_size': 128,
            'num_layers': 2,
            'attention_heads': 4
        }

    def analyze_optimal_config(self) -> Dict[str, Any]:
        """🔥 基于真实最优配置文件分析关键组件"""

        analysis = {
            'optimal_components': {},
            'baseline_components': {},
            'differences': {}
        }

        # 🔥 基于真实文件结构提取最优配置
        model_config = self.optimal_config.get('model', {})
        data_config = self.optimal_config.get('data', {})
        train_config = self.optimal_config.get('train', {})

        # 从experiment_metadata中提取最优配置（更可靠）
        metadata = self.optimal_config.get('experiment_metadata', {})
        optimal_configs_from_metadata = metadata.get('optimal_configs', {})

        print(f"[DEBUG] 从metadata提取的最优配置: {optimal_configs_from_metadata}")

        # CNN配置
        analysis['optimal_components']['cnn_variant'] = model_config.get('cnn', {}).get('variant', 'standard')

        # 通道注意力
        cnn_config = model_config.get('cnn', {})
        if cnn_config.get('use_channel_attention', False):
            analysis['optimal_components']['channel_attention'] = cnn_config.get('channel_attention_type', 'eca')
        else:
            analysis['optimal_components']['channel_attention'] = 'off'

        # RNN配置
        analysis['optimal_components']['rnn_type'] = model_config.get('lstm', {}).get('rnn_type', 'lstm')

        # 注意力配置
        attention_config = model_config.get('attention', {})
        analysis['optimal_components']['attn_variant'] = attention_config.get('variant', 'standard')
        analysis['optimal_components']['pos_encoding'] = attention_config.get('positional_mode', 'none')

        # 预处理配置
        analysis['optimal_components']['use_revin'] = (
            model_config.get('normalization', {}).get('revin', {}).get('enabled', False)
        )
        analysis['optimal_components']['use_decomposition'] = (
            model_config.get('decomposition', {}).get('enabled', False)
        )

        # 小波配置
        wavelet_config = data_config.get('wavelet', {})
        analysis['optimal_components']['wavelet'] = wavelet_config.get('enabled', False)
        analysis['optimal_components']['wavelet_base'] = wavelet_config.get('wavelet', 'db4')
        analysis['optimal_components']['wavelet_level'] = wavelet_config.get('level', 3)
        analysis['optimal_components']['wavelet_take'] = wavelet_config.get('take', 'approx')

        print(f"\n[INFO] 提取的最优配置组件:")
        for comp, value in analysis['optimal_components'].items():
            print(f"  {comp}: {value}")

        # 🔥 对比baseline（从消融对比数据中提取的真实baseline）
        for component, optimal_value in analysis['optimal_components'].items():
            baseline_value = self.baseline_mappings.get(component, optimal_value)
            analysis['baseline_components'][component] = baseline_value

            if str(optimal_value) != str(baseline_value):
                analysis['differences'][component] = {
                    'optimal': optimal_value,
                    'baseline': baseline_value,
                    'changed': True
                }
            else:
                analysis['differences'][component] = {
                    'optimal': optimal_value,
                    'baseline': baseline_value,
                    'changed': False
                }

        changed_count = len([d for d in analysis['differences'].values() if d['changed']])
        print(f"\n[INFO] 配置差异分析:")
        print(f"  发现 {changed_count} 个非baseline组件")
        print(f"  总配置项: {len(analysis['optimal_components'])}")

        # 显示差异
        print(f"\n[INFO] 最优配置与baseline的差异:")
        for component, diff in analysis['differences'].items():
            if diff['changed']:
                print(f"  {component}: {diff['baseline']} → {diff['optimal']}")

        return analysis

    def generate_ablation_configs(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成反向消融实验配置"""

        ablation_configs = []
        changed_components = [
            comp for comp, diff in analysis['differences'].items()
            if diff['changed']
        ]

        print(f"\n[INFO] 生成 {len(changed_components)} 个反向消融配置...")

        for component in changed_components:
            # 创建消融配置：将该组件替换为baseline，其他保持最优
            ablation_config = copy.deepcopy(self.optimal_config)

            # 应用反向消融：将该组件改回baseline
            baseline_value = analysis['baseline_components'][component]
            optimal_value = analysis['optimal_components'][component]

            self._apply_component_config(ablation_config, component, baseline_value)

            # 添加实验元数据
            ablation_config['experiment_metadata'] = {
                'type': 'optimal_ensemble_ablation',
                'ablated_component': component,
                'optimal_value': optimal_value,
                'baseline_value': baseline_value,
                'note': f'验证最优{component}的贡献：{optimal_value}→{baseline_value}的性能影响'
            }

            ablation_configs.append({
                'config': ablation_config,
                'component': component,
                'optimal_value': optimal_value,
                'baseline_value': baseline_value,
                'filename': f'ablation_{component}_optimal_to_baseline.yaml'
            })

            print(f"  生成 {component} 消融配置: {optimal_value} → {baseline_value}")

        return ablation_configs

    def _apply_component_config(self, config: Dict[str, Any], component: str, value: Any) -> None:
        """🔥 基于真实配置结构应用组件配置"""

        print(f"[DEBUG] 应用组件配置: {component} = {value}")

        if component == 'cnn_variant':
            config['model']['cnn']['variant'] = str(value)
            # 🔥 重要：如果从TCN退回standard，需要禁用TCN配置
            if str(value) == 'standard' and 'tcn' in config['model']:
                config['model']['tcn']['enabled'] = False
                print(f"[DEBUG] 禁用TCN配置 (CNN变体改为standard)")

        elif component == 'rnn_type':
            config['model']['lstm']['rnn_type'] = str(value)

        elif component == 'attn_variant':
            config['model']['attention']['variant'] = str(value)

        elif component == 'channel_attention':
            if str(value) == 'off':
                config['model']['cnn']['use_channel_attention'] = False
                # 移除channel_attention_type字段
                if 'channel_attention_type' in config['model']['cnn']:
                    del config['model']['cnn']['channel_attention_type']
                print(f"[DEBUG] 禁用通道注意力")
            else:
                config['model']['cnn']['use_channel_attention'] = True
                config['model']['cnn']['channel_attention_type'] = str(value)
                print(f"[DEBUG] 启用通道注意力: {value}")

        elif component == 'pos_encoding':
            config['model']['attention']['positional_mode'] = str(value)
            # 🔥 重要：根据位置编码类型设置add_positional_encoding
            if str(value) == 'none':
                config['model']['attention']['add_positional_encoding'] = False
                print(f"[DEBUG] 禁用位置编码")
            else:
                config['model']['attention']['add_positional_encoding'] = True
                print(f"[DEBUG] 启用位置编码: {value}")

        elif component == 'use_revin':
            # 处理boolean值的不同表示
            revin_enabled = str(value).lower() in ['true', '1', 'yes', 'on', 'enabled'] or value is True
            config['model']['normalization']['revin']['enabled'] = revin_enabled
            print(f"[DEBUG] 设置RevIN: {revin_enabled}")

        elif component == 'use_decomposition':
            decomp_enabled = str(value).lower() in ['true', '1', 'yes', 'on', 'enabled'] or value is True
            config['model']['decomposition']['enabled'] = decomp_enabled
            print(f"[DEBUG] 设置分解: {decomp_enabled}")

        elif component == 'wavelet':
            wavelet_enabled = str(value).lower() in ['true', '1', 'yes', 'on', 'enabled'] or value is True
            config['data']['wavelet']['enabled'] = wavelet_enabled
            print(f"[DEBUG] 设置小波变换: {wavelet_enabled}")

        elif component == 'wavelet_base':
            # 如果改变小波基函数，确保小波变换保持启用
            config['data']['wavelet']['wavelet'] = str(value)
            print(f"[DEBUG] 设置小波基函数: {value}")

        elif component == 'wavelet_level':
            # 如果改变小波级别，确保小波变换保持启用
            try:
                config['data']['wavelet']['level'] = int(value)
                print(f"[DEBUG] 设置小波级别: {int(value)}")
            except (ValueError, TypeError):
                print(f"[ERROR] 无效的小波级别: {value}")

        elif component == 'wavelet_take':
            config['data']['wavelet']['take'] = str(value)
            print(f"[DEBUG] 设置小波系数选择: {value}")

        else:
            print(f"[WARNING] 未处理的组件: {component} = {value}")

        # 🔥 新增：验证配置修改结果
        self._validate_config_change(config, component, value)

    def _validate_config_change(self, config: Dict[str, Any], component: str, expected_value: Any) -> None:
        """验证配置修改是否正确应用"""

        # 验证修改是否成功
        try:
            if component == 'cnn_variant':
                actual_value = config['model']['cnn']['variant']
                assert str(actual_value) == str(expected_value), f"CNN变体设置失败: 期望{expected_value}, 实际{actual_value}"

            elif component == 'rnn_type':
                actual_value = config['model']['lstm']['rnn_type']
                assert str(actual_value) == str(expected_value), f"RNN类型设置失败: 期望{expected_value}, 实际{actual_value}"

            elif component == 'attn_variant':
                actual_value = config['model']['attention']['variant']
                assert str(actual_value) == str(expected_value), f"注意力变体设置失败: 期望{expected_value}, 实际{actual_value}"

            elif component == 'channel_attention':
                if str(expected_value) == 'off':
                    actual_enabled = config['model']['cnn']['use_channel_attention']
                    assert not actual_enabled, f"通道注意力禁用失败: 仍然启用"
                else:
                    actual_enabled = config['model']['cnn']['use_channel_attention']
                    actual_type = config['model']['cnn']['channel_attention_type']
                    assert actual_enabled, f"通道注意力启用失败"
                    assert str(actual_type) == str(expected_value), f"通道注意力类型设置失败: 期望{expected_value}, 实际{actual_type}"

            elif component == 'pos_encoding':
                actual_mode = config['model']['attention']['positional_mode']
                actual_enabled = config['model']['attention']['add_positional_encoding']
                expected_enabled = (str(expected_value) != 'none')

                assert str(actual_mode) == str(expected_value), f"位置编码模式设置失败: 期望{expected_value}, 实际{actual_mode}"
                assert actual_enabled == expected_enabled, f"位置编码启用状态设置失败: 期望{expected_enabled}, 实际{actual_enabled}"

            print(f"[DEBUG] [VERIFIED] {component} 配置修改成功")

        except AssertionError as e:
            print(f"[ERROR] 配置验证失败: {e}")
        except Exception as e:
            print(f"[WARNING] 配置验证异常: {e}")

    def save_ablation_configs(self, ablation_configs: List[Dict[str, Any]]) -> List[str]:
        """保存所有消融配置文件"""

        saved_files = []

        # 首先保存完整的最优配置作为对比基准
        optimal_reference_path = self.output_dir / 'optimal_ensemble_reference.yaml'
        with open(optimal_reference_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.optimal_config, f, sort_keys=False, allow_unicode=True)
        saved_files.append(str(optimal_reference_path))
        print(f"[INFO] 保存最优配置参考: {optimal_reference_path.name}")

        # 保存所有消融配置
        for ablation_info in ablation_configs:
            config = ablation_info['config']
            filename = ablation_info['filename']

            file_path = self.output_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

            saved_files.append(str(file_path))
            print(f"[INFO] 保存消融配置: {filename}")

        # 生成实验总结
        self._generate_experiment_summary(ablation_configs)

        return saved_files

    def _generate_experiment_summary(self, ablation_configs: List[Dict[str, Any]]) -> None:
        """生成实验总结文档"""

        summary = {
            'experiment_type': 'optimal_ensemble_ablation_validation',
            'total_configs': len(ablation_configs) + 1,  # +1 for optimal reference
            'optimal_config_path': str(self.optimal_config_path),
            'ablation_configs': [
                {
                    'filename': info['filename'],
                    'component': info['component'],
                    'change': f"{info['optimal_value']} → {info['baseline_value']}"
                }
                for info in ablation_configs
            ],
            'training_instructions': {
                'command_template': "python main.py --config {config_file} --data-path /your/data/path",
                'recommended_epochs': 100,
                'expected_outputs': "每个配置会生成一个最优模型到bestmodel目录"
            },
            'evaluation_instructions': {
                'script': "optimal_ensemble_evaluator.py",
                'purpose': "对比最优配置 vs 各消融配置的性能差异",
                'expected_results': "验证每个最优组件的实际贡献度"
            }
        }

        # 保存JSON格式的实验总结
        summary_path = self.output_dir / 'ablation_experiment_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 生成Markdown格式的使用说明
        readme_content = f"""# 最优集合体配置消融验证实验

## 实验目的
验证最优集合体配置中每个组件的实际贡献度，通过反向消融实验确认每个最优选择的重要性。

## 实验设计
- **基准配置**: optimal_ensemble_reference.yaml (完整的最优配置)
- **消融配置**: {len(ablation_configs)} 个配置，每个将一个最优组件替换为baseline

## 消融配置列表
"""
        for i, info in enumerate(ablation_configs, 1):
            readme_content += f"{i}. **{info['filename']}**\n"
            readme_content += f"   - 消融组件: {info['component']}\n"
            readme_content += f"   - 变化: {info['optimal_value']} → {info['baseline_value']}\n"
            readme_content += f"   - 预期: 验证{info['component']}={info['optimal_value']}的贡献度\n\n"

        readme_content += f"""
## 训练命令
```bash
# 训练最优配置（基准）
python main.py --config optimal_ensemble_reference.yaml --data-path /your/data/path

# 训练所有消融配置
"""
        for info in ablation_configs:
            readme_content += f"python main.py --config {info['filename']} --data-path /your/data/path\n"

        readme_content += f"""```

## 评估分析
训练完成后，使用以下脚本分析结果：
```bash
python optimal_ensemble_evaluator.py \\
    --optimal_model_path /path/to/optimal/model.pt \\
    --ablation_models_dir /path/to/ablation/models \\
    --data_path /your/data/path \\
    --output_dir ./evaluation_results
```

## 预期结果
- 每个消融配置的性能下降程度
- 各组件的重要性排序
- 最优集合体配置的有效性验证
- 组件协同效应分析
"""

        readme_path = self.output_dir / 'README.md'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        print(f"[INFO] 实验总结已保存:")
        print(f"  JSON: {summary_path}")
        print(f"  README: {readme_path}")

    def run_complete_ablation_generation(self) -> Dict[str, Any]:
        """运行完整的消融配置生成"""

        print("="*80)
        print("最优集合体配置消融验证实验生成器")
        print("="*80)

        # 1. 分析最优配置
        analysis = self.analyze_optimal_config()

        # 2. 生成消融配置
        ablation_configs = self.generate_ablation_configs(analysis)

        # 3. 保存所有配置
        saved_files = self.save_ablation_configs(ablation_configs)

        result = {
            'analysis': analysis,
            'ablation_configs': ablation_configs,
            'saved_files': saved_files,
            'total_experiments': len(ablation_configs) + 1
        }

        print(f"\n[SUCCESS] 消融验证实验生成完成!")
        print(f"  生成配置数: {result['total_experiments']}")
        print(f"  保存文件数: {len(saved_files)}")
        print(f"  输出目录: {self.output_dir}")

        return result

def main():
    parser = argparse.ArgumentParser(description='最优集合体配置消融验证实验生成器')
    parser.add_argument('--optimal_config', required=True,
                       help='最优集合体配置YAML路径 (optimal_ensemble_config.yaml)')
    parser.add_argument('--ablation_comparison', required=True,
                       help='消融对比CSV路径 (ablation_comparison_detailed.csv)')
    parser.add_argument('--output_dir', default='./optimal_ensemble_ablation',
                       help='输出目录')

    args = parser.parse_args()

    try:
        # 创建消融器
        ablator = OptimalEnsembleAblator(
            optimal_config_path=args.optimal_config,
            ablation_comparison_path=args.ablation_comparison,
            output_dir=args.output_dir
        )

        # 运行完整的消融生成
        result = ablator.run_complete_ablation_generation()

        print(f"\n[INFO] 下一步:")
        print(f"1. 使用批量训练脚本训练所有配置")
        print(f"2. 使用optimal_ensemble_evaluator.py评估结果")
        print(f"3. 验证最优集合体配置的有效性")

    except Exception as e:
        print(f"[ERROR] 消融验证实验生成失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()