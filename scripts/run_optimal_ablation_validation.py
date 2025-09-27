#!/usr/bin/env python3
"""
最优集合体配置消融验证一键运行脚本
One-Click Optimal Ensemble Ablation Validation Runner
"""

import subprocess
import sys
from pathlib import Path
import argparse

def run_command(cmd: str, description: str) -> bool:
    """运行命令并返回是否成功"""
    print(f"\n[STEP] {description}")
    print(f"[CMD] {cmd}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"[WARNING] {result.stderr}")
        print(f"[SUCCESS] {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} 失败")
        print(f"返回码: {e.returncode}")
        print(f"错误输出: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='最优集合体配置消融验证一键运行')
    parser.add_argument('--optimal_config', required=True,
                       help='最优集合体配置YAML路径')
    parser.add_argument('--ablation_comparison', required=True,
                       help='消融对比CSV路径')
    parser.add_argument('--data_path', required=True,
                       help='数据文件路径')
    parser.add_argument('--output_dir', default='./optimal_ablation_validation',
                       help='总输出目录')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--gpus', default='0',
                       help='GPU设备')

    args = parser.parse_args()

    print("="*80)
    print("最优集合体配置消融验证一键运行系统")
    print("="*80)
    print(f"最优配置: {args.optimal_config}")
    print(f"消融数据: {args.ablation_comparison}")
    print(f"数据路径: {args.data_path}")
    print(f"输出目录: {args.output_dir}")

    # 创建输出目录结构
    output_dir = Path(args.output_dir)
    configs_dir = output_dir / "configs"
    training_dir = output_dir / "training_results"
    evaluation_dir = output_dir / "evaluation_results"

    # 第1步: 生成消融验证配置
    step1_cmd = f'''python scripts/optimal_ensemble_ablator.py \\
        --optimal_config "{args.optimal_config}" \\
        --ablation_comparison "{args.ablation_comparison}" \\
        --output_dir "{configs_dir}"'''

    if not run_command(step1_cmd, "生成消融验证配置"):
        print("[FATAL] 配置生成失败，停止执行")
        return False

    # 第2步: 训练消融验证模型
    step2_cmd = f'''python scripts/multi_console_train.py \\
        --yaml-dir "{configs_dir}" \\
        --output-root "{training_dir}" \\
        --data-path "{args.data_path}" \\
        --gpus {args.gpus} \\
        --mode background \\
        --batch-size 3 \\
        --epochs {args.epochs} \\
        --wait-between 5 \\
        --gpu-max-proc 3'''

    print(f"\n[INFO] 将训练 {len(list(configs_dir.glob('*.yaml')))} 个配置...")
    if not run_command(step2_cmd, "批量训练消融验证模型"):
        print("[WARNING] 训练可能有部分失败，继续评估已完成的模型")

    # 第3步: 查找最优模型和消融模型
    optimal_model_path = None
    ablation_models_dir = training_dir / "bestmodel"

    # 查找最优配置的模型
    for model_file in (training_dir / "bestmodel").glob("optimal_ensemble_reference_*.pt"):
        optimal_model_path = model_file
        break

    if not optimal_model_path:
        print("[ERROR] 未找到最优配置的训练模型")
        print("[INFO] 请检查训练是否成功完成")
        return False

    # 第4步: 评估消融验证结果
    step4_cmd = f'''python scripts/optimal_ensemble_evaluator.py \\
        --optimal_model_path "{optimal_model_path}" \\
        --ablation_models_dir "{ablation_models_dir}" \\
        --data_path "{args.data_path}" \\
        --output_dir "{evaluation_dir}"'''

    if not run_command(step4_cmd, "评估消融验证结果"):
        print("[ERROR] 评估失败")
        return False

    # 第5步: 生成最终总结
    print("\n" + "="*80)
    print("最优集合体配置消融验证完成")
    print("="*80)

    print(f"\n[RESULTS] 查看结果:")
    print(f"  配置文件: {configs_dir}")
    print(f"  训练结果: {training_dir}")
    print(f"  评估报告: {evaluation_dir}/ablation_validation_report.md")
    print(f"  重要性分析: {evaluation_dir}/component_importance_analysis.png")
    print(f"  性能对比: {evaluation_dir}/performance_comparison.csv")

    print(f"\n[VALIDATION] 验证结论:")
    try:
        # 读取评估结果
        import json
        results_file = evaluation_dir / "ablation_validation_results.json"
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)

            most_important = results.get('summary', {}).get('most_important_component')
            avg_drop = results.get('summary', {}).get('average_performance_drop', 0)

            if most_important:
                print(f"  最重要组件: {most_important}")
            print(f"  平均性能下降: {avg_drop:.1f}%")
            print(f"  验证了 {results.get('summary', {}).get('total_ablations', 0)} 个组件的重要性")

    except Exception as e:
        print(f"  [ERROR] 无法读取评估结果: {e}")

    print(f"\n[SUCCESS] 最优集合体配置消融验证完成！")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)