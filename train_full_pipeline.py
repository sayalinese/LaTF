#!/usr/bin/env python3
"""
LaTF (LaRE + TruFor Fusion) 训练全流程脚本
包含：数据准备 → 特征提取 → TruFor特征生成 → 模型训练 → 模型评估
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time

def run_command(cmd, description):
    """运行命令并打印输出"""
    print(f"\n{'='*60}")
    print(f"执行: {description}")
    print(f"命令: {cmd}")
    print('='*60)
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} 完成 ({time.time()-start_time:.1f}秒)")
        print(f"输出: {result.stdout[:500]}...")  # 只显示前500字符
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} 失败")
        print(f"错误: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='LaRE 训练全流程')
    parser.add_argument('--skip_data_prep', action='store_true', help='跳过数据准备步骤')
    parser.add_argument('--skip_feature_extract', action='store_true', help='跳过特征提取步骤')
    parser.add_argument('--skip_trufor', action='store_true', help='跳过 TruFor 特征生成')
    parser.add_argument('--skip_training', action='store_true', help='跳过模型训练')
    parser.add_argument('--skip_evaluation', action='store_true', help='跳过模型评估')
    parser.add_argument('--data_root', type=str, default='data', help='数据根目录')
    parser.add_argument('--out_dir', type=str, default='outputs/v13_doubao_focused', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=16, help='训练批次大小')
    parser.add_argument('--epochs', type=int, default=25, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--texture_model', type=str, default='convnext_tiny', help='纹理分支模型')
    parser.add_argument('--clip_type', type=str, default='RN50x64', help='CLIP 视觉骨干')
    parser.add_argument('--highres_size', type=int, default=512, help='高分辨率输入尺寸')
    
    args = parser.parse_args()
    
    # 设置工作目录
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    
    print("="*80)
    print("LaRE 训练全流程开始")
    print("="*80)
    
    # 1. 数据准备
    if not args.skip_data_prep:
        success = run_command(
            "python script\\1_gen_annotations.py",
            "生成标注文件（使用 Doubao 权重增强）"
        )
        if not success:
            print("数据准备失败，退出流程")
            return
    
    # 2. 特征提取
    if not args.skip_feature_extract:
        success = run_command(
            "python script\\2_extract_features.py",
            "提取 LaRE 特征"
        )
        if not success:
            print("特征提取失败，退出流程")
            return
    
    # 3. TruFor 特征生成
    if not args.skip_trufor:
        success = run_command(
            "python script\\5_gen_trufor_maps.py",
            "生成 TruFor 特征图"
        )
        if not success:
            print("TruFor 特征生成失败，退出流程")
            return
    
    # 4. 模型训练
    if not args.skip_training:
        train_cmd = "python script\\5_train_model_v11.py --use_amp"
        
        success = run_command(train_cmd, "训练 V11 Fusion + TruFor 模型")
        if not success:
            print("模型训练失败，退出流程")
            return
    
    # 5. 模型评估
    if not args.skip_evaluation:
        eval_cmd = f"python test\\evaluate_by_category.py --model {args.out_dir}/best.pth"
        
        success = run_command(eval_cmd, "评估模型性能")
        if not success:
            print("模型评估失败")
    
    print("\n" + "="*80)
    print("训练全流程完成！")
    print(f"模型保存在: {args.out_dir}/best.pth")
    print("="*80)

if __name__ == "__main__":
    main()