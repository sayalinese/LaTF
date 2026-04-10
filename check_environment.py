#!/usr/bin/env python3
"""
LaTF (LaRE + FLH Fusion) - 环境检查脚本
验证所有必要的文件、路径和配置是否正确
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
from dotenv import load_dotenv

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    path = Path(filepath)
    exists = path.exists()
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {filepath}")
    if exists and path.is_file():
        print(f"  大小: {path.stat().st_size:,} 字节")
    return exists

def check_directory_exists(dirpath, description):
    """检查目录是否存在"""
    path = Path(dirpath)
    exists = path.exists() and path.is_dir()
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {dirpath}")
    if exists:
        # 统计文件数量
        file_count = len(list(path.rglob("*.*")))
        print(f"  文件数量: {file_count}")
    return exists

def check_python_packages():
    """检查必要的Python包"""
    packages = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "transformers": "Transformers",
        "PIL": "Pillow",
        "cv2": "OpenCV",
        "h5py": "HDF5",
        "dotenv": "python-dotenv",
        "tqdm": "进度条",
        "tensorboard": "TensorBoard"
    }
    
    print("\n检查Python包:")
    all_ok = True
    for import_name, display_name in packages.items():
        try:
            if import_name == "cv2":
                import cv2
                version = cv2.__version__
            elif import_name == "PIL":
                from PIL import Image
                version = Image.__version__
            else:
                module = __import__(import_name)
                version = getattr(module, "__version__", "未知版本")
            
            print(f"✓ {display_name}: {version}")
        except ImportError as e:
            print(f"✗ {display_name}: 未安装 ({e})")
            all_ok = False
    
    return all_ok

def check_gpu():
    """检查GPU配置"""
    print("\n检查GPU配置:")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✓ GPU可用: {gpu_count} 个设备")
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  设备 {i}: {name} ({memory:.1f} GB)")
        return True
    else:
        print("✗ GPU不可用，将使用CPU训练（速度会很慢）")
        return False

def check_flh():
    """检查 FLH 配置"""
    print("\n检查 FLH (取证定位头) 配置:")
    
    # 检查 FLH 模块文件
    flh_module = Path("service/forensic_localization.py")
    if flh_module.exists():
        print(f"✓ FLH 模块文件: {flh_module}")
    else:
        print(f"✗ FLH 模块文件不存在: {flh_module}")
        return False
    
    # 检查 FLH 训练脚本
    flh_train = Path("script/7_train_localization.py")
    if flh_train.exists():
        print(f"✓ FLH 训练脚本: {flh_train}")
    else:
        print(f"✗ FLH 训练脚本不存在: {flh_train}")
        return False
    
    # 检查 FLH 权重文件
    flh_weights = Path("outputs/flh_localization.pth")
    if flh_weights.exists():
        print(f"✓ FLH 权重文件: {flh_weights}")
        print(f"  大小: {flh_weights.stat().st_size:,} 字节")
    else:
        print(f"⚠ FLH 权重文件不存在: {flh_weights}")
        print("  请运行: python script\\7_train_localization.py")
    
    return True

def check_data_structure():
    """检查数据结构"""
    print("\n检查数据结构:")
    
    data_root = Path("data")
    required_dirs = [
        "doubao/fack",
        "doubao/masks",
        "doubao/real",
        "flux/animals",
        "flux/faces", 
        "flux/general",
        "flux/landscapes",
        "sdxl/animals",
        "sdxl/faces",
        "sdxl/general",
        "sdxl/landscapes",
        "Real/FFHQ",
        "Real/FORLAB"
    ]
    
    all_ok = True
    for subdir in required_dirs:
        dir_path = data_root / subdir
        if dir_path.exists():
            file_count = len(list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png")))
            print(f"✓ {subdir}: {file_count} 张图片")
        else:
            print(f"✗ {subdir}: 目录不存在")
            all_ok = False
    
    return all_ok

def check_env_config():
    """检查环境变量配置"""
    print("\n检查环境变量配置:")
    
    # 加载.env文件
    load_dotenv(Path(__file__).resolve().parent / '.env')
    
    env_vars = {
        "DOUBAO_WEIGHT": "Doubao权重倍数",
        "DOUBAO_SAMPLE_RATIO": "Doubao采样比例",
        "REGULAR_SAMPLE_RATIO": "常规数据采样比例",
        "DOUBAO_VAL_RATIO": "验证集Doubao比例",
        "USE_FLH": "启用FLH定位头",
        "FLH_LOSS_WEIGHT": "FLH损失权重",
        "BATCH_SIZE": "批次大小",
        "LR": "学习率",
        "OUT_DIR": "输出目录"
    }
    
    all_ok = True
    for var, desc in env_vars.items():
        value = os.getenv(var)
        if value:
            print(f"✓ {desc}: {value}")
        else:
            print(f"✗ {desc}: 未设置")
            all_ok = False
    
    return all_ok

def main():
    print("="*80)
    print("LaRE 训练环境检查")
    print("="*80)
    
    # 检查Python包
    packages_ok = check_python_packages()
    
    # 检查GPU
    gpu_ok = check_gpu()
    
    # 检查 FLH
    flh_ok = check_flh()
    
    # 检查数据结构
    data_ok = check_data_structure()
    
    # 检查环境配置
    env_ok = check_env_config()
    
    # 检查必要的脚本文件
    print("\n检查必要的脚本文件:")
    scripts = [
        ("script/1_gen_annotations.py", "标注生成脚本"),
        ("script/2_extract_features.py", "特征提取脚本"),
        ("script/7_train_localization.py", "FLH定位头训练脚本"),
        ("script/5_train_model_v11.py", "V11训练脚本"),
        ("test/evaluate_by_category.py", "评估脚本"),
        ("service/data_prep.py", "数据准备模块"),
        ("service/model_v11_fusion.py", "V11模型模块"),
        ("service/dataset.py", "数据集模块")
    ]
    
    scripts_ok = True
    for script_path, description in scripts:
        if Path(script_path).exists():
            print(f"✓ {description}: {script_path}")
        else:
            print(f"✗ {description}: {script_path} 不存在")
            scripts_ok = False
    
    # 总结
    print("\n" + "="*80)
    print("检查结果总结:")
    print(f"Python包: {'✓ 通过' if packages_ok else '✗ 失败'}")
    print(f"GPU配置: {'✓ 通过' if gpu_ok else '⚠ 警告（将使用CPU）'}")
    print(f"FLH配置: {'✓ 通过' if flh_ok else '✗ 失败'}")
    print(f"数据结构: {'✓ 通过' if data_ok else '✗ 失败'}")
    print(f"环境配置: {'✓ 通过' if env_ok else '✗ 失败'}")
    print(f"脚本文件: {'✓ 通过' if scripts_ok else '✗ 失败'}")
    print("="*80)
    
    if all([packages_ok, flh_ok, data_ok, env_ok, scripts_ok]):
        print("\n✅ 所有检查通过！可以开始训练。")
        print("运行命令: python train_full_pipeline.py")
    else:
        print("\n⚠ 存在一些问题，请根据上面的提示修复。")
        if not flh_ok:
            print("\nFLH问题解决方案:")
            print("1. 确保 service/forensic_localization.py 存在")
            print("2. 运行: python script\\7_train_localization.py")
        
        if not data_ok:
            print("\n数据结构问题解决方案:")
            print("确保 data/ 目录下有正确的子目录结构")
            print("参考: data/doubao/fack/, data/doubao/real/, data/flux/, data/sdxl/, data/Real/")

if __name__ == "__main__":
    main()