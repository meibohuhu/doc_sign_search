#!/usr/bin/env python3
"""
诊断训练瓶颈：
1. Vision Encoder是否在更新（参数变化）
2. Vision-Text对齐问题（特征相似度/attention）
"""

import torch
import json
import sys
import os
import argparse
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple


def load_checkpoint_params(checkpoint_path: str, vision_only: bool = True) -> Dict[str, torch.Tensor]:
    """加载checkpoint中的参数，特别是vision相关的"""
    try:
        # 尝试加载DeepSpeed checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # DeepSpeed checkpoint格式
        if 'module' in checkpoint:
            state_dict = checkpoint['module']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif isinstance(checkpoint, dict) and any('visual' in k for k in checkpoint.keys()):
            state_dict = checkpoint
        else:
            print(f"⚠️  Unknown checkpoint format in {checkpoint_path}")
            return {}
        
        # 提取vision相关参数
        vision_params = {}
        if vision_only:
            for key, value in state_dict.items():
                if 'visual' in key.lower() or 'vision' in key.lower():
                    vision_params[key] = value
        else:
            vision_params = state_dict
        
        print(f"✅ Loaded {len(vision_params)} vision parameters from {checkpoint_path}")
        return vision_params
    
    except Exception as e:
        print(f"❌ Error loading checkpoint {checkpoint_path}: {e}")
        return {}


def compare_checkpoints(checkpoint1: str, checkpoint2: str) -> Dict[str, float]:
    """比较两个checkpoint中vision encoder参数的变化"""
    params1 = load_checkpoint_params(checkpoint1)
    params2 = load_checkpoint_params(checkpoint2)
    
    if not params1 or not params2:
        return {}
    
    # 找到共同的参数
    common_keys = set(params1.keys()) & set(params2.keys())
    if not common_keys:
        print("⚠️  No common vision parameters found between checkpoints")
        return {}
    
    print(f"\n📊 Comparing {len(common_keys)} common vision parameters...")
    
    changes = {
        'total_params': len(common_keys),
        'changed_params': 0,
        'mean_relative_change': 0.0,
        'max_relative_change': 0.0,
        'mean_absolute_change': 0.0,
        'params_with_significant_change': 0
    }
    
    relative_changes = []
    absolute_changes = []
    
    for key in common_keys:
        p1 = params1[key].float()
        p2 = params2[key].float()
        
        # 计算相对变化
        if p1.norm() > 1e-8:  # 避免除零
            relative_change = ((p2 - p1).norm() / p1.norm()).item()
            relative_changes.append(relative_change)
            
            if relative_change > 1e-6:  # 有显著变化
                changes['changed_params'] += 1
                if relative_change > 0.001:  # 超过0.1%的变化
                    changes['params_with_significant_change'] += 1
        
        # 计算绝对变化
        absolute_change = (p2 - p1).abs().mean().item()
        absolute_changes.append(absolute_change)
    
    if relative_changes:
        changes['mean_relative_change'] = np.mean(relative_changes)
        changes['max_relative_change'] = np.max(relative_changes)
    if absolute_changes:
        changes['mean_absolute_change'] = np.mean(absolute_changes)
    
    return changes


def diagnose_from_config(config_dict: Dict) -> Dict[str, str]:
    """从训练配置诊断"""
    diagnosis = {
        'vision_encoder_status': 'unknown',
        'learning_rate_issue': 'unknown',
        'capacity_issue': 'unknown',
        'recommendations': []
    }
    
    # 检查vision encoder是否被冻结
    freeze_vision = config_dict.get('freeze_vision_tower', True)
    vision_lr = config_dict.get('vision_lr', None)
    freeze_llm = config_dict.get('freeze_llm', True)
    
    if freeze_vision:
        diagnosis['vision_encoder_status'] = '❌ FROZEN - Vision encoder not updating'
        diagnosis['recommendations'].append('Set --freeze_vision_tower False')
    elif vision_lr is not None:
        if vision_lr < 1e-7:
            diagnosis['vision_encoder_status'] = '⚠️  TOO LOW LR - Vision encoder may be updating too slowly'
            diagnosis['learning_rate_issue'] = 'Vision LR is extremely low'
            diagnosis['recommendations'].append(f'Increase --vision_lr from {vision_lr} to at least 1e-6 or 1e-5')
        elif vision_lr > 1e-4:
            diagnosis['vision_encoder_status'] = '⚠️  TOO HIGH LR - Vision encoder may be unstable'
            diagnosis['recommendations'].append(f'Decrease --vision_lr from {vision_lr} to 1e-5 or 1e-6')
        else:
            diagnosis['vision_encoder_status'] = '✅ NOT FROZEN - Vision encoder should be updating'
    
    # 检查模型容量
    if freeze_llm:
        lora_enable = config_dict.get('lora_enable', False)
        if lora_enable:
            lora_rank = config_dict.get('lora_rank', 8)
            diagnosis['capacity_issue'] = f'⚠️  LIMITED - Only LoRA (rank={lora_rank}) trainable on LLM'
            if lora_rank < 64:
                diagnosis['recommendations'].append(f'Consider increasing --lora_rank from {lora_rank} to 64 or 128')
        else:
            diagnosis['capacity_issue'] = '❌ VERY LIMITED - LLM fully frozen, only merger trainable'
            diagnosis['recommendations'].append('Consider --freeze_llm False or enable LoRA')
    
    return diagnosis


def analyze_training_output(log_file: str) -> Dict:
    """分析训练日志，查找梯度信息"""
    analysis = {
        'vision_gradients_found': False,
        'gradient_norm': None,
        'loss_trend': []
    }
    
    if not os.path.exists(log_file):
        return analysis
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # 查找vision相关的梯度信息
        for line in lines:
            if 'vision' in line.lower() and 'grad' in line.lower():
                analysis['vision_gradients_found'] = True
                if 'norm' in line.lower():
                    # 尝试提取梯度范数
                    import re
                    nums = re.findall(r'[\d.]+', line)
                    if nums:
                        analysis['gradient_norm'] = float(nums[0])
                break
        
        # 查找loss趋势（最后几行）
        for line in lines[-100:]:
            if 'loss' in line.lower():
                import re
                loss_vals = re.findall(r'loss[:\s]+([\d.]+)', line, re.IGNORECASE)
                if loss_vals:
                    try:
                        analysis['loss_trend'].append(float(loss_vals[0]))
                    except:
                        pass
    except Exception as e:
        print(f"⚠️  Error analyzing log file: {e}")
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description='诊断训练瓶颈')
    parser.add_argument('--checkpoint1', type=str, help='第一个checkpoint路径（如checkpoint-1000）')
    parser.add_argument('--checkpoint2', type=str, help='第二个checkpoint路径（如checkpoint-5000）')
    parser.add_argument('--config', type=str, help='训练配置JSON或训练脚本路径')
    parser.add_argument('--log', type=str, help='训练日志文件路径')
    parser.add_argument('--training_script', type=str, 
                       help='训练脚本路径（会自动解析配置）',
                       default='/home/mh2803/projects/sign_language_llm/qwenvl/Qwen2-VL-Finetune/scripts/how2sign_train_update/finetune_qwen2vl_how2sign_2xa100_filtered_320resolution_scavenger.sh')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔍 训练瓶颈诊断工具")
    print("=" * 80)
    
    # 1. 从训练脚本提取配置
    print("\n1️⃣  分析训练配置...")
    config = {}
    if args.training_script and os.path.exists(args.training_script):
        with open(args.training_script, 'r') as f:
            script_content = f.read()
        
        # 简单解析bash脚本中的参数
        if '--freeze_vision_tower False' in script_content:
            config['freeze_vision_tower'] = False
        elif '--freeze_vision_tower True' in script_content:
            config['freeze_vision_tower'] = True
        
        if '--freeze_llm False' in script_content:
            config['freeze_llm'] = False
        elif '--freeze_llm True' in script_content:
            config['freeze_llm'] = True
        
        # 提取vision_lr
        import re
        vision_lr_match = re.search(r'--vision_lr\s+([\d.e-]+)', script_content)
        if vision_lr_match:
            config['vision_lr'] = float(vision_lr_match.group(1))
        
        merger_lr_match = re.search(r'--merger_lr\s+([\d.e-]+)', script_content)
        if merger_lr_match:
            config['merger_lr'] = float(merger_lr_match.group(1))
        
        lora_match = re.search(r'--lora_enable\s+(True|False)', script_content)
        if lora_match:
            config['lora_enable'] = lora_match.group(1) == 'True'
        
        lora_rank_match = re.search(r'--lora_rank\s+(\d+)', script_content)
        if lora_rank_match:
            config['lora_rank'] = int(lora_rank_match.group(1))
        
        print(f"  📝 配置解析:")
        print(f"     - freeze_vision_tower: {config.get('freeze_vision_tower', 'unknown')}")
        print(f"     - freeze_llm: {config.get('freeze_llm', 'unknown')}")
        print(f"     - vision_lr: {config.get('vision_lr', 'not set')}")
        print(f"     - merger_lr: {config.get('merger_lr', 'not set')}")
        print(f"     - lora_enable: {config.get('lora_enable', 'unknown')}")
        print(f"     - lora_rank: {config.get('lora_rank', 'unknown')}")
    
    # 2. 诊断配置
    if config:
        config_diag = diagnose_from_config(config)
        print(f"\n  🔍 Vision Encoder状态: {config_diag['vision_encoder_status']}")
        print(f"  🔍 容量问题: {config_diag['capacity_issue']}")
        if config_diag['recommendations']:
            print(f"\n  💡 建议:")
            for rec in config_diag['recommendations']:
                print(f"     - {rec}")
    
    # 3. 比较checkpoints（如果提供）
    if args.checkpoint1 and args.checkpoint2:
        print("\n2️⃣  比较checkpoint参数变化...")
        changes = compare_checkpoints(args.checkpoint1, args.checkpoint2)
        
        if changes:
            print(f"\n  📊 参数变化统计:")
            print(f"     - 总参数数: {changes.get('total_params', 0)}")
            print(f"     - 有变化的参数: {changes.get('changed_params', 0)}")
            print(f"     - 平均相对变化: {changes.get('mean_relative_change', 0):.2e}")
            print(f"     - 最大相对变化: {changes.get('max_relative_change', 0):.2e}")
            print(f"     - 平均绝对变化: {changes.get('mean_absolute_change', 0):.2e}")
            print(f"     - 显著变化参数 (>0.1%): {changes.get('params_with_significant_change', 0)}")
            
            # 判断vision encoder是否在更新
            if changes.get('mean_relative_change', 0) < 1e-8:
                print("\n  ❌ 判断: Vision Encoder可能没有更新（变化太小）")
                print("     原因可能是:")
                print("     - 学习率太低")
                print("     - Vision encoder被冻结")
                print("     - 梯度没有回传到vision encoder")
            elif changes.get('mean_relative_change', 0) < 1e-6:
                print("\n  ⚠️  判断: Vision Encoder更新非常缓慢")
                print("     建议: 增加vision_lr")
            else:
                print("\n  ✅ 判断: Vision Encoder在更新")
                print("     如果BLEU仍然很低，可能是vision-text对齐问题")
    
    # 4. 分析训练日志
    if args.log and os.path.exists(args.log):
        print("\n3️⃣  分析训练日志...")
        log_analysis = analyze_training_output(args.log)
        if log_analysis['vision_gradients_found']:
            print("  ✅ 发现vision梯度信息")
        else:
            print("  ⚠️  未发现vision梯度信息（可能需要查看完整日志）")
    
    # 5. 综合诊断
    print("\n" + "=" * 80)
    print("📋 综合诊断结论")
    print("=" * 80)
    
    print("\n🎯 可能的瓶颈:")
    
    if config.get('freeze_vision_tower', True):
        print("1. ❌ Vision Encoder被冻结 → 这是主要瓶颈！")
        print("   → 必须设置 --freeze_vision_tower False")
    elif config.get('vision_lr', 1) < 1e-6:
        print("1. ⚠️  Vision Encoder学习率太低 (2e-6)")
        print("   → 建议增加到 1e-5 或 2e-5")
        print("   → 这是最可能的瓶颈！")
    else:
        print("1. ✅ Vision Encoder配置正常")
    
    if config.get('freeze_llm', True) and config.get('lora_rank', 8) < 32:
        print("2. ⚠️  模型容量受限 (LLM冻结 + LoRA rank < 32)")
        print("   → 可能需要更多参数学习sign language mapping")
    else:
        print("2. ✅ 模型容量配置合理")
    
    print("\n💡 建议的修复顺序:")
    print("   1. 首先确认vision encoder未被冻结")
    print("   2. 增加vision_lr从2e-6到1e-5（最重要！）")
    print("   3. 如果仍不行，增加LoRA rank或unfreeze部分LLM layers")
    print("   4. 添加vision-text对齐的loss或contrastive learning")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

