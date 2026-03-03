"""
Simple Local Anomaly Detector (Statistical Method)
不需要训练，基于LaRE统计特性检测局部篡改

原理：
- 真实图：LaRE分布均匀，无明显峰值
- 全局AI：LaRE整体偏高，分布均匀
- 局部篡改：LaRE有局部峰值，集中在小区域
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class StatisticalLocalDetector:
    """
    基于LaRE统计特性的局部篡改检测器
    无需训练，直接分析LaRE分布模式
    """
    
    def __init__(
        self,
        peak_threshold: float = 3.0,       # 提高：减少误报，只捕捉明显异常
        concentration_max: float = 0.15,   # 降低：更严格的局部集中要求
        min_std_ratio: float = 0.35,       # 提高：需要更明显的变化
    ):
        """
        Args:
            peak_threshold: 判定存在峰值的阈值（max/mean > threshold）
            concentration_max: 高激活区域占比上限（局部篡改特征）
            min_std_ratio: 最小标准差比例（过低说明太均匀）
        """
        self.peak_threshold = peak_threshold
        self.concentration_max = concentration_max
        self.min_std_ratio = min_std_ratio
    
    def analyze(self, loss_map: torch.Tensor) -> dict:
        """
        分析LaRE分布特征
        
        Args:
            loss_map: [1, 4, 32, 32] or [4, 32, 32] from LaRE
            
        Returns:
            dict with analysis results
        """
        if loss_map.dim() == 4:
            loss_map = loss_map.squeeze(0)
        
        # 取4通道均值 [32, 32]
        lm = loss_map.float().mean(dim=0)
        
        # 基础统计
        mean = lm.mean().item()
        std = lm.std().item()
        min_val = lm.min().item()
        max_val = lm.max().item()
        
        # 峰值比：最大值相对均值的倍数
        peak_ratio = max_val / (mean + 1e-6)
        
        # 标准差比：衡量分布离散程度
        std_ratio = std / (mean + 1e-6)
        
        # 高激活区域占比：超过 mean + 2*std 的像素比例
        threshold_high = mean + 2 * std
        high_activation_mask = (lm > threshold_high).float()
        concentration = high_activation_mask.mean().item()
        
        # 局部峰值检测：用max pooling找局部最大值
        lm_4d = lm.unsqueeze(0).unsqueeze(0)  # [1, 1, 32, 32]
        local_max = F.max_pool2d(lm_4d, kernel_size=5, stride=1, padding=2)
        is_local_max = (lm == local_max.squeeze())
        num_peaks = (is_local_max & (lm > threshold_high)).sum().item()
        
        # 边界效应检测：边界2像素内的高激活占比
        border_mask = torch.zeros_like(lm, dtype=torch.bool)
        border_mask[:2, :] = True
        border_mask[-2:, :] = True
        border_mask[:, :2] = True
        border_mask[:, -2:] = True
        border_high = (high_activation_mask.bool() & border_mask).sum().item()
        total_high = high_activation_mask.sum().item()
        border_ratio = border_high / (total_high + 1e-6)
        
        return {
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val,
            'peak_ratio': peak_ratio,
            'std_ratio': std_ratio,
            'concentration': concentration,
            'num_peaks': num_peaks,
            'border_ratio': border_ratio,
            'high_activation_mask': high_activation_mask,
        }
    
    def detect(self, loss_map: torch.Tensor) -> Tuple[bool, float, Optional[np.ndarray]]:
        """
        检测局部篡改
        
        Args:
            loss_map: [1, 4, 32, 32] from LaRE
            
        Returns:
            is_tampered: 是否检测到局部篡改
            score: 篡改置信度 (0-1)
            loc_map: 定位热力图 (numpy array, 0-1)
        """
        stats = self.analyze(loss_map)
        
        # 判断条件：
        # 1. 存在明显峰值（peak_ratio > threshold）
        # 2. 高激活区域集中（concentration < max）
        # 3. 分布不太均匀（std_ratio > min）
        # 4. 不是边界伪影（border_ratio < 0.8）
        
        has_peak = stats['peak_ratio'] > self.peak_threshold
        is_localized = stats['concentration'] < self.concentration_max
        has_variation = stats['std_ratio'] > self.min_std_ratio
        not_border_artifact = stats['border_ratio'] < 0.8
        
        is_tampered = has_peak and is_localized and has_variation and not_border_artifact
        
        # 计算置信度分数
        # 基于peak_ratio的置信度，归一化到0-1
        score = min(1.0, (stats['peak_ratio'] - 1.0) / (self.peak_threshold * 2))
        
        # 如果是边界伪影，降低分数
        if stats['border_ratio'] > 0.5:
            score *= 0.5
        
        # 如果分布太均匀（全局AI特征），这个检测器不适用
        if stats['concentration'] > 0.5:
            score *= 0.3  # 可能是全局AI，不是局部篡改
        
        # 生成定位图
        loc_map = stats['high_activation_mask'].cpu().numpy()
        
        return is_tampered, max(0, min(1, score)), loc_map
    
    def get_explanation(self, loss_map: torch.Tensor) -> str:
        """
        生成人类可读的分析解释
        """
        stats = self.analyze(loss_map)
        
        lines = [
            f"LaRE分析结果:",
            f"  均值: {stats['mean']:.4f}",
            f"  标准差: {stats['std']:.4f}",
            f"  峰值比(max/mean): {stats['peak_ratio']:.2f}",
            f"  高激活区域占比: {stats['concentration']*100:.1f}%",
            f"  边界伪影比例: {stats['border_ratio']*100:.1f}%",
            f"  局部峰值数量: {stats['num_peaks']}",
        ]
        
        # 判断分析
        if stats['peak_ratio'] > self.peak_threshold:
            lines.append(f"  → 检测到明显峰值 (>{self.peak_threshold})")
        
        if stats['concentration'] < self.concentration_max:
            lines.append(f"  → 高激活区域集中 (<{self.concentration_max*100:.0f}%)")
        elif stats['concentration'] > 0.5:
            lines.append(f"  → 高激活区域分散，可能是全局AI生成")
        
        if stats['border_ratio'] > 0.5:
            lines.append(f"  → 警告：高激活主要在边界，可能是伪影")
        
        return "\n".join(lines)


class DualModelDetector:
    """
    双模型协同检测器
    - Model1: 全局AI检测（已训练的V9Lite）
    - Model2: 统计方法（仅对粗糙篡改有效，对高质量inpaint如Doubao无效）
    
    注意：经测试，Doubao的LaRE特征与真实图几乎相同，
    统计方法无法区分。因此对局部篡改主要依赖全局模型。
    """
    
    def __init__(
        self,
        global_model,
        lare_extractor,
        device: str = "cuda",
        global_high_threshold: float = 0.85,  # 降低：让更多样本进入检测
        global_low_threshold: float = 0.25,   # 提高：减少漏检
        local_weight: float = 0.3,            # 降低：统计方法效果差，降低权重
    ):
        """
        Args:
            global_model: 全局检测模型 (V9Lite)
            lare_extractor: LaRE特征提取器
            device: 推理设备
            global_high_threshold: 全局模型高置信度阈值
            global_low_threshold: 全局模型低置信度阈值
            local_weight: 灰度区域时局部模型的权重
        """
        self.global_model = global_model
        self.lare_extractor = lare_extractor
        self.device = device
        self.global_high = global_high_threshold
        self.global_low = global_low_threshold
        self.local_weight = local_weight
        
        # 统计局部检测器
        self.local_detector = StatisticalLocalDetector()
        
        # 预处理
        from torchvision import transforms
        self.preprocess = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])
    
    def predict(self, image) -> dict:
        """
        双模型协同预测
        
        Args:
            image: PIL Image
            
        Returns:
            dict with prediction results
        """
        # 1. 提取特征
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        loss_map = self.lare_extractor.extract_single(image).to(self.device)
        
        # 2. 全局模型推理
        self.global_model.eval()
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = self.global_model(img_tensor, loss_map)
            logits = logits.float()
            probs = F.softmax(logits, dim=1)
            prob_global = probs[0, 1].item()
        
        # 3. 快速判定（高/低置信度）
        if prob_global > self.global_high:
            return {
                'prediction': 'AI Generated',
                'confidence': prob_global,
                'prob_global': prob_global,
                'prob_local': None,
                'detection_type': 'global_high_confidence',
                'loc_map': None,
                'explanation': f"全局检测器高置信度判定 ({prob_global:.1%})"
            }
        
        if prob_global < self.global_low:
            return {
                'prediction': 'Real Photo',
                'confidence': 1 - prob_global,
                'prob_global': prob_global,
                'prob_local': None,
                'detection_type': 'global_low_confidence',
                'loc_map': None,
                'explanation': f"全局检测器判定真实 ({1-prob_global:.1%})"
            }
        
        # 4. 灰度区域 → 局部检测
        is_tampered, prob_local, loc_map = self.local_detector.detect(loss_map)
        local_explanation = self.local_detector.get_explanation(loss_map)
        
        # 5. 融合决策
        final_prob = (1 - self.local_weight) * prob_global + self.local_weight * prob_local
        
        # 局部检测器明确判定篡改时，提高权重
        if is_tampered and prob_local > 0.5:
            final_prob = max(final_prob, prob_local)
        
        if final_prob > 0.5:
            prediction = 'AI Modified (Local)'
            confidence = final_prob
        else:
            prediction = 'Real Photo'
            confidence = 1 - final_prob
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'prob_global': prob_global,
            'prob_local': prob_local,
            'detection_type': 'dual_model_fusion',
            'loc_map': loc_map,
            'explanation': f"双模型融合检测\n全局: {prob_global:.1%}, 局部: {prob_local:.1%}\n{local_explanation}"
        }


# 测试代码
if __name__ == "__main__":
    # 创建随机测试数据
    import torch
    
    detector = StatisticalLocalDetector()
    
    # 模拟真实图（均匀低激活）
    real_lare = torch.rand(1, 4, 32, 32) * 0.3
    print("=== 真实图 ===")
    is_tampered, score, _ = detector.detect(real_lare)
    print(f"检测结果: {is_tampered}, 分数: {score:.3f}")
    print(detector.get_explanation(real_lare))
    
    # 模拟局部篡改（有明显峰值）
    tampered_lare = torch.rand(1, 4, 32, 32) * 0.2
    tampered_lare[:, :, 10:15, 10:15] = 1.5  # 局部高激活
    print("\n=== 局部篡改 ===")
    is_tampered, score, _ = detector.detect(tampered_lare)
    print(f"检测结果: {is_tampered}, 分数: {score:.3f}")
    print(detector.get_explanation(tampered_lare))
    
    # 模拟全局AI（整体高激活）
    global_ai_lare = torch.rand(1, 4, 32, 32) * 0.5 + 0.5
    print("\n=== 全局AI ===")
    is_tampered, score, _ = detector.detect(global_ai_lare)
    print(f"检测结果: {is_tampered}, 分数: {score:.3f}")
    print(detector.get_explanation(global_ai_lare))
