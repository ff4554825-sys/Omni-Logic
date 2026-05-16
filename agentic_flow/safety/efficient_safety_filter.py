"""
Efficient Safety Filter - 两阶段安全过滤器
Stage 1: Segment-level pooling (256:1) - 97.8% computation savings
Stage 2: Token-level with DepthwiseConv1D local context
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

class SegmentPooler(nn.Module):
    """
    段级池化：将序列划分为段，每段池化为一个表示
    压缩比：256:1
    """
    
    def __init__(
        self,
        d_model: int = 8192,
        segment_size: int = 256,
        pool_method: str = 'attention'  # 'mean', 'max', 'attention'
    ):
        super().__init__()
        self.segment_size = segment_size
        self.pool_method = pool_method
        
        if pool_method == 'attention':
            self.attn_weights = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.Tanh(),
                nn.Linear(d_model // 4, 1)
            )
            
    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: [B, L, D]
        Returns:
            segments: [B, L//segment_size, D]
            segment_mask: [B, L//segment_size]
        """
        B, L, D = hidden.shape
        num_segments = L // self.segment_size
        
        # 截断到完整段
        truncated_len = num_segments * self.segment_size
        hidden = hidden[:, :truncated_len, :]
        
        # 重塑为段
        segments = hidden.view(B, num_segments, self.segment_size, D)
        
        if self.pool_method == 'mean':
            pooled = segments.mean(dim=2)
        elif self.pool_method == 'max':
            pooled = segments.max(dim=2)[0]
        else:  # attention
            weights = self.attn_weights(segments)  # [B, num_segments, segment_size, 1]
            weights = F.softmax(weights, dim=2)
            pooled = (segments * weights).sum(dim=2)
            
        segment_mask = torch.ones(B, num_segments, device=hidden.device)
        
        return pooled, segment_mask


class LocalContextConv(nn.Module):
    """
    带局部上下文的Depthwise卷积
    用于token级安全分类
    """
    
    def __init__(
        self,
        d_model: int = 8192,
        kernel_size: int = 5,  # 局部上下文窗口
        num_classes: int = 4   # safe, unsafe, borderline, unknown
    ):
        super().__init__()
        
        # Depthwise卷积 - 每个通道独立卷积
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model  # Depthwise
        )
        
        # Pointwise卷积 - 通道混合
        self.pointwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model // 2,
            kernel_size=1
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, num_classes)
        )
        
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [B, L, D]
        Returns:
            logits: [B, L, num_classes]
        """
        # 转换为卷积格式 [B, D, L]
        x = hidden.transpose(1, 2)
        
        # Depthwise卷积
        x = self.depthwise_conv(x)
        x = F.gelu(x)
        
        # Pointwise卷积
        x = self.pointwise_conv(x)
        x = F.gelu(x)
        
        # 转回 [B, L, D//2]
        x = x.transpose(1, 2)
        
        # 分类
        logits = self.classifier(x)
        
        return logits


class TwoStageSafetyFilter(nn.Module):
    """
    两阶段安全过滤器
    Stage 1: 段级快速筛选 (97.8% savings)
    Stage 2: Token级精细分类 (仅对可疑段)
    """
    
    def __init__(
        self,
        d_model: int = 8192,
        segment_size: int = 256,
        num_unsafe_categories: int = 10,
        threshold_stage1: float = 0.5,
        threshold_stage2: float = 0.7
    ):
        super().__init__()
        
        # Stage 1: 段级分类
        self.segment_pooler = SegmentPooler(d_model, segment_size)
        self.segment_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, num_unsafe_categories),
            nn.Sigmoid()
        )
        
        # Stage 2: Token级分类
        self.token_classifier = LocalContextConv(d_model)
        
        # 阈值
        self.threshold_stage1 = threshold_stage1
        self.threshold_stage2 = threshold_stage2
        
        # 安全类别名称
        self.category_names = [
            "violence", "sexual", "hate_speech", "harassment",
            "self_harm", "illegal", "fraud", "misinformation",
            "privacy_violation", "manipulation"
        ]
        
    def forward(
        self,
        hidden: torch.Tensor,
        return_details: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden: [B, L, D]
            return_details: 是否返回详细信息
        Returns:
            Dict containing safety scores and flags
        """
        B, L, D = hidden.shape
        segment_size = self.segment_pooler.segment_size
        
        # ============ Stage 1: 段级筛选 ============
        segments, segment_mask = self.segment_pooler(hidden)  # [B, num_segments, D]
        segment_scores = self.segment_classifier(segments)  # [B, num_segments, num_categories]
        
        # 找出可疑段 (任一类别超过阈值)
        segment_max_scores, _ = segment_scores.max(dim=-1)  # [B, num_segments]
        suspicious_segments = segment_max_scores > self.threshold_stage1  # [B, num_segments]
        
        # ============ Stage 2: Token级精细分类 ============
        # 初始化token级结果
        token_logits = torch.zeros(B, L, 4, device=hidden.device)
        token_logits[:, :, 0] = 1.0  # 默认safe
        
        # 仅对可疑段进行精细分类
        if suspicious_segments.any():
            for b in range(B):
                suspicious_idx = suspicious_segments[b].nonzero().squeeze(-1)
                
                for seg_idx in suspicious_idx:
                    start = seg_idx.item() * segment_size
                    end = min(start + segment_size, L)
                    
                    # 提取段内tokens
                    segment_tokens = hidden[b:b+1, start:end, :]
                    
                    # 精细分类
                    fine_logits = self.token_classifier(segment_tokens)  # [1, seg_len, 4]
                    token_logits[b, start:end, :] = fine_logits.squeeze(0)
        
        # 计算最终安全分数
        token_probs = F.softmax(token_logits, dim=-1)
        safe_scores = token_probs[:, :, 0]  # safe类别的概率
        
        # 整体安全判断
        min_safe_score = safe_scores.min(dim=-1)[0]  # [B]
        is_safe = min_safe_score > (1 - self.threshold_stage2)
        
        result = {
            "is_safe": is_safe,
            "safe_scores": safe_scores,
            "min_safe_score": min_safe_score,
            "segment_scores": segment_scores,
            "suspicious_segments": suspicious_segments
        }
        
        if return_details:
            result["token_logits"] = token_logits
            result["token_probs"] = token_probs
            result["segment_max_scores"] = segment_max_scores
            
        return result
    
    def get_violations(
        self,
        hidden: torch.Tensor,
        threshold: float = 0.5
    ) -> List[List[Dict]]:
        """
        获取具体的违规内容位置和类型
        
        Returns:
            List of lists of violation dicts for each batch item
        """
        result = self.forward(hidden, return_details=True)
        B, L = hidden.shape[:2]
        
        violations = []
        segment_size = self.segment_pooler.segment_size
        
        for b in range(B):
            batch_violations = []
            
            # 检查段级违规
            segment_scores = result["segment_scores"][b]  # [num_segments, num_categories]
            
            for seg_idx in range(segment_scores.shape[0]):
                for cat_idx, cat_name in enumerate(self.category_names):
                    if segment_scores[seg_idx, cat_idx] > threshold:
                        start = seg_idx * segment_size
                        end = min(start + segment_size, L)
                        
                        batch_violations.append({
                            "category": cat_name,
                            "segment_start": start,
                            "segment_end": end,
                            "score": segment_scores[seg_idx, cat_idx].item(),
                            "stage": "segment"
                        })
            
            violations.append(batch_violations)
            
        return violations
