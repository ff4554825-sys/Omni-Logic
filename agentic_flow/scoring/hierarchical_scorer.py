"""
Hierarchical Scorer - 分层评分系统
Coarse Scorer (0.15B) + Fine Scorer (0.5B) = 0.65B total
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math

class EmbeddingEntropyCalculator(nn.Module):
    """
    基于embedding的熵计算（替代attention entropy）
    不依赖attention map，直接在hidden states上计算
    """
    
    def __init__(self, num_bins: int = 32):
        super().__init__()
        self.num_bins = num_bins
        
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [B, L, D] hidden states
        Returns:
            entropy: [B] 标量熵值
        """
        # 计算每个token到均值的距离
        mean_emb = hidden.mean(dim=1, keepdim=True)  # [B, 1, D]
        distances = torch.norm(hidden - mean_emb, dim=-1)  # [B, L]
        
        # 转换为概率分布
        probs = torch.softmax(-distances, dim=-1)  # [B, L]
        
        # 计算熵
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [B]
        
        # 归一化到[0, 1]
        max_entropy = math.log(hidden.shape[1])
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy


class CoarseScorer(nn.Module):
    """
    粗评分器：0.15B参数
    使用统计特征快速筛选候选
    """
    
    def __init__(
        self,
        d_model: int = 8192,
        hidden_dim: int = 2048,
        num_features: int = 16,  # 统计特征数量
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # 统计特征提取
        self.feature_extractors = nn.ModuleDict({
            'length_norm': nn.Identity(),  # 序列长度归一化
            'entropy': EmbeddingEntropyCalculator(),
            'variance': nn.Identity(),  # 方差
            'sparsity': nn.Identity(),  # 稀疏度
        })
        
        # 轻模态统计聚合
        self.modality_stats = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_features)
        )
        
        # 快速评分MLP
        self.score_mlp = nn.Sequential(
            nn.Linear(num_features + 4, hidden_dim),  # 4个统计特征
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 参数量: ~0.15B
        # d_model * hidden_dim + hidden_dim * num_features + (num_features+4) * hidden_dim + ... ≈ 0.15B
        
    def extract_statistics(self, hidden: torch.Tensor) -> torch.Tensor:
        """提取统计特征"""
        B, L, D = hidden.shape
        
        # 1. Embedding熵
        entropy = self.feature_extractors['entropy'](hidden)  # [B]
        
        # 2. 方差
        variance = hidden.var(dim=(1, 2))  # [B]
        variance = torch.log1p(variance) / 10  # 归一化
        
        # 3. 稀疏度 (L0近似)
        sparsity = (hidden.abs() < 0.01).float().mean(dim=(1, 2))  # [B]
        
        # 4. 序列长度归一化
        length_norm = torch.log1p(torch.tensor(L, dtype=torch.float32)) / 10
        length_norm = length_norm.expand(B).to(hidden.device)
        
        return torch.stack([entropy, variance, sparsity, length_norm], dim=-1)  # [B, 4]
        
    def forward(
        self,
        hidden: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            hidden: [B, L, D]
            modality_mask: [B, L] 可选的模态mask
        Returns:
            scores: [B] 粗评分
            stats: 统计信息字典
        """
        # 提取统计特征
        stats_features = self.extract_statistics(hidden)  # [B, 4]
        
        # 跨模态统计聚合
        if modality_mask is not None:
            masked_hidden = hidden * modality_mask.unsqueeze(-1)
        else:
            masked_hidden = hidden
            
        pooled = masked_hidden.mean(dim=1)  # [B, D]
        modality_features = self.modality_stats(pooled)  # [B, num_features]
        
        # 拼接特征
        combined = torch.cat([modality_features, stats_features], dim=-1)  # [B, num_features+4]
        
        # 快速评分
        scores = self.score_mlp(combined).squeeze(-1)  # [B]
        
        stats = {
            "entropy": stats_features[:, 0],
            "variance": stats_features[:, 1],
            "sparsity": stats_features[:, 2],
            "length_norm": stats_features[:, 3]
        }
        
        return scores, stats


class FineScorer(nn.Module):
    """
    精细评分器：0.5B参数
    使用跨注意力进行细粒度评估
    """
    
    def __init__(
        self,
        d_model: int = 8192,
        num_heads: int = 16,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # 跨注意力层
        self.cross_attn_layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                ),
                'norm1': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model)
                ),
                'norm2': nn.LayerNorm(d_model)
            })
            for _ in range(num_layers)
        ])
        
        # 评分头
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # 参数量: ~0.5B
        # num_layers * (d_model^2 * 4 + d_model * d_model*4 * 2) + d_model * d_model/2 + ... ≈ 0.5B
        
    def forward(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        coarse_scores: torch.Tensor,
        threshold: float = 0.3
    ) -> torch.Tensor:
        """
        Args:
            hidden: [B, L, D] 当前表示
            context: [B, L_ctx, D] 上下文
            coarse_scores: [B] 粗评分
            threshold: 粗评分阈值，低于此值的不进行精细评分
        Returns:
            fine_scores: [B] 精细评分
        """
        B = hidden.shape[0]
        
        # 根据粗评分筛选
        mask = coarse_scores > threshold
        
        # 初始化精细评分为粗评分
        fine_scores = coarse_scores.clone()
        
        if not mask.any():
            return fine_scores
            
        # 对通过粗筛选的样本进行精细评分
        hidden_filtered = hidden[mask]
        context_filtered = context[mask]
        
        # 跨注意力处理
        x = hidden_filtered
        for layer in self.cross_attn_layers:
            # 跨注意力
            attn_out, _ = layer['cross_attn'](
                query=x,
                key=context_filtered,
                value=context_filtered
            )
            x = layer['norm1'](x + attn_out)
            
            # FFN
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + ffn_out)
        
        # 池化并评分
        pooled = x.mean(dim=1)  # [B', D]
        scores_filtered = self.score_head(pooled).squeeze(-1)  # [B']
        
        # 合并结果
        fine_scores[mask] = scores_filtered
        
        return fine_scores


class HierarchicalScorer(nn.Module):
    """
    分层评分器：粗评分 + 精细评分
    总参数量：0.15B + 0.5B = 0.65B
    """
    
    def __init__(
        self,
        d_model: int = 8192,
        coarse_threshold: float = 0.3,
        **kwargs
    ):
        super().__init__()
        
        self.coarse_scorer = CoarseScorer(d_model=d_model)
        self.fine_scorer = FineScorer(d_model=d_model)
        self.coarse_threshold = coarse_threshold
        
        # 自适应阈值调整
        self.threshold_adapter = nn.Sequential(
            nn.Linear(4, 64),  # 4个统计特征
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
        adaptive_threshold: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden: [B, L, D]
            context: [B, L_ctx, D]
            modality_mask: [B, L]
            adaptive_threshold: 是否使用自适应阈值
        Returns:
            Dict containing scores and statistics
        """
        # 粗评分
        coarse_scores, stats = self.coarse_scorer(hidden, modality_mask)
        
        # 自适应阈值
        if adaptive_threshold:
            stats_tensor = torch.stack([
                stats['entropy'],
                stats['variance'],
                stats['sparsity'],
                stats['length_norm']
            ], dim=-1)
            threshold = self.threshold_adapter(stats_tensor).squeeze(-1)
            threshold = 0.2 + 0.2 * threshold  # 映射到[0.2, 0.4]
        else:
            threshold = self.coarse_threshold
            
        # 精细评分
        fine_scores = self.fine_scorer(hidden, context, coarse_scores, threshold)
        
        return {
            "scores": fine_scores,
            "coarse_scores": coarse_scores,
            "threshold": threshold,
            "statistics": stats
        }
