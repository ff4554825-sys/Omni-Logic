"""
ModalityAligner - 多模态对齐模块
将不同模态的输入映射到统一的语义空间
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class ModalityAligner(nn.Module):
    """
    多模态对齐器：支持文本、图像、音频、视频四种模态
    输出统一维度：d_model = 8192
    """
    
    def __init__(
        self,
        d_model: int = 8192,
        text_dim: int = 4096,      # 来自文本embedding
        image_dim: int = 768,      # 来自ViT
        audio_dim: int = 1280,     # 来自Whisper
        video_dim: int = 1024,     # 来自Video Encoder
        num_heads: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # 各模态投影层
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 跨模态注意力对齐
        self.cross_modal_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 模态类型embedding
        self.modality_embedding = nn.Embedding(4, d_model)  # 0:text, 1:image, 2:audio, 3:video
        
        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        text_emb: Optional[torch.Tensor] = None,
        image_emb: Optional[torch.Tensor] = None,
        audio_emb: Optional[torch.Tensor] = None,
        video_emb: Optional[torch.Tensor] = None,
        return_modality_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            text_emb: [B, L_text, text_dim]
            image_emb: [B, L_image, image_dim] 
            audio_emb: [B, L_audio, audio_dim]
            video_emb: [B, L_video, video_dim]
            
        Returns:
            Dict containing:
                - aligned: [B, L_total, d_model] 统一表示
                - modality_mask: 各模态的mask
        """
        aligned_features = []
        modality_ids = []
        
        # 投影各模态
        if text_emb is not None:
            text_aligned = self.text_proj(text_emb)
            text_aligned = text_aligned + self.modality_embedding(
                torch.zeros(text_emb.shape[1], dtype=torch.long, device=text_emb.device)
            )
            aligned_features.append(text_aligned)
            modality_ids.append(0)
            
        if image_emb is not None:
            image_aligned = self.image_proj(image_emb)
            image_aligned = image_aligned + self.modality_embedding(
                torch.ones(image_emb.shape[1], dtype=torch.long, device=image_emb.device)
            )
            aligned_features.append(image_aligned)
            modality_ids.append(1)
            
        if audio_emb is not None:
            audio_aligned = self.audio_proj(audio_emb)
            audio_aligned = audio_aligned + self.modality_embedding(
                torch.full((audio_emb.shape[1],), 2, dtype=torch.long, device=audio_emb.device)
            )
            aligned_features.append(audio_aligned)
            modality_ids.append(2)
            
        if video_emb is not None:
            video_aligned = self.video_proj(video_emb)
            video_aligned = video_aligned + self.modality_embedding(
                torch.full((video_emb.shape[1],), 3, dtype=torch.long, device=video_emb.device)
            )
            aligned_features.append(video_aligned)
            modality_ids.append(3)
        
        # 拼接所有模态
        concat_features = torch.cat(aligned_features, dim=1)  # [B, L_total, d_model]
        
        # 跨模态注意力精炼
        refined, _ = self.cross_modal_attn(
            query=concat_features,
            key=concat_features,
            value=concat_features
        )
        
        # 门控融合原始和对齐后的特征
        gate_input = torch.cat([concat_features, refined], dim=-1)
        gate_weights = self.gate(gate_input)
        output = gate_weights * concat_features + (1 - gate_weights) * refined
        
        result = {
            "aligned": output,
            "modality_ids": modality_ids,
            "sequence_lengths": [f.shape[1] for f in aligned_features]
        }
        
        if return_modality_features:
            result["modality_features"] = aligned_features
            
        return result


class ModalityTokenizer(nn.Module):
    """将连续特征转换为离散token"""
    
    def __init__(self, d_model: int = 8192, vocab_size: int = 128000):
        super().__init__()
        self.codebook = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, L, d_model]
        Returns:
            tokens: [B, L] 离散token
            reconstructed: [B, L, d_model] 重建特征
        """
        logits = self.proj(features)
        tokens = logits.argmax(dim=-1)
        reconstructed = self.codebook(tokens)
        return tokens, reconstructed
