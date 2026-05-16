"""
Context Inflation Veto - 上下文膨胀否决机制
当 T'/T > 2.0 时，在生成前触发否决
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

class VetoReason(Enum):
    """否决原因"""
    CONTEXT_INFLATION = "context_inflation"
    MAX_ROUNDS = "max_rounds"
    SAFETY_VIOLATION = "safety_violation"
    COMPUTE_BUDGET = "compute_budget"
    QUALITY_DEGRADATION = "quality_degradation"

@dataclass
class VetoDecision:
    """否决决策"""
    should_veto: bool
    reason: Optional[VetoReason]
    ratio: float
    message: str

class ContextInflationVeto(nn.Module):
    """
    上下文膨胀否决器
    在生成前检查上下文膨胀比例
    """
    
    def __init__(
        self,
        inflation_threshold: float = 2.0,
        max_rounds: int = 10,
        compute_budget_tokens: int = 1_500_000,  # 1.5M
        quality_threshold: float = 0.3
    ):
        super().__init__()
        
        self.inflation_threshold = inflation_threshold
        self.max_rounds = max_rounds
        self.compute_budget_tokens = compute_budget_tokens
        self.quality_threshold = quality_threshold
        
        # 自适应阈值调整
        self.threshold_adapter = nn.Sequential(
            nn.Linear(3, 64),  # 3个上下文特征
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def compute_inflation_ratio(
        self,
        original_context_len: int,
        current_context_len: int
    ) -> float:
        """计算上下文膨胀比例"""
        if original_context_len == 0:
            return 1.0
        return current_context_len / original_context_len
        
    def get_context_features(
        self,
        original_len: int,
        current_len: int,
        num_tools_called: int
    ) -> torch.Tensor:
        """提取上下文特征用于自适应阈值"""
        ratio = self.compute_inflation_ratio(original_len, current_len)
        density = num_tools_called / max(current_len, 1)
        
        features = torch.tensor([
            ratio / 10.0,  # 归一化
            density * 100,
            num_tools_called / 100.0
        ])
        
        return features
        
    def check_veto(
        self,
        original_context_len: int,
        current_context_len: int,
        current_round: int,
        num_tools_called: int = 0,
        safety_score: Optional[float] = None,
        quality_score: Optional[float] = None,
        adaptive_threshold: bool = True
    ) -> VetoDecision:
        """
        检查是否应该否决继续执行
        
        Args:
            original_context_len: 原始上下文长度
            current_context_len: 当前上下文长度
            current_round: 当前轮次
            num_tools_called: 已调用工具次数
            safety_score: 安全分数
            quality_score: 质量分数
            adaptive_threshold: 是否使用自适应阈值
        Returns:
            VetoDecision: 否决决策
        """
        # 计算膨胀比例
        ratio = self.compute_inflation_ratio(original_context_len, current_context_len)
        
        # 确定阈值
        if adaptive_threshold:
            features = self.get_context_features(
                original_context_len, current_context_len, num_tools_called
            )
            # 自适应调整: 复杂任务允许更高阈值
            adjustment = self.threshold_adapter(features.unsqueeze(0)).item()
            threshold = self.inflation_threshold + 0.5 * adjustment
        else:
            threshold = self.inflation_threshold
        
        # 检查各种否决条件
        
        # 1. 上下文膨胀
        if ratio > threshold:
            return VetoDecision(
                should_veto=True,
                reason=VetoReason.CONTEXT_INFLATION,
                ratio=ratio,
                message=f"上下文膨胀超过阈值: {ratio:.2f} > {threshold:.2f}"
            )
        
        # 2. 最大轮次
        if current_round >= self.max_rounds:
            return VetoDecision(
                should_veto=True,
                reason=VetoReason.MAX_ROUNDS,
                ratio=ratio,
                message=f"达到最大轮次: {current_round} >= {self.max_rounds}"
            )
        
        # 3. 计算预算
        if current_context_len > self.compute_budget_tokens:
            return VetoDecision(
                should_veto=True,
                reason=VetoReason.COMPUTE_BUDGET,
                ratio=ratio,
                message=f"超过计算预算: {current_context_len} > {self.compute_budget_tokens}"
            )
        
        # 4. 安全违规
        if safety_score is not None and safety_score < 0.5:
            return VetoDecision(
                should_veto=True,
                reason=VetoReason.SAFETY_VIOLATION,
                ratio=ratio,
                message=f"安全分数过低: {safety_score:.2f} < 0.5"
            )
        
        # 5. 质量退化
        if quality_score is not None and quality_score < self.quality_threshold:
            return VetoDecision(
                should_veto=True,
                reason=VetoReason.QUALITY_DEGRADATION,
                ratio=ratio,
                message=f"质量分数过低: {quality_score:.2f} < {self.quality_threshold}"
            )
        
        # 不否决
        return VetoDecision(
            should_veto=False,
            reason=None,
            ratio=ratio,
            message="继续执行"
        )


class AgenticFlowController(nn.Module):
    """
    智能体流控制器
    整合否决机制和策略选择
    """
    
    def __init__(
        self,
        d_model: int = 8192,
        inflation_threshold: float = 2.0,
        max_rounds: int = 10
    ):
        super().__init__()
        
        self.veto = ContextInflationVeto(
            inflation_threshold=inflation_threshold,
            max_rounds=max_rounds
        )
        
        # 策略选择器
        self.strategy_selector = nn.Sequential(
            nn.Linear(d_model + 3, d_model // 4),  # +3 for context features
            nn.GELU(),
            nn.Linear(d_model // 4, 5),  # 5种策略
            nn.Softmax(dim=-1)
        )
        
        self.strategies = [
            "direct_answer",
            "tool_call",
            "decomposition",
            "clarification",
            "escalation"
        ]
        
    def forward(
        self,
        hidden: torch.Tensor,
        original_context_len: int,
        current_context_len: int,
        current_round: int,
        **kwargs
    ) -> Dict:
        """
        智能体流控制
        """
        # 首先检查否决
        veto_decision = self.veto.check_veto(
            original_context_len=original_context_len,
            current_context_len=current_context_len,
            current_round=current_round,
            **kwargs
        )
        
        if veto_decision.should_veto:
            return {
                "action": "veto",
                "veto_decision": veto_decision,
                "strategy": None
            }
        
        # 选择策略
        ratio = current_context_len / max(original_context_len, 1)
        context_features = torch.tensor([
            ratio / 10.0,
            current_round / 10.0,
            kwargs.get('num_tools_called', 0) / 10.0
        ], device=hidden.device)
        
        strategy_input = torch.cat([
            hidden.mean(dim=1),  # [B, D]
            context_features.unsqueeze(0).expand(hidden.shape[0], -1)
        ], dim=-1)
        
        strategy_probs = self.strategy_selector(strategy_input)
        strategy_idx = strategy_probs.argmax(dim=-1)
        
        return {
            "action": "continue",
            "veto_decision": veto_decision,
            "strategy": self.strategies[strategy_idx.item()],
            "strategy_probs": strategy_probs
        }
