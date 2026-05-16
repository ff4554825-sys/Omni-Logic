"""
GRPO with Anti-Reward Hacking - 带反奖励黑客机制的GRPO训练
包含：动态权重调整、4信号检测、对抗验证集
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import deque

@dataclass
class RewardHackingSignal:
    """奖励黑客信号"""
    name: str
    value: float
    threshold: float
    is_triggered: bool

class RewardHackingDetector(nn.Module):
    """
    奖励黑客检测器
    检测4种信号：
    1. 格式变化但语义不变 (format_only_change)
    2. 奖励上升但下游任务性能下降
    3. 输出分布异常集中
    4. KL散度过大
    """
    
    def __init__(
        self,
        d_model: int = 8192,
        semantic_encoder: str = 'all-MiniLM-L6-v2',
        kl_threshold: float = 0.5,
        format_change_threshold: float = 0.1,
        distribution_entropy_threshold: float = 0.3
    ):
        super().__init__()
        
        # 独立语义编码器 (避免循环验证)
        self.semantic_encoder_name = semantic_encoder
        # 注: 实际使用时需要加载预训练模型
        
        # 格式变化检测器
        self.format_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # 阈值
        self.kl_threshold = kl_threshold
        self.format_change_threshold = format_change_threshold
        self.distribution_entropy_threshold = distribution_entropy_threshold
        
        # 历史记录
        self.reward_history = deque(maxlen=100)
        self.task_performance_history = deque(maxlen=100)
        
    def compute_semantic_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        计算两个文本的语义相似度
        使用独立的语义编码器
        """
        # 注: 实际实现需要加载sentence-transformers
        # from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer(self.semantic_encoder_name)
        # emb1 = model.encode(text1)
        # emb2 = model.encode(text2)
        # return cosine_similarity(emb1, emb2)
        raise NotImplementedError("需要加载sentence-transformers库")
        
    def detect_format_only_change(
        self,
        original: torch.Tensor,
        modified: torch.Tensor,
        original_text: str,
        modified_text: str
    ) -> RewardHackingSignal:
        """
        检测格式变化但语义不变
        """
        # 计算表示变化
        repr_change = torch.norm(modified - original) / torch.norm(original)
        
        # 计算语义相似度
        try:
            semantic_sim = self.compute_semantic_similarity(original_text, modified_text)
        except NotImplementedError:
            semantic_sim = 1.0  # 默认假设语义不变
            
        # 格式变化但语义不变
        format_only = repr_change.item() > self.format_change_threshold and semantic_sim > 0.95
        
        return RewardHackingSignal(
            name="format_only_change",
            value=repr_change.item(),
            threshold=self.format_change_threshold,
            is_triggered=format_only
        )
        
    def detect_reward_task_divergence(
        self,
        current_reward: float,
        task_performance: float
    ) -> RewardHackingSignal:
        """
        检测奖励上升但任务性能下降
        """
        self.reward_history.append(current_reward)
        self.task_performance_history.append(task_performance)
        
        if len(self.reward_history) < 10:
            return RewardHackingSignal(
                name="reward_task_divergence",
                value=0.0,
                threshold=0.0,
                is_triggered=False
            )
        
        # 计算趋势
        reward_trend = torch.tensor(list(self.reward_history)).diff().mean().item()
        perf_trend = torch.tensor(list(self.task_performance_history)).diff().mean().item()
        
        # 奖励上升但性能下降
        divergence = reward_trend > 0 and perf_trend < 0
        
        return RewardHackingSignal(
            name="reward_task_divergence",
            value=reward_trend - perf_trend,
            threshold=0.0,
            is_triggered=divergence
        )
        
    def detect_output_distribution_collapse(
        self,
        output_logits: torch.Tensor
    ) -> RewardHackingSignal:
        """
        检测输出分布异常集中
        """
        probs = F.softmax(output_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        
        # 归一化熵
        max_entropy = torch.log(torch.tensor(output_logits.shape[-1], dtype=torch.float))
        normalized_entropy = entropy / max_entropy
        
        # 熵过低表示分布集中
        collapsed = normalized_entropy.item() < self.distribution_entropy_threshold
        
        return RewardHackingSignal(
            name="distribution_collapse",
            value=normalized_entropy.item(),
            threshold=self.distribution_entropy_threshold,
            is_triggered=collapsed
        )
        
    def detect_kl_divergence(
        self,
        current_policy: torch.Tensor,
        reference_policy: torch.Tensor
    ) -> RewardHackingSignal:
        """
        检测KL散度过大
        """
        kl = F.kl_div(
            F.log_softmax(current_policy, dim=-1),
            F.softmax(reference_policy, dim=-1),
            reduction='batchmean'
        )
        
        return RewardHackingSignal(
            name="kl_divergence",
            value=kl.item(),
            threshold=self.kl_threshold,
            is_triggered=kl.item() > self.kl_threshold
        )
        
    def forward(
        self,
        original_output: torch.Tensor,
        current_output: torch.Tensor,
        output_logits: torch.Tensor,
        current_policy: torch.Tensor,
        reference_policy: torch.Tensor,
        current_reward: float,
        task_performance: float,
        original_text: str = "",
        modified_text: str = ""
    ) -> Dict[str, RewardHackingSignal]:
        """
        检测所有奖励黑客信号
        """
        signals = {}
        
        signals["format_only_change"] = self.detect_format_only_change(
            original_output, current_output, original_text, modified_text
        )
        
        signals["reward_task_divergence"] = self.detect_reward_task_divergence(
            current_reward, task_performance
        )
        
        signals["distribution_collapse"] = self.detect_output_distribution_collapse(
            output_logits
        )
        
        signals["kl_divergence"] = self.detect_kl_divergence(
            current_policy, reference_policy
        )
        
        return signals


class DynamicWeightAdapter(nn.Module):
    """
    动态权重调整器
    根据训练阶段和检测信号调整GRPO权重
    """
    
    def __init__(
        self,
        initial_weights: Dict[str, float] = None,
        min_weights: Dict[str, float] = None,
        max_weights: Dict[str, float] = None
    ):
        super().__init__()
        
        self.weight_names = ['reward', 'kl', 'entropy', 'safety']
        
        # 默认权重
        self.initial_weights = initial_weights or {
            'reward': 1.0,
            'kl': 0.1,
            'entropy': 0.01,
            'safety': 0.5
        }
        
        self.min_weights = min_weights or {
            'reward': 0.5,
            'kl': 0.01,
            'entropy': 0.001,
            'safety': 0.1
        }
        
        self.max_weights = max_weights or {
            'reward': 2.0,
            'kl': 0.5,
            'entropy': 0.1,
            'safety': 1.0
        }
        
        # 当前权重 (可学习参数)
        self.weights = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(w), requires_grad=False)
            for name, w in self.initial_weights.items()
        })
        
        # 调整历史
        self.adjustment_history = []
        
    def adjust(
        self,
        signals: Dict[str, RewardHackingSignal],
        training_progress: float  # 0到1
    ) -> Dict[str, float]:
        """
        根据检测信号调整权重
        
        Args:
            signals: 奖励黑客检测信号
            training_progress: 训练进度
        Returns:
            调整后的权重
        """
        adjusted = {}
        
        for name in self.weight_names:
            base_weight = self.weights[name].item()
            
            # 根据信号调整
            if name == 'kl' and signals.get('kl_divergence', None):
                # KL过大时增加KL惩罚
                if signals['kl_divergence'].is_triggered:
                    base_weight *= 1.5
                    
            if name == 'entropy' and signals.get('distribution_collapse', None):
                # 分布坍缩时增加熵奖励
                if signals['distribution_collapse'].is_triggered:
                    base_weight *= 2.0
                    
            if name == 'reward' and signals.get('reward_task_divergence', None):
                # 奖励-任务发散时降低奖励权重
                if signals['reward_task_divergence'].is_triggered:
                    base_weight *= 0.7
                    
            # 根据训练进度调整
            # 后期更保守
            progress_factor = 1.0 - 0.3 * training_progress
            base_weight *= progress_factor
            
            # 限制范围
            base_weight = max(self.min_weights[name], min(self.max_weights[name], base_weight))
            adjusted[name] = base_weight
            
        # 记录调整
        self.adjustment_history.append({
            'weights': adjusted.copy(),
            'signals': {k: v.is_triggered for k, v in signals.items()},
            'progress': training_progress
        })
        
        return adjusted


class GRPOAntiHacking(nn.Module):
    """
    带反奖励黑客机制的GRPO训练器
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        reward_model: nn.Module,
        d_model: int = 8192,
        adversarial_set_size: int = 1000
    ):
        super().__init__()
        
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.reward_model = reward_model
        
        # 检测器
        self.hacking_detector = RewardHackingDetector(d_model)
        self.weight_adapter = DynamicWeightAdapter()
        
        # 对抗验证集
        self.adversarial_set = []
        self.adversarial_set_size = adversarial_set_size
        
    def generate_adversarial_samples(
        self,
        prompts: List[str],
        num_samples: int = 100
    ) -> List[Tuple[str, str, float]]:
        """
        生成对抗样本
        用于检测模型是否通过格式操纵来欺骗奖励模型
        """
        adversarial_samples = []
        
        for prompt in prompts[:num_samples]:
            # 生成正常输出
            with torch.no_grad():
                normal_output = self.policy_model.generate(prompt)
                
            # 尝试生成格式变体
            # (实际实现需要更复杂的逻辑)
            
            adversarial_samples.append((prompt, normal_output, 0.0))
            
        return adversarial_samples
        
    def compute_grpo_loss(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        rewards: torch.Tensor,
        weights: Dict[str, float]
    ) -> torch.Tensor:
        """
        计算GRPO损失
        """
        # 策略对数概率
        policy_log_probs = self.policy_model.get_log_probs(inputs, outputs)
        ref_log_probs = self.reference_model.get_log_probs(inputs, outputs)
        
        # 重要性权重
        ratio = torch.exp(policy_log_probs - ref_log_probs)
        
        # PPO裁剪
        epsilon = 0.2
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        
        # 策略损失
        policy_loss = -torch.min(
            ratio * rewards,
            clipped_ratio * rewards
        ).mean()
        
        # KL散度
        kl_loss = (policy_log_probs - ref_log_probs).mean()
        
        # 熵奖励
        entropy = -(policy_log_probs.exp() * policy_log_probs).sum(dim=-1).mean()
        
        # 总损失
        total_loss = (
            weights['reward'] * policy_loss +
            weights['kl'] * kl_loss -
            weights['entropy'] * entropy
        )
        
        return total_loss
        
    def forward(
        self,
        inputs: torch.Tensor,
        training_progress: float
    ) -> Dict[str, torch.Tensor]:
        """
        GRPO训练步骤
        """
        # 生成输出
        outputs = self.policy_model(inputs)
        
        # 计算奖励
        rewards = self.reward_model(inputs, outputs)
        
        # 检测奖励黑客
        ref_outputs = self.reference_model(inputs)
        
        signals = self.hacking_detector(
            original_output=ref_outputs,
            current_output=outputs,
            output_logits=outputs,
            current_policy=outputs,
            reference_policy=ref_outputs,
            current_reward=rewards.mean().item(),
            task_performance=0.0  # 需要从外部获取
        )
        
        # 调整权重
        weights = self.weight_adapter.adjust(signals, training_progress)
        
        # 计算损失
        loss = self.compute_grpo_loss(inputs, outputs, rewards, weights)
        
        return {
            "loss": loss,
            "rewards": rewards,
            "signals": signals,
            "weights": weights
        }
