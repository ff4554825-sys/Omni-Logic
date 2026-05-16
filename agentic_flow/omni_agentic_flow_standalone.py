"""
Omni Agentic Flow — Multi-round correction loop with context inflation veto.

Extracted from Omni-Logic v2.3 architecture.
Works at API level: no model internals needed, all scoring uses surface-level features.

Typical usage:
    from omni_agentic_flow import OmniCorrectionController, FormatChangeDetector

    controller = OmniCorrectionController()
    result = controller.run(candidates=["resp1","resp2","resp3","resp4"],
                            original_prompt="...",
                            context_tokens_before=500,
                            context_tokens_after=1200)
    print(result["best_response"])
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ──────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────


class VetoReason(Enum):
    CONTEXT_INFLATION = "context_inflation"
    MAX_ROUNDS = "max_rounds"
    COMPUTE_BUDGET = "compute_budget"


@dataclass
class VetoDecision:
    should_veto: bool
    reason: Optional[VetoReason] = None
    ratio: float = 1.0
    message: str = ""


# ──────────────────────────────────────────────
# Veto: Context inflation + max rounds + budget
# ──────────────────────────────────────────────


class ContextInflationVeto:
    """一票否决机制：生成前检查上下文膨胀、最大轮次、计算预算。"""

    def __init__(
        self,
        inflation_threshold: float = 2.0,
        max_rounds: int = 5,
        compute_budget_tokens: int = 1_500_000,
    ):
        self.inflation_threshold = inflation_threshold
        self.max_rounds = max_rounds
        self.compute_budget_tokens = compute_budget_tokens

    def check(
        self,
        original_context_len: int,
        current_context_len: int,
        current_round: int,
        cumulative_tokens_used: int = 0,
    ) -> VetoDecision:
        # Context inflation
        ratio = (
            current_context_len / original_context_len
            if original_context_len > 0
            else 1.0
        )
        if ratio > self.inflation_threshold:
            return VetoDecision(
                should_veto=True,
                reason=VetoReason.CONTEXT_INFLATION,
                ratio=ratio,
                message=f"Context inflation: T'/T = {ratio:.2f} > {self.inflation_threshold}",
            )

        # Max rounds
        if current_round >= self.max_rounds:
            return VetoDecision(
                should_veto=True,
                reason=VetoReason.MAX_ROUNDS,
                ratio=ratio,
                message=f"Max rounds reached: {current_round} >= {self.max_rounds}",
            )

        # Compute budget
        if cumulative_tokens_used >= self.compute_budget_tokens:
            return VetoDecision(
                should_veto=True,
                reason=VetoReason.COMPUTE_BUDGET,
                ratio=ratio,
                message=f"Compute budget exceeded: {cumulative_tokens_used} tokens",
            )

        return VetoDecision(should_veto=False, ratio=ratio)


# ──────────────────────────────────────────────
# Coarse scorer — statistical features only
# ──────────────────────────────────────────────


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token)."""
    return max(1, len(text) // 4)


def _word_set(text: str) -> set[str]:
    return set(text.lower().split())


class CoarseScorer:
    """
    粗评分器：使用表面统计特征，零模型依赖。
    6 特征：长度归一化、词汇多样性、熵方差、停用词比、句长方差、重复度。
    """

    def score(self, text: str, reference_text: str = "") -> dict[str, float]:
        tokens = _estimate_tokens(text)
        words = text.split()
        n_words = len(words)
        if n_words == 0:
            return self._default_stats()

        unique = _word_set(text)
        ref_unique = _word_set(reference_text) if reference_text else set()

        # 1. 长度归一化 (相对 prompt)
        length_norm = min(1.0, tokens / 8000)

        # 2. 词汇多样性 (type-token ratio)
        ttr = len(unique) / n_words
        # 3. 熵方差 (用字符频率代替 embedding)
        char_freq: dict[str, float] = {}
        for c in text:
            char_freq[c] = char_freq.get(c, 0) + 1
        total_chars = len(text) or 1
        entropy = -sum((c / total_chars) * math.log(c / total_chars + 1e-10) for c in char_freq.values())
        max_entropy = math.log(min(total_chars, 128))
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.5

        # 4. 停用词比 (简单停用词)
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
                      "at", "to", "for", "of", "with", "by", "and", "or", "but"}
        stop_ratio = sum(1 for w in words if w.lower() in stop_words) / n_words

        # 5. 句长方差
        sentences = [s.split() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        sent_lens = [len(s) for s in sentences]
        sent_var = statistics.pvariance(sent_lens) if len(sent_lens) > 1 else 0.0
        norm_sent_var = min(1.0, sent_var / 500)

        # 6. 与 reference 的词汇重合率
        overlap = 0.0
        if ref_unique:
            overlap = len(unique & ref_unique) / len(ref_unique | unique)

        return {
            "length_norm": length_norm,
            "ttr": ttr,
            "entropy": norm_entropy,
            "stop_ratio": stop_ratio,
            "sent_var": norm_sent_var,
            "overlap": overlap,
        }

    @staticmethod
    def _default_stats() -> dict[str, float]:
        return {k: 0.0 for k in ["length_norm", "ttr", "entropy", "stop_ratio", "sent_var", "overlap"]}


# ──────────────────────────────────────────────
# Fine scorer — semantic similarity
# ──────────────────────────────────────────────


class FineScorer:
    """
    精细评分器：使用 sentence-transformers 的语义相似度。
    fallback: 如果 sentence-transformers 不可用，用词汇重合度近似。
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self.model_name)
        except ImportError:
            self._model = None

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        self._load_model()
        if self._model is not None:
            emb = self._model.encode([text_a, text_b])
            sim = (emb[0] @ emb[1]) / (math.sqrt((emb[0]**2).sum()) * math.sqrt((emb[1]**2).sum()) + 1e-10)
            return float(max(0.0, min(1.0, sim)))
        # fallback: word overlap
        words_a = _word_set(text_a)
        words_b = _word_set(text_b)
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    def compute_quality_scores(self, candidate: str, reference: str = "") -> dict[str, float]:
        """
        返回 5 个质量维度: relevance, accuracy, completeness, coherence, safety
        基于启发式 + semantic similarity
        """
        if reference:
            sim = self.compute_similarity(candidate, reference)
        else:
            sim = 0.5

        words = candidate.split()
        n_words = len(words)

        # relevance: similarity to reference
        relevance = sim

        # accuracy: higher similarity = higher accuracy proxy
        accuracy = 0.5 + 0.5 * sim

        # completeness: longer tends to be more complete (capped)
        completeness = min(1.0, n_words / 300)

        # coherence: sentence length variance (moderate variance = good)
        sentences = [s for s in candidate.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        if len(sentences) > 1:
            sent_lens = [len(s.split()) for s in sentences]
            avg_len = statistics.mean(sent_lens)
            variance = statistics.pvariance(sent_lens) if len(sent_lens) > 1 else 0.0
            cv = math.sqrt(variance) / avg_len if avg_len > 0 else 1.0
            coherence = max(0.0, 1.0 - abs(cv - 0.6))
        else:
            coherence = 0.5

        # safety: heuristic — penalize very short or suspicious patterns
        safety = 1.0
        danger_patterns = ["ignore previous", "forget", "disregard", "system prompt", "jailbreak"]
        for p in danger_patterns:
            if p in candidate.lower():
                safety -= 0.2
        safety = max(0.0, safety)

        return {
            "relevance": round(relevance, 4),
            "accuracy": round(accuracy, 4),
            "completeness": round(completeness, 4),
            "coherence": round(coherence, 4),
            "safety": round(safety, 4),
        }

    def overall_score(self, candidate: str, reference: str = "") -> float:
        dims = self.compute_quality_scores(candidate, reference)
        return sum(dims.values()) / len(dims)


# ──────────────────────────────────────────────
# Format-only change detector
# ──────────────────────────────────────────────


class FormatChangeDetector:
    """
    检测"格式变化但语义不变"的假改进。
    当语义相似度高(≥0.95)但句法差异大时，标记为 format-only change。
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.scorer = FineScorer(model_name)

    def _levenshtein_ratio(self, a: str, b: str) -> float:
        """快速编辑距离比率 (字符级)。"""
        if not a and not b:
            return 1.0
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                prev = temp
        return 1.0 - dp[n] / max(m, n)

    def detect_format_only_change(self, text_a: str, text_b: str) -> dict[str, Any]:
        """
        返回检测结果。semantic_sim > 0.95 且 lexical_sim < 0.5 → format-only change。
        """
        semantic_sim = self.scorer.compute_similarity(text_a, text_b)
        lexical_sim = self._levenshtein_ratio(text_a, text_b)

        # Simple structural comparison
        sentences_a = text_a.replace("!", ".").replace("?", ".").split(".")
        sentences_b = text_b.replace("!", ".").replace("?", ".").split(".")
        n_sent_a = len([s for s in sentences_a if s.strip()])
        n_sent_b = len([s for s in sentences_b if s.strip()])

        is_format_only = (
            semantic_sim >= 0.95 and lexical_sim < 0.5
        )
        is_length_increase = len(text_b) > len(text_a) * 1.5 and semantic_sim > 0.90

        return {
            "is_format_only": is_format_only,
            "semantic_similarity": round(semantic_sim, 4),
            "lexical_similarity": round(lexical_sim, 4),
            "sentences_diff": abs(n_sent_a - n_sent_b),
            "length_inflation_warning": is_length_increase,
        }


# ──────────────────────────────────────────────
# Correction controller — orchestrates the loop
# ──────────────────────────────────────────────


@dataclass
class CorrectionRound:
    round_num: int
    candidates: list[str]
    coarse_scores: list[dict[str, float]]
    fine_scores: list[float]
    best_text: str
    best_score: float
    quality_dims: dict[str, float]
    veto_result: VetoDecision
    format_change: Optional[dict[str, Any]] = None


class OmniCorrectionController:
    """
    完整的多轮修正控制器。

    Generate(N candidates) → CoarseScore(×N) → FineScore(top-2) →
        ≥threshold? → Output | InjectCorrection → repeat
    """

    def __init__(
        self,
        n_candidates: int = 4,
        threshold_weights: Optional[dict[str, float]] = None,
        round_targets: Optional[list[float]] = None,
        max_rounds: int = 5,
        inflation_ratio_limit: float = 2.0,
        exit_weight_sum: float = 0.6,
        coarse_filter_top_k: int = 2,
        similarity_model: str = "all-MiniLM-L6-v2",
    ):
        self.n_candidates = n_candidates
        self.round_targets = round_targets or [0.70, 0.62, 0.54, 0.46, 0.38]
        self.max_rounds = max_rounds
        self.combined_exit_threshold = exit_weight_sum
        self.coarse_filter_top_k = coarse_filter_top_k
        self.threshold_weights = threshold_weights or {
            "threshold": 0.4,
            "improvement": 0.3,
            "uncertainty": 0.2,
            "budget": 0.1,
        }

        self.veto = ContextInflationVeto(
            inflation_threshold=inflation_ratio_limit,
            max_rounds=max_rounds,
        )
        self.coarse = CoarseScorer()
        self.fine = FineScorer(similarity_model)
        self.detector = FormatChangeDetector(similarity_model)

        # State
        self.history: list[str] = []
        self.history_scores: list[float] = []
        self.rounds: list[CorrectionRound] = []
        self.cumulative_compute = 0.0

    def _estimate_compute(self, text: str) -> float:
        """归一化计算开销估算 (粗糙)。"""
        return _estimate_tokens(text) / 1000

    def _compute_weighted_exit(
        self,
        current_score: float,
        round_idx: int,
        prev_score: Optional[float],
        uncertainty: float,
    ) -> tuple[bool, float]:
        """计算 4 策略加权退出决定。返回 (should_exit, weighted_sum)。"""
        signals: dict[str, float] = {}

        # 1. Threshold signal
        target = self.round_targets[round_idx] if round_idx < len(self.round_targets) else 0.30
        signals["threshold"] = 1.0 if current_score >= target else 0.0

        # 2. Improvement signal
        if prev_score is not None:
            improvement = current_score - prev_score
            signals["improvement"] = 1.0 if improvement < 0.03 else 0.0
        else:
            signals["improvement"] = 0.0

        # 3. Uncertainty signal (confidence)
        signals["uncertainty"] = 1.0 if uncertainty < 0.15 else 0.0

        # 4. Budget signal
        budget_ratio = self.cumulative_compute / 4.0  # 4.0 = normalized budget
        signals["budget"] = 1.0 if budget_ratio >= 1.0 else 0.0

        weighted_sum = sum(
            signals[k] * self.threshold_weights.get(k, 0.0)
            for k in signals
        )
        return weighted_sum >= self.combined_exit_threshold, weighted_sum

    def run(
        self,
        candidates: list[str],
        original_prompt: str,
        context_tokens_before: int = 0,
        context_tokens_after: int = 0,
        reference_answer: str = "",
    ) -> dict[str, Any]:
        """
        执行多轮修正循环。

        Args:
            candidates: 初始生成的 N 个候选回答
            original_prompt: 原始 prompt
            context_tokens_before: 初始上下文 token 数
            context_tokens_after: 当前上下文 token 数
            reference_answer: 可选的参考答案 (如果有)

        Returns:
            dict with keys: best_response, rounds, exit_reason, final_score,
                            quality_dims, all_candidates, scores, round_details
        """
        self.history = [original_prompt]
        self.history_scores = []
        self.rounds = []
        self.cumulative_compute = 0.0

        round_idx = 0
        best_overall = ""
        best_overall_score = -1.0
        best_overall_dims: dict[str, float] = {}

        while round_idx < self.max_rounds:
            # Veto check
            original_len = context_tokens_before or _estimate_tokens(original_prompt)
            current_len = context_tokens_after or (original_len + sum(
                _estimate_tokens(c) for c in candidates
            ))
            veto_result = self.veto.check(
                original_context_len=original_len,
                current_context_len=current_len,
                current_round=round_idx,
                cumulative_tokens_used=int(self.cumulative_compute * 1000),
            )

            if veto_result.should_veto:
                break

            # Coarse scoring
            coarse_scores = [
                self.coarse.score(c, original_prompt)
                for c in candidates
            ]

            # Coarse filter: sort by composite heuristic score and take top-k
            def _composite(stats: dict[str, float]) -> float:
                # Scores: longer normalized length, better entropy, moderate stop-ratio
                return (
                    stats["length_norm"] * 0.2
                    + stats["ttr"] * 0.3
                    + stats["entropy"] * 0.3
                    - min(stats["stop_ratio"], 0.7) * 0.1
                    + stats["sent_var"] * 0.1
                )

            ranked = sorted(
                zip(candidates, coarse_scores),
                key=lambda x: _composite(x[1]),
                reverse=True,
            )
            top_candidates = [r[0] for r in ranked[: self.coarse_filter_top_k]]

            # Fine scoring on top-k
            fine_scores = []
            quality_dim_list = []
            for cand in top_candidates:
                dims = self.fine.compute_quality_scores(cand, reference_answer)
                score = sum(dims.values()) / len(dims)
                fine_scores.append(score)
                quality_dim_list.append(dims)

            # Pick best
            best_idx = fine_scores.index(max(fine_scores))
            best_text = top_candidates[best_idx]
            best_score = fine_scores[best_idx]
            best_dims = quality_dim_list[best_idx]

            self.history.append(best_text)
            self.history_scores.append(best_score)

            # Format-only detection
            format_change = None
            if len(self.history) >= 2 and len(self.history_scores) >= 2:
                format_change = self.detector.detect_format_only_change(
                    self.history[-2],
                    self.history[-1],
                )

            # Track overall best
            if best_score > best_overall_score:
                best_overall = best_text
                best_overall_score = best_score
                best_overall_dims = best_dims

            # Estimated compute
            self.cumulative_compute += sum(
                self._estimate_compute(c) for c in candidates
            )

            # Compute uncertainty: std between top-2 candidate scores
            uncertainty = (
                statistics.stdev(fine_scores) if len(fine_scores) > 1 else 0.5
            )

            # Weighted exit check
            prev_score = self.history_scores[-2] if len(self.history_scores) >= 2 else None
            should_exit, weighted_sum = self._compute_weighted_exit(
                best_score, round_idx, prev_score, uncertainty,
            )

            self.rounds.append(CorrectionRound(
                round_num=round_idx,
                candidates=candidates,
                coarse_scores=coarse_scores,
                fine_scores=fine_scores,
                best_text=best_text,
                best_score=best_score,
                quality_dims=best_dims,
                veto_result=veto_result,
                format_change=format_change,
            ))

            if should_exit:
                break

            if round_idx >= self.max_rounds - 1:
                break

            # Prepare for next round: replace candidates with correction prompts
            # (In practice, this would be an LLM call with correction instruction)
            # Here we just reuse candidates as placeholder
            candidates = top_candidates  # Keep best candidates
            round_idx += 1

        return {
            "best_response": best_overall,
            "rounds": len(self.rounds),
            "exit_reason": (
                veto_result.message
                if veto_result.should_veto
                else f"Weighted exit (sum={weighted_sum:.2f})"
                if len(self.rounds) < self.max_rounds
                else "Max rounds"
            ),
            "final_score": best_overall_score,
            "quality_dims": best_overall_dims,
            "all_candidates": self.history,
            "scores": self.history_scores,
            "round_details": [
                {
                    "round": r.round_num,
                    "best_score": r.best_score,
                    "quality": r.quality_dims,
                    "veto": r.veto_result.reason.value if r.veto_result.reason else None,
                    "format_only": r.format_change["is_format_only"]
                    if r.format_change else False,
                }
                for r in self.rounds
            ],
        }


# ──────────────────────────────────────────────
# Convenience high-level function
# ──────────────────────────────────────────────


def judge_and_correct(
    prompt: str,
    responses: list[str],
    reference: str = "",
) -> dict[str, Any]:
    """
    快速入口：给一个 prompt 和一组响应，返回评分和修正建议。
    """
    controller = OmniCorrectionController(n_candidates=len(responses))
    result = controller.run(
        candidates=responses,
        original_prompt=prompt,
        reference_answer=reference,
        context_tokens_before=len(prompt) // 4,
        context_tokens_after=sum(len(r) // 4 for r in responses),
    )
    return result


# ──────────────────────────────────────────────
# Diagnostic / demo
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Demo with dummy responses
    prompt = "Explain how transformers work in one paragraph."
    responses = [
        "Transformers use self-attention to process sequences. They have encoder-decoder architecture with multi-head attention layers.",
        "Transformers are neural networks that use attention mechanisms. The key innovation is self-attention which allows parallel processing of sequences. They have multiple layers with feed-forward networks.",
        "A transformer is a deep learning architecture based on attention. It was introduced in 'Attention Is All You Need'. It uses multi-head attention and positional encoding.",
    ]

    print("=== Omni Agentic Flow Demo ===")
    result = judge_and_correct(prompt, responses, reference=prompt)
    print(f"Best response (round {result['rounds']}):")
    print(f"  {result['best_response'][:100]}...")
    print(f"Score: {result['final_score']:.4f}")
    print(f"Exit reason: {result['exit_reason']}")
    print(f"Quality: {result['quality_dims']}")
    print(f"\nRound details:")
    for rd in result["round_details"]:
        print(f"  Round {rd['round']}: score={rd['best_score']:.4f}, "
              f"format_only={rd['format_only']}, veto={rd['veto']}")
