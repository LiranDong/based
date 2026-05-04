# Adapted from train/metrics/perplexity.py in the Based codebase.
# Removed torchmetrics dependency. Kept the correct perplexity formula:
#   ppl = exp(average(nll)), not average(exp(nll))

import math
from typing import Optional

import torch


class Perplexity:
    """Standalone perplexity metric (no torchmetrics dependency).

    Computes perplexity as exp(average NLL) over all seen tokens.
    Correctly accumulates log-probabilities across batches.

    Usage:
        metric = Perplexity()
        for batch in dataloader:
            ...
            metric.update(loss, num_tokens)
        ppl = metric.compute()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset accumulated statistics."""
        self._total_nll = 0.0  # sum of negative log-likelihood * num_tokens
        self._total_tokens = 0

    def update(
        self,
        loss: torch.Tensor,
        num_tokens: int,
    ) -> None:
        """Accumulate statistics from one batch.

        Args:
            loss: Average cross-entropy loss for this batch (scalar tensor).
            num_tokens: Total number of tokens in this batch.
        """
        # loss is already average NLL per token
        self._total_nll += loss.detach().item() * num_tokens
        self._total_tokens += num_tokens

    def compute(self) -> float:
        """Compute perplexity from accumulated statistics.

        Returns:
            Perplexity value as a Python float.
        """
        if self._total_tokens == 0:
            return float("nan")
        avg_nll = self._total_nll / self._total_tokens
        return math.exp(avg_nll)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def avg_loss(self) -> float:
        if self._total_tokens == 0:
            return float("nan")
        return self._total_nll / self._total_tokens
