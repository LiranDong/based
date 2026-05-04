"""Cloze-completion evaluation for SWDE, FDA, and SQuAD-Completion tasks.

Exact recreation of the original Based paper evaluation methodology:
  - Task format: long context + key/attribute name → model completes with value
  - Generation: greedy (do_sample=False), max_new_tokens=48, stop at first "\\n"
  - Scoring: contains_score — case-insensitive substring match
  - Aggregation: mean over all examples (= accuracy)
  - 0-shot (no training examples)

Dataset sources on HuggingFace:
  - hazyresearch/based-swde (1,111 examples)
  - hazyresearch/based-fda  (FDA 510k documents)
  - hazyresearch/based-squad (2,984 examples)

Reference: EleutherAI/lm-evaluation-harness PR #1728 (simran-arora)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# ── Task registry ──────────────────────────────────────────────────────

TASK_REGISTRY = {
    "swde": {
        "dataset_name": "hazyresearch/based-swde-v2",
        "dataset_config": "default",
        "split": "validation",
        "description": "SWDE — Structured Web Data Extraction (semi-structured HTML)",
    },
    "fda": {
        "dataset_name": "hazyresearch/based-fda",
        "dataset_config": "default",
        "split": "validation",
        "description": "FDA — Information extraction from FDA 510(k) documents",
    },
    "squad_completion": {
        "dataset_name": "hazyresearch/based-squad",
        "dataset_config": "default",
        "split": "validation",
        "description": "SQuAD-Completion — Document QA reformatted as next-token prediction",
    },
}


# ── Scoring ────────────────────────────────────────────────────────────

def contains_score(prediction: str, labels: list) -> bool:
    """Check if prediction contains any of the target labels (case-insensitive).

    Exact recreation of the original contains_score from the Based paper's
    lm-evaluation-harness implementation (PR #1728):

        def contains_score(prediction, labels):
            return max(
                int(bool(re.search(re.compile(re.escape(label), re.IGNORECASE), prediction)))
                for label in labels
            )

    Args:
        prediction: The model's generated text.
        labels: List of acceptable answer strings.

    Returns:
        True if any label is found (case-insensitive) in prediction.
    """
    if not prediction or not labels:
        return False
    return max(
        int(bool(re.search(re.compile(re.escape(label), re.IGNORECASE), prediction)))
        for label in labels
    ) == 1


# ── Generation ─────────────────────────────────────────────────────────

def generate_completions_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list,
    max_new_tokens: int = 48,
    device: str = "cuda",
) -> list:
    """Generate completions for a batch of prompts.

    Uses left-padding so that generation starts from the right side of each
    sequence. Greedy decoding (do_sample=False). Stops at the first newline
    in each decoded output.

    Args:
        model: The HuggingFace causal LM.
        tokenizer: The tokenizer.
        prompts: List of prompt strings.
        max_new_tokens: Maximum tokens to generate per sequence.
        device: Device to run on.

    Returns:
        List of generated text strings (new tokens only, truncated at \\n).
    """
    # Tokenize with left-padding for batched generation
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        padding_side="left",
    ).to(device)

    # Track original lengths (excluding padding) to slice generated tokens
    attention_mask = inputs.attention_mask
    input_lengths = attention_mask.sum(dim=1)  # [batch_size]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only new tokens for each sequence in the batch
    results = []
    for i in range(len(prompts)):
        full_len = outputs[i].shape[0]
        new_tokens = outputs[i][input_lengths[i]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Stop at first newline (matches original "until": ["\\n"])
        nl = text.find("\n")
        if nl != -1:
            text = text[:nl]

        results.append(text.strip())

    return results


# ── Main evaluation logic ──────────────────────────────────────────────

@dataclass
class ClozeEvalResult:
    """Result of a cloze-completion evaluation."""
    task_name: str
    total_examples: int
    correct: int
    accuracy: float
    per_example_results: list = field(default_factory=list)


def evaluate_cloze_task(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    task_name: str,
    max_new_tokens: int = 48,
    max_examples: Optional[int] = None,
    batch_size: int = 8,
    device: str = "cuda",
) -> ClozeEvalResult:
    """Evaluate a model on a single cloze-completion task.

    For each example:
      1. doc["text"] is the prompt (context + attribute key)
      2. doc["value"] is the expected answer
      3. Model generates a completion
      4. contains_score checks if answer appears in generation

    This faithfully reproduces the evaluation from the Based paper:
      - SWDE: 1,111 examples of semi-structured HTML extraction
      - FDA: FDA 510(k) document information extraction
      - SQuAD-Completion: 2,984 examples of document QA as completion

    Args:
        model: The HuggingFace causal LM.
        tokenizer: The tokenizer for the model.
        task_name: One of "swde", "fda", "squad_completion".
        max_new_tokens: Maximum tokens to generate per example.
        max_examples: Limit number of examples (for quick testing).
        batch_size: Number of examples to process in parallel.
        device: Device to run on.

    Returns:
        ClozeEvalResult with accuracy and per-example details.
    """
    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{task_name}'. Available: {list(TASK_REGISTRY.keys())}"
        )

    task_info = TASK_REGISTRY[task_name]
    logger.info(f"Loading dataset: {task_info['dataset_name']} ...")
    dataset = load_dataset(
        task_info["dataset_name"],
        task_info["dataset_config"],
        split=task_info["split"],
    )

    total = len(dataset)
    if max_examples is not None:
        total = min(total, max_examples)

    correct = 0
    per_example_results = []

    logger.info(
        f"Evaluating {task_name}: {total} examples, batch_size={batch_size}, "
        f"max_new_tokens={max_new_tokens}"
    )
    model.eval()

    # Tokenize all prompts upfront for consistency with original lm-eval
    prompts = [dataset[i]["text"] for i in range(total)]
    targets = [dataset[i]["value"] for i in range(total)]

    num_batches = (total + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"  {task_name}", unit="batch"):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch_prompts = prompts[start:end]
        batch_targets = targets[start:end]

        try:
            batch_generated = generate_completions_batch(
                model, tokenizer, batch_prompts,
                max_new_tokens=max_new_tokens,
                device=device,
            )
        except Exception as e:
            logger.warning(f"Error on batch {batch_idx}: {e}")
            batch_generated = [""] * len(batch_prompts)

        for j, (generated, target) in enumerate(zip(batch_generated, batch_targets)):
            i = start + j
            is_correct = contains_score(generated, [target])
            if is_correct:
                correct += 1
            per_example_results.append({
                "idx": i,
                "correct": is_correct,
                "generated": generated,
                "target": target,
            })

            # Log first few examples for debugging
            if i < 3:
                logger.debug(
                    f"  Example {i}: target='{target}' | generated='{generated}' | "
                    f"correct={is_correct}"
                )

    accuracy = correct / total if total > 0 else 0.0
    logger.info(
        f"  {task_name}: {correct}/{total} correct, accuracy={accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    return ClozeEvalResult(
        task_name=task_name,
        total_examples=total,
        correct=correct,
        accuracy=accuracy,
        per_example_results=per_example_results,
    )


def evaluate_all_cloze_tasks(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    tasks: list,
    max_new_tokens: int = 48,
    max_examples: Optional[int] = None,
    batch_size: int = 8,
    device: str = "cuda",
) -> dict:
    """Evaluate model on multiple cloze-completion tasks.

    Args:
        model: The HuggingFace causal LM.
        tokenizer: The tokenizer for the model.
        tasks: List of task names (e.g., ["swde", "fda", "squad_completion"]).
        max_new_tokens: Maximum tokens to generate per example.
        max_examples: Limit number of examples per task (for quick testing).
        batch_size: Number of examples to process in parallel.
        device: Device to run on.

    Returns:
        Dict mapping task_name → ClozeEvalResult.
    """
    results = {}
    for task_name in tasks:
        result = evaluate_cloze_task(
            model=model,
            tokenizer=tokenizer,
            task_name=task_name,
            max_new_tokens=max_new_tokens,
            max_examples=max_examples,
            batch_size=batch_size,
            device=device,
        )
        results[task_name] = result
    return results
