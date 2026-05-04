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

def generate_completion(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 48,
    device: str = "cuda",
) -> str:
    """Generate a completion for a single prompt.

    Uses greedy decoding (do_sample=False). Stops at the first newline
    character in the decoded output, matching the original "until: ['\\n']"
    behavior from lm-evaluation-harness.

    Args:
        model: The HuggingFace causal LM.
        tokenizer: The tokenizer for the model.
        prompt: The full input text (context + key).
        max_new_tokens: Maximum tokens to generate.
        device: Device to run on.

    Returns:
        The generated text (new tokens only), truncated at first newline.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    generated_ids = outputs[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Stop at first newline (matches "until": ["\\n"] in original config)
    newline_idx = generated_text.find("\n")
    if newline_idx != -1:
        generated_text = generated_text[:newline_idx]

    return generated_text.strip()


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

    logger.info(f"Evaluating {task_name}: {total} examples, max_new_tokens={max_new_tokens}")
    model.eval()

    for i in tqdm(range(total), desc=f"  {task_name}", unit="ex"):
        example = dataset[i]
        prompt = example["text"]
        target = example["value"]

        try:
            generated = generate_completion(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            is_correct = contains_score(generated, [target])
        except Exception as e:
            logger.warning(f"Error on example {i}: {e}")
            generated = ""
            is_correct = False

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
    device: str = "cuda",
) -> dict:
    """Evaluate model on multiple cloze-completion tasks.

    Args:
        model: The HuggingFace causal LM.
        tokenizer: The tokenizer for the model.
        tasks: List of task names (e.g., ["swde", "fda", "squad_completion"]).
        max_new_tokens: Maximum tokens to generate per example.
        max_examples: Limit number of examples per task (for quick testing).
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
            device=device,
        )
        results[task_name] = result
    return results
