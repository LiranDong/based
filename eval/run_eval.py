#!/usr/bin/env python3
"""Language model evaluation script. Supports two modes:

1. Perplexity mode (--dataset):
   Evaluate perplexity on language modeling datasets using the original
   Based data pipeline (LMDataset + LMDataModule).

   python eval/run_eval.py --model fla-hub/rwkv7-168M-pile --dataset wikitext
   python eval/run_eval.py --model fla-hub/rwkv7-168M-pile --dataset openwebtext

2. Cloze-completion mode (--task):
   Evaluate accuracy on SWDE, FDA, and SQuAD-Completion recall-intensive tasks.
   Exact recreation of the Based paper evaluation: generate_until + contains_score.

   python eval/run_eval.py --model fla-hub/rwkv7-168M-pile --task swde
   python eval/run_eval.py --model fla-hub/rwkv7-168M-pile --task swde --task fda --task squad_completion
   python eval/run_eval.py --model fla-hub/rwkv7-168M-pile --task squad_completion --max_examples 100
"""

import argparse
import logging
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.data_module import LMDataModule
from eval.metrics import Perplexity
from eval.cloze_tasks import evaluate_all_cloze_tasks, TASK_REGISTRY

from ramnet import RAMNetConfig
from fla.models import TransformerConfig, LinearAttentionConfig, HGRN2Config

logger = logging.getLogger(__name__)

# ── Dataset presets ────────────────────────────────────────────────────

DATASET_PRESETS = {
    "wikitext": {
        "dataset_name": "wikitext",
        "dataset_config_name": "wikitext-103-v1",
        "tokenizer_name": "gpt2",
        "max_length": 2048,
    },
    "openwebtext": {
        "dataset_name": "openwebtext",
        "dataset_config_name": None,
        "tokenizer_name": "gpt2",
        "max_length": 1024,
    },
    "thepile": {
        "dataset_name": "EleutherAI/pile",
        "dataset_config_name": None,
        "tokenizer_name": "gpt2",
        "max_length": 2048,
    },
    "slimpajama": {
        "dataset_name": "DKYoon/SlimPajama-6B",
        "dataset_config_name": "default",
        "tokenizer_name": "gpt2",
        "max_length": 2048,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a language model: perplexity (--dataset) or "
        "cloze-completion accuracy (--task)."
    )

    # ── Model ──────────────────────────────────────────────────────
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model name or path.",
    )
    parser.add_argument(
        "--revision", type=str, default=None,
        help="Model revision/branch.",
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype (default: bfloat16).",
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False,
        help="Trust remote code (required for some FLA models).",
    )
    parser.add_argument(
        "--attn_implementation", type=str, default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
    )

    # ── Mode: perplexity (--dataset) ────────────────────────────────
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Dataset preset or HF path for perplexity eval. "
        "Presets: wikitext, openwebtext, thepile, slimpajama. "
        "Omit to use --task mode instead.",
    )
    parser.add_argument(
        "--dataset_config", type=str, default=None,
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None,
        help="Tokenizer name (defaults to --model).",
    )
    parser.add_argument(
        "--max_length", type=int, default=None,
        help="Sequence length for perplexity eval.",
    )
    parser.add_argument(
        "--val_only", action="store_true", default=False,
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
    )
    parser.add_argument(
        "--max_eval_batches", type=int, default=None,
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["val", "test"],
    )

    # ── Mode: cloze-completion (--task) ─────────────────────────────
    parser.add_argument(
        "--task", type=str, action="append", default=None,
        choices=list(TASK_REGISTRY.keys()),
        help="Cloze-completion task(s): swde, fda, squad_completion. "
        "Can specify multiple: --task swde --task fda --task squad_completion",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=48,
        help="Max generation tokens for cloze tasks (default: 48).",
    )
    parser.add_argument(
        "--cloze_batch_size", type=int, default=8,
        help="Batch size for cloze generation (default: 8).",
    )
    parser.add_argument(
        "--context_length", type=int, default=4096,
        help="Max context length to truncate prompts for cloze tasks (default: 4096).",
    )
    parser.add_argument(
        "--max_examples", type=int, default=None,
        help="Limit examples per cloze task (for quick testing).",
    )

    # ── Common ──────────────────────────────────────────────────────
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (default: cuda if available).",
    )
    parser.add_argument(
        "--cache_dir", type=str, default="./cache",
        help="Cache directory for tokenized datasets.",
    )
    parser.add_argument(
        "--no_cache", action="store_true", default=False,
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
    )

    return parser.parse_args()


def get_dtype(dtype_str: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_str]


def load_model(args, device: str):
    """Load model and tokenizer from HuggingFace."""
    dtype = get_dtype(args.dtype)
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.revision is not None:
        model_kwargs["revision"] = args.revision
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation

    logger.info(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    if not getattr(model, "hf_device_map", None):
        model = model.to(device)
    model.eval()
    logger.info(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def run_perplexity_eval(args, model, tokenizer, device: str):
    """Perplexity evaluation on a language modeling dataset."""
    # Resolve dataset preset
    if args.dataset in DATASET_PRESETS:
        preset = DATASET_PRESETS[args.dataset]
        dataset_name = preset["dataset_name"]
        dataset_config_name = args.dataset_config or preset["dataset_config_name"]
        max_length = args.max_length or preset["max_length"]
        tokenizer_name = args.tokenizer or preset["tokenizer_name"]
    else:
        dataset_name = args.dataset
        dataset_config_name = args.dataset_config
        max_length = args.max_length or 2048
        tokenizer_name = args.tokenizer or args.model

    logger.info(f"Dataset: {dataset_name} (config: {dataset_config_name})")
    logger.info(f"Tokenizer: {tokenizer_name}, max_length: {max_length}")

    cache_dir = None if args.no_cache else args.cache_dir
    datamodule = LMDataModule(
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        cache_dir=cache_dir,
        batch_size=args.batch_size,
        val_only=args.val_only,
    )

    logger.info("Preparing dataset...")
    datamodule.prepare_data()
    datamodule.setup()

    dataloader = (
        datamodule.val_dataloader() if args.split == "val"
        else datamodule.test_dataloader()
    )
    total_batches = len(dataloader)

    metric = Perplexity()
    max_batches = args.max_eval_batches or total_batches

    logger.info(f"Split: {args.split}, batches: {total_batches}")
    logger.info("Evaluating perplexity...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            input_ids, labels = batch
            input_ids, labels = input_ids.to(device), labels.to(device)
            loss = model(input_ids=input_ids, labels=labels).loss
            metric.update(loss, labels.numel())

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                logger.info(
                    f"  Batch {batch_idx + 1}/{max_batches} | "
                    f"loss: {metric.avg_loss:.4f} | "
                    f"ppl: {metric.compute():.4f} | "
                    f"tokens: {metric.total_tokens:,}"
                )

    ppl = metric.compute()
    _print_perplexity_result(args.model, dataset_name, args.split, metric, ppl)
    return ppl


def _print_perplexity_result(model_name, dataset_name, split, metric, ppl):
    logger.info("=" * 60)
    logger.info("Perplexity evaluation complete.")
    logger.info(f"  Model:      {model_name}")
    logger.info(f"  Dataset:    {dataset_name}")
    logger.info(f"  Split:      {split}")
    logger.info(f"  Tokens:     {metric.total_tokens:,}")
    logger.info(f"  Avg loss:   {metric.avg_loss:.4f}")
    logger.info(f"  Perplexity: {ppl:.4f}")
    logger.info("=" * 60)
    print(f"\n{'='*40}")
    print(f"PERPLEXITY: {ppl:.4f}")
    print(f"{'='*40}")


def run_cloze_eval(args, model, tokenizer, device: str):
    """Cloze-completion accuracy evaluation on SWDE / FDA / SQuAD."""
    tasks = args.task
    logger.info(f"Cloze-completion tasks: {tasks}")
    logger.info(f"max_new_tokens: {args.max_new_tokens}, max_examples: {args.max_examples}")

    results = evaluate_all_cloze_tasks(
        model=model,
        tokenizer=tokenizer,
        tasks=tasks,
        max_new_tokens=args.max_new_tokens,
        max_examples=args.max_examples,
        batch_size=args.cloze_batch_size,
        context_length=args.context_length,
        device=device,
    )

    _print_cloze_result(args.model, results)
    return results


def _print_cloze_result(model_name: str, results: dict):
    logger.info("=" * 60)
    logger.info("Cloze-completion evaluation complete.")
    logger.info(f"  Model: {model_name}")
    logger.info("-" * 60)
    total_correct = 0
    total_examples = 0
    for task_name, r in results.items():
        logger.info(
            f"  {task_name:20s}  {r.correct:5d}/{r.total_examples:<5d}  "
            f"accuracy: {r.accuracy:.4f}  ({r.accuracy * 100:.2f}%)"
        )
        total_correct += r.correct
        total_examples += r.total_examples
    if len(results) > 1:
        overall = total_correct / total_examples if total_examples > 0 else 0
        logger.info("-" * 60)
        logger.info(
            f"  {'OVERALL':20s}  {total_correct:5d}/{total_examples:<5d}  "
            f"accuracy: {overall:.4f}  ({overall * 100:.2f}%)"
        )
    logger.info("=" * 60)

    # Bare output for parsing
    print(f"\n{'='*40}")
    for task_name, r in results.items():
        print(f"{task_name}: {r.accuracy:.4f}")
    print(f"{'='*40}")


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not args.dataset and not args.task:
        logger.error("Must specify either --dataset (perplexity) or --task (cloze-completion).")
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model, tokenizer = load_model(args, device)

    if args.task:
        run_cloze_eval(args, model, tokenizer, device)
    else:
        run_perplexity_eval(args, model, tokenizer, device)


if __name__ == "__main__":
    main()
