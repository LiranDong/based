# Adapted from train/datamodules/language_modeling_hf.py in the Based codebase.
# Removed PyTorch Lightning dependency. Training-only features (fault_tolerant_sampler,
# DDP, fast_forward) are stripped. Core data processing pipeline is preserved.

import logging
import pickle
import subprocess
import mmap
from itertools import chain
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

from eval.lm_dataset import LMDataset

logger = logging.getLogger(__name__)


class LMDataModule:
    """Standalone language modeling data module (no Lightning dependency).

    Downloads a HuggingFace dataset, tokenizes it, concatenates all tokens
    into flat arrays, and provides DataLoader(s) over fixed-length sequences.

    Args:
        dataset_name: HuggingFace dataset name (e.g., 'wikitext', 'openwebtext').
        tokenizer_name: HuggingFace tokenizer name (e.g., 'gpt2').
        dataset_config_name: Dataset config/subset name.
        max_length: Sequence length for each training sample.
        cache_dir: Directory to cache tokenized data. If None, only downloads.
        val_ratio: Fraction of training data to use for validation (if no val split).
        val_split_seed: Random seed for train/val split.
        add_eos: Whether to append EOS token to each text.
        val_only: If True, use validation data for both train and test.
        batch_size: Batch size for evaluation.
        num_workers: Number of workers for data preprocessing.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: str,
        dataset_config_name: Optional[str] = None,
        max_length: int = 1024,
        cache_dir: Optional[str] = None,
        val_ratio: float = 0.0005,
        val_split_seed: int = 2357,
        add_eos: bool = True,
        detokenize: bool = False,
        val_only: bool = False,
        batch_size: int = 32,
        num_workers: int = 1,
    ):
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.max_length = max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.num_workers = num_workers

    # ── Public API ──────────────────────────────────────────────────────

    def prepare_data(self):
        """Download the dataset. Called once before setup()."""
        if self.cache_dir is None:
            load_dataset(self.dataset_name, self.dataset_config_name)
        else:
            self._process_dataset()

    def setup(self, stage: Optional[str] = None):
        """Tokenize the dataset and create train/val/test LMDataset splits."""
        if stage == "test" and hasattr(self, "dataset_test"):
            return
        concat_ids, self.tokenizer = self._process_dataset()
        self.vocab_size = len(self.tokenizer)
        self.dataset_train = LMDataset(concat_ids["train"], seq_len=self.max_length)
        self.dataset_val = LMDataset(concat_ids["validation"], seq_len=self.max_length)
        self.dataset_test = LMDataset(concat_ids["test"], seq_len=self.max_length)

    def val_dataloader(self) -> DataLoader:
        """Return DataLoader for the validation split."""
        return self._make_dataloader(self.dataset_val)

    def test_dataloader(self) -> DataLoader:
        """Return DataLoader for the test split."""
        return self._make_dataloader(self.dataset_test)

    # ── Internal ────────────────────────────────────────────────────────

    def _make_dataloader(self, dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=0,  # Data is already in memory
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    @property
    def _cache_dir_name(self) -> str:
        return (
            f"tokenizer_name-{self.tokenizer_name}"
            f"-val_ratio-{self.val_ratio}"
            f"-val_split_seed-{self.val_split_seed}"
            f"-add_eos-{self.add_eos}"
            f"-detokenize-{self.detokenize}"
        )

    def _process_dataset(self):
        """Download, tokenize, and concatenate the dataset. Returns (concat_ids, tokenizer)."""
        cache_dir = None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        if cache_dir is not None and cache_dir.is_dir():
            return self._load_from_cache(cache_dir)

        raw_datasets = load_dataset(self.dataset_name, self.dataset_config_name)

        # Create validation split if not present
        if "validation" not in raw_datasets:
            assert "train" in raw_datasets, "Dataset must have a 'train' split"
            raw_datasets = raw_datasets["train"].train_test_split(
                test_size=self.val_ratio,
                seed=self.val_split_seed,
                shuffle=True,
            )
            raw_datasets["validation"] = raw_datasets["test"]

        if self.val_only:
            raw_datasets["train"] = raw_datasets["validation"]

        # Detokenize if requested (useful for wikitext-103 zero-shot transfer)
        if self.detokenize:
            from train.datamodules.datasets.detokenizer import DATASET_TOKENIZATION_REGISTRY

            if self.dataset_name in DATASET_TOKENIZATION_REGISTRY:
                detokenizer = DATASET_TOKENIZATION_REGISTRY[self.dataset_name]
                raw_datasets = raw_datasets.map(
                    lambda example: {"text": detokenizer(example["text"])},
                    num_proc=max(self.num_workers, 1),
                    desc="Running detokenizer on dataset",
                )

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)

        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        if self.add_eos:
            add_eos = lambda seq: (seq + tokenizer.eos_token) if seq else seq
            add_eos_batched = lambda seqs: [add_eos(seq) for seq in seqs]
            tokenize = lambda example: tokenizer(add_eos_batched(example[text_column_name]))
        else:
            tokenize = lambda example: tokenizer(example[text_column_name])

        dtype = np.uint16 if tokenizer.vocab_size < 64 * 1024 else np.int32

        def tokenize_concat(examples):
            input_ids = np.fromiter(chain(*tokenize(examples)["input_ids"]), dtype=dtype)
            return {"input_ids": [input_ids], "len": [len(input_ids)]}

        tokenized_datasets = raw_datasets.map(
            tokenize_concat,
            batched=True,
            num_proc=max(self.num_workers, 1),
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )

        # Concatenate all input_ids into flat arrays (on disk, memory-mapped)
        concat_ids = {}
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

        for name, ds in tokenized_datasets.items():
            tokenized_datasets[name] = ds.add_column(
                "len_offset", np.cumsum(ds["len"])
            )
            array_len = tokenized_datasets[name][-1]["len_offset"]
            filename = cache_dir / f"{name}.bin" if cache_dir is not None else None

            if filename is not None:
                # Create a file of the exact size needed
                subprocess.run(
                    [
                        "truncate",
                        "-s",
                        str(array_len * np.dtype(dtype).itemsize),
                        str(filename),
                    ],
                    check=True,
                )

                # Write token IDs to memory-mapped file
                def write_ids_to_disk(example, filename):
                    with open(filename, "r+b") as f:
                        mm = mmap.mmap(f.fileno(), 0)
                        start_idx = example["len_offset"] - len(example["input_ids"])
                        arr = np.ndarray(
                            (len(example["input_ids"]),),
                            dtype=dtype,
                            buffer=mm,
                            offset=np.dtype(dtype).itemsize * start_idx,
                        )
                        arr[:] = example["input_ids"]
                        mm.flush()

                tokenized_datasets[name].map(
                    write_ids_to_disk,
                    fn_kwargs={"filename": str(filename)},
                    batched=False,
                    num_proc=max(self.num_workers, 1),
                    desc=f"Writing {name} to disk",
                )
                concat_ids[name] = np.memmap(
                    filename, dtype=dtype, mode="r", shape=(array_len,)
                )
            else:
                concat_ids[name] = np.concatenate(
                    [ds["input_ids"] for ds in tokenized_datasets[name]]
                )

        if cache_dir is not None:
            self._save_to_cache(concat_ids, tokenizer, cache_dir)

        return concat_ids, tokenizer

    def _save_to_cache(self, concat_ids, tokenizer, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving tokenized data to {cache_dir}")
        for k, v in concat_ids.items():
            np.save(str(cache_dir / f"{k}.npy"), v)
        with open(str(cache_dir / "tokenizer.pkl"), "wb") as f:
            pickle.dump(tokenizer, f)

    def _load_from_cache(self, cache_dir):
        logger.info(f"Loading tokenized data from {cache_dir}")
        concat_ids = {
            split: np.load(str(cache_dir / f"{split}.npy"), mmap_mode="r")
            for split in ["train", "validation", "test"]
        }
        with open(str(cache_dir / "tokenizer.pkl"), "rb") as f:
            tokenizer = pickle.load(f)
        return concat_ids, tokenizer
