"""
(FROM HANGAR)
Shard-based dataloader for pretokenized datasets from HuggingFace Hub.

Supports:
- Multi-epoch training with different shuffle orderings
- Deterministic token subsets via max_tokens
- Distributed training with shard-level assignment
"""

import json
import math
import random

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from torch.utils.data import DataLoader, IterableDataset


class ShardedPreTokenizedDataset(IterableDataset):
    """
    Streams pretokenized data from HuggingFace Hub with shard-level shuffling.

    Each shard (~100M tokens) is loaded into memory, shuffled, and yielded.
    Shard order is also shuffled per epoch for varied training.
    """

    def __init__(
        self,
        hf_repo: str,
        seq_len: int,
        rank: int = 0,
        world_size: int = 1,
        split: str = "train",
        max_tokens: int | None = None,
        epoch: int = 0,
        version: str | None = None,
        subset: str | None = None,
        replicate_shards: bool = False,
    ):
        """
        Args:
            hf_repo: HuggingFace dataset repo (e.g., "org/fineweb-edu_gpt2")
            seq_len: Sequence length for training
            rank: Current process rank (for distributed training)
            world_size: Total number of processes
            split: Dataset split ("train" or "validation")
            max_tokens: Maximum tokens to use (deterministic subset). None = use all.
            epoch: Current epoch (controls shuffle ordering)
            version: Dataset version tag (e.g., "v1"). None = latest.
            subset: Subdirectory in the repo to load from.
            replicate_shards: If True, all ranks load all shards but filter sequences.
                Useful for validation when there are fewer shards than ranks.
        """
        self.hf_repo = hf_repo
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.split = split
        self.max_tokens = max_tokens
        self.epoch = epoch
        self.version = version
        self.subset = subset
        self.replicate_shards = replicate_shards

        # Load metadata
        self.metadata = self._load_metadata()
        self.shard_size = self.metadata["shard_size"]
        self.chunk_size = self.metadata["chunk_size"]

        # List all shards for this split
        all_shards = self._list_shards()

        # Determine how many shards we need for max_tokens
        if max_tokens is not None:
            num_shards_needed = math.ceil(max_tokens / self.shard_size)
            all_shards = all_shards[:num_shards_needed]

        # Assign shards to this rank
        if replicate_shards:
            # All ranks load all shards (for validation with few shards)
            self.my_shards = all_shards
        else:
            # Round-robin assignment (for training)
            self.my_shards = [shard for i, shard in enumerate(all_shards) if i % world_size == rank]

    def __iter__(self):
        # Split shards across DataLoader workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            shards = self.my_shards
        else:
            shards = [
                s
                for i, s in enumerate(self.my_shards)
                if i % worker_info.num_workers == worker_info.id
            ]

        # Shuffle shard order for this epoch
        shard_rng = random.Random(self.epoch)
        shard_order = shards.copy()
        shard_rng.shuffle(shard_order)

        for shard_name in shard_order:
            yield from self._iter_shard(shard_name)

    def _iter_shard(self, shard_name: str):
        """Load a shard, shuffle sequences, and yield (x, y) pairs."""
        # Load shard from HF
        if self.rank == 0:
            print(f"Loading shard {shard_name} from HF Hub.")
        shard_ds = load_dataset(
            self.hf_repo,
            data_files=f"{shard_name}.parquet",
            split="train",  # parquet files use "train" as default split name
            revision=self.version,
        )
        if self.rank == 0:
            print(f"Shard {shard_name} loaded from HF Hub.")

        # Collect all tokens from this shard
        if self.rank == 0:
            print(f"Collecting all tokens from shard {shard_name}.")
        all_tokens = np.concatenate(shard_ds.with_format("numpy")["tokens"])

        # Create sequences
        if self.rank == 0:
            print(f"Creating sequences from shard {shard_name}.")
        sequences = []
        for i in range(0, len(all_tokens) - self.seq_len, self.seq_len):
            seq = all_tokens[i : i + self.seq_len + 1]
            sequences.append(seq)

        # Shuffle sequences within this shard
        # Use (epoch, shard_name) for unique but reproducible seed
        if self.rank == 0:
            print(f"Shuffling sequences from shard {shard_name}.")
        seq_rng = random.Random(f"{self.epoch}_{shard_name}")
        seq_rng.shuffle(sequences)

        # Yield sequences as tensors
        for seq_idx, seq in enumerate(sequences):
            # When replicating shards, each rank takes every Nth sequence
            if self.replicate_shards and seq_idx % self.world_size != self.rank:
                continue

            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)
            yield x, y

    def _load_metadata(self) -> dict:
        """Fetch metadata.json from HF repo."""
        path = hf_hub_download(
            self.hf_repo,
            "metadata.json",
            repo_type="dataset",
            revision=self.version,
        )
        with open(path) as f:
            return json.load(f)

    def _list_shards(self) -> list[str]:
        """Get sorted list of shard names for this split."""
        files = list_repo_files(
            self.hf_repo,
            repo_type="dataset",
            revision=self.version,
        )
        shards = sorted(
            [
                f.replace(".parquet", "")
                for f in files
                if f.startswith(f"{self.split}-") and f.endswith(".parquet")
            ]
        )
        return shards


def get_dataloader(
    hf_repo: str,
    batch_size: int,
    seq_len: int,
    rank: int = 0,
    world_size: int = 1,
    split: str = "train",
    max_tokens: int | None = None,
    epoch: int = 0,
    version: str | None = None,
    subset: str | None = None,
    replicate_shards: bool = False,
    num_workers: int | None = None,
) -> DataLoader:
    """
    Returns a DataLoader for pretokenized data from HuggingFace Hub.

    Args:
        hf_repo: HuggingFace dataset repo (e.g., "org/fineweb-edu_gpt2")
        batch_size: Batch size
        seq_len: Sequence length for training
        rank: Current process rank (for distributed training)
        world_size: Total number of processes
        split: Dataset split ("train" or "validation")
        max_tokens: Maximum tokens to use (deterministic subset). None = use all.
        epoch: Current epoch (controls shuffle ordering)
        version: Dataset version tag (e.g., "v1"). None = latest.
        subset: Subdirectory in the repo to load from.
        replicate_shards: If True, all ranks load all shards but filter sequences.
        num_workers: Number of DataLoader workers. Default 4 for training, 0 recommended for validation.

    Returns:
        DataLoader yielding (x, y) batches of shape (batch_size, seq_len)
    """
    dataset = ShardedPreTokenizedDataset(
        hf_repo=hf_repo,
        seq_len=seq_len,
        rank=rank,
        world_size=world_size,
        split=split,
        max_tokens=max_tokens,
        epoch=epoch,
        version=version,
        subset=subset,
        replicate_shards=replicate_shards,
    )

    # Default: 4 workers for training, 0 for validation with replicated shards
    if num_workers is None:
        num_workers = 0 if replicate_shards else 4

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=True,
    )
