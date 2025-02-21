import torch
import numpy as np
import random
from pathlib import Path
import glob
from typing import Iterator, Tuple
import torch.distributed as dist

class PreTokDataset(torch.utils.data.IterableDataset):
    """
    Dataset class that handles pre-tokenized data, implementing an iterable interface.
    Supports distributed training and multiple workers by managing data sharding.
    """
    def __init__(self, split: str, max_seq_len: int):
        """
        Initialize dataset with training/validation split and sequence length parameters.
        Args:
            split (str): Dataset split ('train' or 'val')
            max_seq_len (int): Maximum sequence length for each training example
        """
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # Set up worker and distributed training info for proper data sharding
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = 42 + worker_id + 1337 * rank

        # Get list of data shard files and split between train/val
        bin_dir = Path("data/TinyStories_all_data")
        shard_filenames = sorted(glob.glob(str(bin_dir / "*.bin")))
        shard_filenames = (
            shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        )
        
        # Initialize random number generator for shuffling
        rng = random.Random(seed)
        
        # Main data loading loop
        while True:
            # Shuffle shard order for each epoch
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # Memory-map the binary data file for efficient reading
                data = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(data) // self.max_seq_len - 1
                idxs = list(range(num_batches))
                rng.shuffle(idxs)

                # Generate training examples from the shard
                for idx in idxs:
                    # Extract sequence chunk and create input/target pairs
                    start = idx * self.max_seq_len
                    end = (idx + 1) * self.max_seq_len
                    chunk = torch.from_numpy(data[start:end].astype(np.int64))
                    x = chunk[:-1]  # Input sequence
                    y = chunk[1:]   # Target sequence (shifted by 1)
                    yield x, y


class Task:
    """
    Utility class that provides a high-level interface for creating data iterators
    with proper batching and device placement.
    """
    @staticmethod
    def iter_batches(
        batch_size: int, device: str, num_workers: int = 0, **dataset_kwargs
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Creates an iterator that yields batches of data.
        Args:
            batch_size (int): Size of each batch
            device (str): Device to place the batches on ('cpu' or 'cuda')
            num_workers (int): Number of worker processes for data loading
            **dataset_kwargs: Additional arguments passed to PreTokDataset
        Returns:
            Iterator yielding (input, target) batches placed on the specified device
        """
        ds = PreTokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            yield x, y
