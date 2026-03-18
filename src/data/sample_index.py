"""
Precomputed sample index + index-backed dataset for efficient batching and weighted sampling.

Each index row describes one training sample (e.g. query segment + prompt refs). The index
stores lengths and optional weights so you can:
  - Batch samples with similar lengths (less padding)
  - Weighted sampling (oversample rare tasks, failures, high-reward)
  - Curriculum by quality

Interface (dataset-agnostic):
  - SampleIndex: wraps a DataFrame (from Parquet or in-memory). Optional: weights, length_bin.
  - IndexBackedDataset: PyTorch Dataset that takes (index, loader_fn). __getitem__(i) = loader_fn(index.row(i)).
  - GroupedBatchSampler: yields batches of index row indices that share the same group key (e.g. length bin).
  - WeightedIndexSampler: optional sampler using index["weight"].

Standard index columns (convention; not all required):
  - episode_id / query_episode_id, start / query_start, length / query_len
  - task_id, is_success
  - For in-context: prompt_episode_ids, prompt_starts, prompt_lens (lists) or prompt_episode_id, prompt_start, prompt_len
  - weight (float, optional)
  - prompt_len (total steps) and query_len for grouped batching

Example row (in-context):
  {"query_episode_id": 81, "query_start": 40, "query_len": 32, "task_id": 7, "is_success": 1,
   "prompt_episode_ids": [12, 3], "prompt_starts": [5, 0], "prompt_lens": [24, 50], "prompt_len": 74}

Example usage (any dataset):
  index = SampleIndex("path/to/sample_index.parquet", length_bin_columns=["query_len", "prompt_len"])
  loader_fn = your_dataset_make_loader(...)  # (row) -> training example tuple
  dataset = IndexBackedDataset(index, loader_fn)
  batch_sampler = GroupedBatchSampler(index, batch_size=32, shuffle=True, seed=0)
  loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=your_collate_fn)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class InContextDatasetResult:
    """Result of building an in-context index-backed dataset (dataset, loader, state stats, task_instructions)."""

    def __init__(
        self,
        dataset: Any,
        loader: DataLoader,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        task_instructions: List[str],
        total_prompt_len: Optional[int],
        max_prompt_trajectory_length: Optional[int],
    ):
        self.dataset = dataset
        self.loader = loader
        self.state_mean = state_mean
        self.state_std = state_std
        self.task_instructions = task_instructions
        self.total_prompt_len = total_prompt_len
        self.max_prompt_trajectory_length = max_prompt_trajectory_length


_IN_CONTEXT_BUILDERS: Dict[str, Callable[..., Optional[InContextDatasetResult]]] = {}


def register_in_context_builder(
    source: str,
    builder: Callable[..., Optional[InContextDatasetResult]],
) -> None:
    """Register a builder: (data_dir, data_cfg, device, state_dim, action_dim, collate_fn) -> InContextDatasetResult or None."""
    _IN_CONTEXT_BUILDERS[source] = builder


def build_in_context_dataset(
    source: str,
    data_dir: Union[str, Path],
    data_cfg: Any,
    device: torch.device,
    state_dim: int,
    action_dim: int,
    collate_fn: Callable,
) -> Optional[InContextDatasetResult]:
    """Build dataset + loader from a precomputed sample index for the given source. Returns None if no index."""
    builder = _IN_CONTEXT_BUILDERS.get(source)
    if builder is None and source == "libero":
        import src.data.libero_dataset
        builder = _IN_CONTEXT_BUILDERS.get("libero")
    if builder is None:
        return None
    return builder(data_dir, data_cfg, device, state_dim, action_dim, collate_fn)


class SampleIndex:
    """
    Wraps a table of training samples (one row = one sample). Load from Parquet or pass a DataFrame.
    Optional: add weight column, add length_bin for grouped batching.
    """

    def __init__(
        self,
        data: Union[str, "pd.DataFrame", Any],
        weight_column: Optional[str] = None,
        length_bin_columns: Optional[List[str]] = None,
        length_bin_bins: Optional[Dict[str, Union[int, List[int]]]] = None,
    ):
        """
        Args:
            data: Path to .parquet or DataFrame.
            weight_column: If set, use this column for weighted sampling (must exist or be computed).
            length_bin_columns: Column names used to form a group key for batching (e.g. ["query_len", "prompt_len"]).
            length_bin_bins: If set, discretize these columns into bins. E.g. {"query_len": 8, "prompt_len": 4}
                bins the column into that many quantile bins. Or {"query_len": [0, 16, 32, 64]} for fixed edges.
        """
        import pandas as pd

        if isinstance(data, (str, Path)):
            self._df = pd.read_parquet(data)
        else:
            self._df = data.copy() if hasattr(data, "copy") else pd.DataFrame(data)
        self._weight_col = weight_column
        self._length_bin_cols = length_bin_columns or []
        self._length_bin_bins = length_bin_bins or {}
        self._group_key_col = "_group_key"

        if length_bin_columns or length_bin_bins:
            self._add_group_key(length_bin_columns, length_bin_bins)

    def _add_group_key(
        self,
        length_bin_columns: Optional[List[str]],
        length_bin_bins: Optional[Dict[str, Union[int, List[int]]]],
    ) -> None:
        import pandas as pd

        df = self._df
        cols = length_bin_columns or list((length_bin_bins or {}).keys())
        bin_series = []
        for col in cols:
            if col not in df.columns:
                continue
            ser = df[col]
            if length_bin_bins and col in length_bin_bins:
                spec = length_bin_bins[col]
                if isinstance(spec, int):
                    try:
                        ser = pd.qcut(ser.astype(float), q=min(spec, max(2, ser.nunique())), labels=False, duplicates="drop")
                    except Exception:
                        ser = pd.Series(0, index=df.index)
                else:
                    ser = pd.cut(ser.astype(float), bins=spec, labels=False)
            bin_series.append(ser.astype(str))
        if bin_series:
            self._df = self._df.copy()
            if len(bin_series) == 1:
                self._df[self._group_key_col] = bin_series[0]
            else:
                self._df[self._group_key_col] = pd.Series(
                    ["_".join(v) for v in zip(*(s.values for s in bin_series))],
                    index=df.index,
                )
        else:
            self._group_key_col = None

    def __len__(self) -> int:
        return len(self._df)

    def row(self, i: int) -> Dict[str, Any]:
        """Return the i-th row as a dict (str keys, values as scalars or lists)."""
        row = self._df.iloc[i]
        return row.to_dict()

    def row_at(self, i: int) -> Dict[str, Any]:
        return self.row(i)

    @property
    def df(self):
        return self._df

    def weights(self) -> Optional[np.ndarray]:
        """Weights for weighted sampling; None if no weight column."""
        if not self._weight_col or self._weight_col not in self._df.columns:
            return None
        return np.asarray(self._df[self._weight_col], dtype=np.float64)

    def group_keys(self) -> Optional[np.ndarray]:
        """Group key per row for GroupedBatchSampler; None if not computed."""
        if not self._group_key_col or self._group_key_col not in self._df.columns:
            return None
        return self._df[self._group_key_col].values

    def groups(self) -> Optional[Dict[Any, np.ndarray]]:
        """Map group_key -> indices. For GroupedBatchSampler."""
        keys = self.group_keys()
        if keys is None:
            return None
        out: Dict[Any, List[int]] = {}
        for i, k in enumerate(keys):
            out.setdefault(k, []).append(i)
        return {k: np.array(v, dtype=np.int64) for k, v in out.items()}


class IndexBackedDataset(Dataset):
    """
    Dataset that samples from a SampleIndex; each __getitem__(i) calls loader_fn(index.row(i)).
    Use with GroupedBatchSampler to batch by length, or WeightedRandomSampler for weighted sampling.
    """

    def __init__(
        self,
        index: SampleIndex,
        loader_fn: Callable[[Dict[str, Any]], Any],
    ):
        self.index = index
        self.loader_fn = loader_fn

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Any:
        row = self.index.row(i)
        return self.loader_fn(row)


class GroupedBatchSampler(Sampler[List[int]]):
    """
    Yields batches of indices such that each batch contains only indices from the same group
    (e.g. same length_bin). Reduces padding when segment lengths vary.
    """

    def __init__(
        self,
        index: SampleIndex,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.index = index
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        groups = index.groups()
        if groups is None:
            self._batches = self._batches_flat()
        else:
            self._batches = self._batches_grouped(groups)

    def _batches_flat(self) -> List[List[int]]:
        n = len(self.index)
        indices = list(range(n))
        batches = []
        for start in range(0, n, self.batch_size):
            b = indices[start : start + self.batch_size]
            if len(b) == self.batch_size or not self.drop_last:
                batches.append(b)
        return batches

    def _batches_grouped(self, groups: Dict[Any, np.ndarray]) -> List[List[int]]:
        rng = np.random.default_rng(self.seed)
        batches = []
        for _key, indices in groups.items():
            idx = indices.tolist()
            if self.shuffle:
                rng.shuffle(idx)
            for start in range(0, len(idx), self.batch_size):
                b = idx[start : start + self.batch_size]
                if len(b) == self.batch_size or not self.drop_last:
                    batches.append(b)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        for b in self._batches:
            yield b

    def __len__(self) -> int:
        return len(self._batches)


class WeightedIndexSampler(Sampler[int]):
    """Samples index row indices with replacement using index['weight']."""

    def __init__(
        self,
        index: SampleIndex,
        num_samples: Optional[int] = None,
        replacement: bool = True,
        seed: Optional[int] = None,
    ):
        self.index = index
        self.num_samples = num_samples or len(index)
        self.replacement = replacement
        self.seed = seed
        w = index.weights()
        self.weights = torch.from_numpy(w / w.sum()) if w is not None else None

    def __iter__(self) -> Iterator[int]:
        if self.weights is None:
            g = torch.Generator()
            if self.seed is not None:
                g.manual_seed(self.seed)
            for _ in range(self.num_samples):
                yield int(torch.randint(0, len(self.index), (1,), generator=g).item())
            return
        perm = torch.multinomial(
            self.weights,
            self.num_samples,
            self.replacement,
            generator=torch.Generator().manual_seed(self.seed) if self.seed is not None else None,
        )
        for i in perm.tolist():
            yield i

    def __len__(self) -> int:
        return self.num_samples


