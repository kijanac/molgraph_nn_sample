from __future__ import annotations
from typing import Iterable

import numpy as np
import pathlib
import torch


def graph_collate(batch: Iterable[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate multiple Data objects containing graph data.
    Parameters
    ----------
    batch : Iterable[luz.Data]
        Data objects to be collated
    Returns
    -------
    luz.Data
        Collated Data object
    """
    node_counts = [sample["x"].shape[0] for sample in batch]
    edge_index_offsets = np.roll(np.cumsum(node_counts), shift=1)
    edge_index_offsets[0] = 0

    d = {}

    for k in batch[0].keys():
        if k in ("x", "edges"):  # , "y"):
            d[k] = torch.cat([torch.as_tensor(sample[k]) for sample in batch], dim=0)
        elif k == "edge_index":
            d[k] = torch.cat(
                [
                    torch.as_tensor(sample[k] + offset, dtype=torch.long)
                    for sample, offset in zip(batch, edge_index_offsets)
                ],
                dim=1,
            )
        else:
            d[k] = torch.stack([torch.as_tensor(sample[k]) for sample in batch], dim=0)

    d["batch"] = torch.cat(
        [torch.full((nc,), i, dtype=torch.long) for i, nc in enumerate(node_counts)]
    )

    return d


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root: str) -> None:
        """Dataset which reads data from disk.
        Parameters
        ----------
        root
            Root directory containing data stored in .pt files.
        """
        p = pathlib.Path(root).expanduser()
        self.root = str(p.expanduser().resolve())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return torch.load(pathlib.Path(self.root, f"{index}.pt"))

    def __len__(self) -> int:
        return len(tuple(pathlib.Path(self.root).glob("[0-9]*.pt")))
