from __future__ import annotations
from typing import Callable

import numpy as np
import optuna
import random
import torch

from dataset import Dataset, graph_collate
from modules import Net, Scale, Transform


class MeanStd:
    def __init__(self, key: str = "x", batch_dim: int = 0) -> None:
        """Compute mean and standard deviation of data.
        Parameters
        ----------
        key
            Data key, by default "x".
        batch_dim
            Batch dimension, by default 0.
        """
        self.key = key
        self.batch_dim = batch_dim
        self.reset()

    def reset(self) -> None:
        """Reset metric state."""
        self.mean = 0.0
        self.var = 0.0
        self.n = 0.0

    def update(self, data: dict[str, torch.Tensor]) -> None:
        """Update metric state."""
        x = data[self.key].detach()
        self.n += x.size(self.batch_dim)
        delta = x.detach() - self.mean
        self.mean += delta.sum(self.batch_dim) / self.n
        self.var += (delta * (x.detach() - self.mean)).sum(self.batch_dim)

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute metric."""
        return self.mean, torch.sqrt(self.var / (self.n - 1))


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def preprocess(dataset: torch.utils.data.Dataset) -> Transform:
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, collate_fn=graph_collate
    )
    metrics = dict(
        x=MeanStd(key="x"),
        edges=MeanStd(key="edges"),
        u=MeanStd(key="u"),
    )

    for batch in loader:
        for m in metrics.values():
            m.update(batch)

    metrics = {k: m.compute() for k, m in metrics.items()}

    x_mean, x_std = metrics["x"]
    edges_mean, edges_std = metrics["edges"]
    u_mean, u_std = metrics["u"]

    x = Scale(x_mean, x_std)
    edges = Scale(edges_mean, edges_std)
    u = Scale(u_mean, u_std)

    return Transform(x=x, edges=edges, u=u)


def run_batch(
    batch: dict[str, torch.Tensor],
    model: torch.nn.Module,
    transform: torch.nn.Module,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = transform(batch)

    output = model(
        batch["x"],
        batch["edge_index"],
        batch["edges"],
        batch["u"],
        batch["batch"],
    )
    loss = criterion(output, batch["y"])

    return output, loss


def train(
    model: torch.nn.Module,
    fit_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    transform: Transform,
    max_epochs: int,
) -> list[float]:
    loader = torch.utils.data.DataLoader(
        fit_dataset, batch_size=64, collate_fn=graph_collate
    )

    val_losses = []

    for _ in range(max_epochs):
        for batch in loader:
            output, loss = run_batch(batch, model, transform, criterion)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        val_loss = validate(model, val_dataset, criterion, transform)
        val_losses.append(np.mean(val_loss))

    return val_losses


def validate(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    transform: Transform,
) -> float:
    losses = []

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, collate_fn=graph_collate
    )

    model.eval()
    for batch in loader:
        with torch.no_grad():
            output, loss = run_batch(batch, model, transform, criterion)

        losses.append(loss)

    model.train()

    return np.mean(losses)


def test(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    transform: Transform,
) -> float:
    losses = []

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, collate_fn=graph_collate
    )

    model.eval()
    for batch in loader:
        with torch.no_grad():
            output, loss = run_batch(batch, model, transform, criterion)

        losses.append(loss)

    model.train()

    return np.mean(losses)


def prepare_datasets(
    dataset: torch.utils.data.Dataset, val_fraction: float, test_fraction: float
):
    n = len(dataset)

    indices = torch.randperm(n).tolist()

    n_val = round(val_fraction * n)
    n_test = round(test_fraction * n)
    n_fit = n - n_val - n_test

    fit_dataset = torch.utils.data.Subset(dataset, indices[:n_fit])
    val_dataset = torch.utils.data.Subset(dataset, indices[n_fit : n_fit + n_val])
    test_dataset = torch.utils.data.Subset(dataset, indices[:-n_test])

    return fit_dataset, val_dataset, test_dataset


def learn(
    dataset: torch.utils.data.Dataset,
    max_epochs: int,
    val_fraction: float,
    test_fraction: float,
    num_iterations: int,
):
    study = optuna.create_study()

    # FIXME: move fit-val split inside tuning loop

    for _ in range(num_iterations):
        trial = study.ask()

        fit_dataset, val_dataset, test_dataset = prepare_datasets(
            dataset, val_fraction, test_fraction
        )

        transform = preprocess(fit_dataset)

        model = Net(
            num_layers=trial.suggest_int("num_layers", 1, 4),
            m_e=trial.suggest_int("m_e", 1, 4),
            K_v=trial.suggest_int("K_v", 1, 4),
            d_attn_v=trial.suggest_int("d_attn_v", 5, 100),
            m_v=trial.suggest_int("m_v", 1, 4),
            K_u=trial.suggest_int("K_u", 1, 4),
            d_attn_u=trial.suggest_int("d_attn_u", 5, 100),
            m_u=trial.suggest_int("m_u", 1, 4),
        )
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.MSELoss()

        val_losses = train(
            model,
            fit_dataset,
            val_dataset,
            optimizer,
            criterion,
            transform,
            max_epochs,
        )

        score = val_losses[-1]

        study.tell(trial, score)

        if score == study.best_value:
            best_model = model

    model = best_model

    test_loss = test(model, test_dataset, criterion, transform)

    return model, test_loss, study.best_trial


if __name__ == "__main__":
    set_seed(12345)
    device = "cuda:0"

    model, test_loss, optimal_hparams = learn(
        dataset=Dataset(root="data"),
        max_epochs=1,
        val_fraction=0.1,
        test_fraction=0.1,
        num_iterations=1,
    )

    torch.save(model.state_dict(), "model.pt")

    print(test_loss)
    print(optimal_hparams)
