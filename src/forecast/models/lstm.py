"""Encoder LSTM with a linear head for direct multi-horizon forecasting.

Training is a faithful port of the original study (same RNG call order, same
early stopping) so seeded runs reproduce it; covariates and batched inference
are additions that leave the univariate path unchanged.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from forecast.features.windows import make_windows

log = logging.getLogger(__name__)


class LSTMNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        max_horizon: int,
    ):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, max_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.encoder(x)
        return self.head(h_n[-1])


class LSTMForecaster:
    name = "lstm"

    def __init__(
        self,
        *,
        input_window: int = 168,
        max_horizon: int = 24,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 30,
        patience: int = 5,
        gradient_clip_norm: float = 1.0,
        loss: Literal["huber", "l1"] = "huber",
        val_fraction: float = 0.2,
        min_epochs: int = 0,
        torch_threads: int | None = None,
        seed: int = 42,
    ):
        self.min_epochs = min_epochs
        self.input_window = input_window
        self.max_horizon = max_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.gradient_clip_norm = gradient_clip_norm
        self.loss = loss
        self.val_fraction = val_fraction
        self.torch_threads = torch_threads
        self.seed = seed

        self.n_covariates = 0
        self.history: list[dict[str, float]] = []
        self.best_epoch: int | None = None
        self._model: LSTMNet | None = None
        self._device = torch.device("cpu")

    # ------------------------------------------------------------------ training

    def _set_seed(self) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def fit(self, y_train: np.ndarray, **context: Any) -> LSTMForecaster:
        if self.torch_threads:
            torch.set_num_threads(self.torch_threads)
        self._set_seed()
        covariates = context.get("covariates")
        self.n_covariates = 0 if covariates is None else int(np.asarray(covariates).shape[1])

        X, Y = make_windows(y_train, self.input_window, self.max_horizon, covariates=covariates)
        ok = np.isfinite(X).all(axis=(1, 2)) & np.isfinite(Y).all(axis=1)
        if not ok.all():  # windows touching an unfilled outage are unusable
            log.info("dropping %d of %d windows that contain NaN", int((~ok).sum()), len(ok))
            X, Y = X[ok], Y[ok]
        split = int(len(X) * (1.0 - self.val_fraction))
        X_tr, X_val = torch.from_numpy(X[:split]), torch.from_numpy(X[split:])
        Y_tr, Y_val = torch.from_numpy(Y[:split]), torch.from_numpy(Y[split:])

        self._model = LSTMNet(
            input_size=1 + self.n_covariates,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            max_horizon=self.max_horizon,
        ).to(self._device)
        loss_fn = nn.HuberLoss() if self.loss == "huber" else nn.L1Loss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)

        best_val, patience_counter, best_state = float("inf"), 0, None
        n_batches = (len(X_tr) + self.batch_size - 1) // self.batch_size
        start = time.perf_counter()
        self.history = []

        for epoch in range(self.max_epochs):
            self._model.train()
            epoch_loss = 0.0
            perm = torch.randperm(len(X_tr))
            for b in range(n_batches):
                idx = perm[b * self.batch_size : (b + 1) * self.batch_size]
                loss = loss_fn(self._model(X_tr[idx]), Y_tr[idx])
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), self.gradient_clip_norm)
                optimizer.step()
                epoch_loss += loss.item()

            self._model.eval()
            with torch.no_grad():
                val_loss = loss_fn(self._model(X_val), Y_val).item()

            if val_loss < best_val:
                best_val, patience_counter = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in self._model.state_dict().items()}
                self.best_epoch = epoch + 1
            else:
                patience_counter += 1
            self.history.append(
                {"epoch": epoch + 1, "train_loss": epoch_loss / n_batches, "val_loss": val_loss}
            )
            if (epoch + 1) % 10 == 0 or patience_counter >= self.patience:
                log.info(
                    "epoch %3d train=%.4f val=%.4f best=%.4f patience=%d/%d [%.0fs]",
                    epoch + 1,
                    epoch_loss / n_batches,
                    val_loss,
                    best_val,
                    patience_counter,
                    self.patience,
                    time.perf_counter() - start,
                )
            if patience_counter >= self.patience and epoch + 1 >= self.min_epochs:
                break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.eval()
        return self

    # ------------------------------------------------------------------ inference

    @property
    def network(self) -> LSTMNet:
        if self._model is None:
            raise RuntimeError("model not fitted; call fit() first")
        return self._model

    def _inputs(self, windows: np.ndarray, covariates: np.ndarray | None) -> torch.Tensor:
        x = np.asarray(windows, dtype=np.float32)[:, -self.input_window :, None]
        if self.n_covariates:
            if covariates is None:
                raise ValueError("model was trained with covariates; pass them at predict time")
            cov = np.asarray(covariates, dtype=np.float32)[:, -self.input_window :, :]
            x = np.concatenate([x, cov], axis=-1)
        return torch.from_numpy(np.ascontiguousarray(x))

    def predict_batch(self, windows: np.ndarray, horizon: int, **context: Any) -> np.ndarray:
        with torch.no_grad():
            out = self.network(self._inputs(windows, context.get("covariates")))
        return out.numpy()[:, :horizon]

    def predict(self, y_input: np.ndarray, horizon: int, **context: Any) -> np.ndarray:
        cov = context.get("covariates")
        cov = None if cov is None else np.asarray(cov)[None, ...]
        return self.predict_batch(np.asarray(y_input)[None, :], horizon, covariates=cov)[0]

    # ------------------------------------------------------------------ persistence

    def save(self, path: Path) -> None:
        torch.save({"state_dict": self.network.state_dict(), "config": self.config()}, path)

    def config(self) -> dict[str, Any]:
        return {
            "input_window": self.input_window,
            "max_horizon": self.max_horizon,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "n_covariates": self.n_covariates,
        }
