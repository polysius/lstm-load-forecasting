"""Sequence-to-sequence LSTM for direct multi-horizon forecasting (SPEC §7)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / "config" / "lstm.yaml"
_DATA_CONFIG = _ROOT / "config" / "data.yaml"


def load_config() -> dict:
    with open(_CONFIG) as f:
        return yaml.safe_load(f)


def load_data_config() -> dict:
    with open(_DATA_CONFIG) as f:
        return yaml.safe_load(f)


class LSTMModel(nn.Module):
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
        out = self.head(h_n[-1])
        return out


def _create_windows(
    y: np.ndarray, input_window: int, max_horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    n = len(y) - input_window - max_horizon + 1
    if n <= 0:
        raise ValueError(f"Not enough data for windows: {len(y)} < {input_window + max_horizon}")
    X = np.empty((n, input_window, 1), dtype=np.float32)
    Y = np.empty((n, max_horizon), dtype=np.float32)
    for i in range(n):
        X[i, :, 0] = y[i : i + input_window]
        Y[i, :] = y[i + input_window : i + input_window + max_horizon]
    return X, Y


class LSTMForecaster:
    def __init__(
        self,
        hidden_size: int | None = None,
        num_layers: int | None = None,
        dropout: float | None = None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
        max_epochs: int | None = None,
        patience: int | None = None,
        gradient_clip_norm: float | None = None,
        seed: int = 42,
    ):
        cfg = load_config()
        dcfg = load_data_config()

        self.hidden_size = hidden_size or cfg["model"]["hidden_size"]
        self.num_layers = num_layers or cfg["model"]["num_layers"]
        self.dropout = dropout if dropout is not None else cfg["model"]["dropout"]
        self.learning_rate = learning_rate or cfg["training"]["learning_rate"]
        self.batch_size = batch_size or cfg["training"]["batch_size"]
        self.max_epochs = max_epochs or cfg["training"]["max_epochs"]
        self.patience = patience or cfg["training"]["patience"]
        self.gradient_clip_norm = gradient_clip_norm or cfg["training"]["gradient_clip_norm"]
        self.seed = seed

        self.input_window = dcfg["task"]["input_window"]
        self.max_horizon = max(dcfg["task"]["forecast_horizons"])
        self.loss_fn_name = cfg["training"]["loss"]

        self._model: LSTMModel | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def name(self) -> str:
        return "lstm"

    def _set_seed(self) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def fit(self, y_train: np.ndarray, **kwargs) -> None:
        self._set_seed()

        X, Y = _create_windows(y_train, self.input_window, self.max_horizon)

        # Chronological train/val split (80/20)
        split = int(len(X) * 0.8)
        X_tr, X_val = X[:split], X[split:]
        Y_tr, Y_val = Y[:split], Y[split:]

        X_tr_t = torch.from_numpy(X_tr).to(self._device)
        Y_tr_t = torch.from_numpy(Y_tr).to(self._device)
        X_val_t = torch.from_numpy(X_val).to(self._device)
        Y_val_t = torch.from_numpy(Y_val).to(self._device)

        self._model = LSTMModel(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            max_horizon=self.max_horizon,
        ).to(self._device)

        loss_fn = nn.HuberLoss() if self.loss_fn_name == "huber" else nn.L1Loss()

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        n_batches = (len(X_tr_t) + self.batch_size - 1) // self.batch_size
        start = time.time()

        for epoch in range(self.max_epochs):
            self._model.train()
            epoch_loss = 0.0
            perm = torch.randperm(len(X_tr_t))

            for b in range(n_batches):
                idx = perm[b * self.batch_size : (b + 1) * self.batch_size]
                xb = X_tr_t[idx]
                yb = Y_tr_t[idx]

                pred = self._model(xb)
                loss = loss_fn(pred, yb)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), self.gradient_clip_norm)
                optimizer.step()
                epoch_loss += loss.item()

            # Validation
            self._model.eval()
            with torch.no_grad():
                val_pred = self._model(X_val_t)
                val_loss = loss_fn(val_pred, Y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or patience_counter >= self.patience:
                elapsed = time.time() - start
                print(
                    f"  Epoch {epoch+1:3d}  train_loss={epoch_loss/n_batches:.4f}  "
                    f"val_loss={val_loss:.4f}  best={best_val_loss:.4f}  "
                    f"patience={patience_counter}/{self.patience}  [{elapsed:.0f}s]"
                )

            if patience_counter >= self.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.eval()

    def predict(self, y_input: np.ndarray, horizon: int) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        x = np.asarray(y_input[-self.input_window :], dtype=np.float32).reshape(1, -1, 1)
        x_t = torch.from_numpy(x).to(self._device)

        with torch.no_grad():
            pred = self._model(x_t).cpu().numpy().ravel()

        return pred[:horizon]
