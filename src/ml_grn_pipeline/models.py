
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import torch
from torch import nn
import torch.nn.functional as F

try:  # optional dependency
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional
    XGBRegressor = None  # type: ignore

try:  # optional dependency
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover - optional
    CatBoostRegressor = None  # type: ignore

from .config import TrainingConfig


def _rf_params(
    training: TrainingConfig,
    *,
    default_estimators: int,
    default_min_leaf: int = 2,
    default_max_features: float | str | None = None,
    default_bootstrap: bool = True,
) -> dict[str, object]:
    n_estimators = training.rf_n_estimators or default_estimators
    max_depth = training.rf_max_depth
    min_samples_leaf = training.rf_min_samples_leaf or default_min_leaf
    max_features = training.rf_max_features if training.rf_max_features is not None else default_max_features
    bootstrap = default_bootstrap if training.rf_bootstrap is None else training.rf_bootstrap

    params: dict[str, object] = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "n_jobs": -1,
        "random_state": training.random_state,
        "bootstrap": bootstrap,
        "oob_score": bootstrap,
    }

    if max_features is not None:
        params["max_features"] = max_features

    return params


@dataclass
class TorchModelBundle:
    model: nn.Module
    reshape: str = "flat"  # "flat" or "sequence"


class CNNRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        target_segments = max(8, min(128, max(1, input_dim // 32)))
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.pool = nn.AdaptiveAvgPool1d(target_segments)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * target_segments, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.backbone(x)
        x = self.pool(x)
        return self.head(x)


class RNNRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 96, num_layers: int = 1, output_dim: int = 1):
        super().__init__()
        target_segments = max(8, min(128, max(1, input_dim // 32)))
        self.project = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=8, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(target_segments),
        )
        dropout = 0.0 if num_layers <= 1 else 0.1
        self.rnn = nn.RNN(
            input_size=96,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.size(-1) == 1:
            x = x.transpose(1, 2)
        x = self.project(x)
        x = x.transpose(1, 2)
        output, _ = self.rnn(x)
        last = output[:, -1, :]
        return self.head(last)


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 1, output_dim: int = 1):
        super().__init__()
        target_segments = max(8, min(128, max(1, input_dim // 32)))
        self.project = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=8, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(target_segments),
        )
        dropout = 0.0 if num_layers <= 1 else 0.1
        self.lstm = nn.LSTM(
            input_size=96,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.size(-1) == 1:
            x = x.transpose(1, 2)
        x = self.project(x)
        x = x.transpose(1, 2)
        output, _ = self.lstm(x)
        last = output[:, -1, :]
        return self.head(last)


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: tuple[int, ...] = (256, 256, 128), output_dim: int = 1):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for units in hidden_layers:
            layers.append(nn.Linear(in_dim, units))
            layers.append(nn.LayerNorm(units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_dim = units
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        embed_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        target_segments = max(8, min(256, max(1, input_dim // 32)))
        self.project = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 96, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Conv1d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(target_segments)
        self.channel_proj = nn.Conv1d(128, embed_dim, kernel_size=1)
        self.positional = nn.Parameter(torch.randn(1, target_segments, embed_dim) * 0.02)
        head_options = [h for h in (8, 4, 2, 1) if embed_dim % h == 0]
        head_count = head_options[0] if head_options else 1
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=head_count,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.size(-1) == 1:
            x = x.transpose(1, 2)
        x = self.project(x)
        x = self.pool(x)
        x = self.channel_proj(x)
        x = x.transpose(1, 2)
        pos = self.positional[:, : x.size(1), :]
        x = x + pos
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x)


class GraphRegressor(nn.Module):
    """Simple message passing network over an implicit 1D graph of ATAC bins."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        max_nodes: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        num_nodes = max(8, min(max_nodes, max(1, input_dim // 8)))
        node_dim = math.ceil(input_dim / num_nodes)
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.pad_dim = (node_dim * num_nodes) - input_dim

        self.node_encoder = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.update_block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(hidden_dim * num_nodes),
            nn.Linear(hidden_dim * num_nodes, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, output_dim),
        )

        positions = torch.arange(num_nodes, dtype=torch.float32)
        distance = positions[:, None] - positions[None, :]
        adjacency = torch.exp(-(distance**2) / (2.0 * (num_nodes / 6.0) ** 2))
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True)
        self.register_buffer("adjacency", adjacency)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_dim > 0:
            x = F.pad(x, (0, self.pad_dim))
        nodes = x.view(x.size(0), self.num_nodes, self.node_dim)
        h = self.node_encoder(nodes)
        agg = torch.einsum("ij,bjd->bid", self.adjacency, h)
        h = h + self.update_block(agg)
        return self.head(h)


def build_model(
    name: str,
    input_dim: int,
    training: TrainingConfig,
    output_dim: int = 1,
    artifacts_dir: Optional[Path] = None,
) -> object:
    name = name.lower()
    if name == "cnn":
        return TorchModelBundle(CNNRegressor(input_dim, output_dim=output_dim))
    if name == "rnn":
        return TorchModelBundle(RNNRegressor(input_dim, output_dim=output_dim), reshape="sequence")
    if name == "lstm":
        return TorchModelBundle(LSTMRegressor(input_dim, output_dim=output_dim), reshape="sequence")
    if name == "transformer":
        return TorchModelBundle(TransformerRegressor(input_dim, output_dim=output_dim), reshape="sequence")
    if name == "mlp":
        return TorchModelBundle(MLPRegressor(input_dim, output_dim=output_dim))
    if name == "graph":
        return TorchModelBundle(GraphRegressor(input_dim, output_dim=output_dim), reshape="flat")
    if name == "svr":
        svr_estimator = SVR(C=10.0, epsilon=0.1, kernel="rbf", gamma="scale")
        if output_dim == 1:
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("regressor", svr_estimator),
                ]
            )
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", MultiOutputRegressor(svr_estimator)),
            ]
        )
    if name == "xgboost":
        if XGBRegressor is None:
            raise RuntimeError("xgboost is not installed. Install with `pip install xgboost` to enable this model.")
        base_model = XGBRegressor(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=training.random_state,
        )
        if output_dim == 1:
            return base_model
        return MultiOutputRegressor(base_model)
    if name == "random_forest":
        params = _rf_params(
            training,
            default_estimators=600,
            default_min_leaf=2,
            default_max_features=None,
        )
        return RandomForestRegressor(**params)
    if name == "catboost":
        if CatBoostRegressor is None:
            raise RuntimeError("catboost is not installed. Install with `pip install catboost` to enable this model.")
        catboost_params = {
            "iterations": 1200,
            "depth": 6,
            "learning_rate": 0.03,
            "loss_function": "RMSE",
            "verbose": False,
            "random_seed": training.random_state,
            "thread_count": -1,
        }
        if artifacts_dir is not None:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            catboost_params["train_dir"] = str(artifacts_dir)
            catboost_params["allow_writing_files"] = True
        else:
            catboost_params["allow_writing_files"] = False
        base_model = CatBoostRegressor(**catboost_params)
        if output_dim == 1:
            return base_model
        return MultiOutputRegressor(base_model)
    if name == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(
            n_estimators=800,
            max_depth=None,
            n_jobs=-1,
            random_state=training.random_state,
        )
    if name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        base_model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=6,
            max_iter=600,
            max_leaf_nodes=64,
            min_samples_leaf=20,
            l2_regularization=1e-3,
            random_state=training.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )
        if output_dim == 1:
            return base_model
        return MultiOutputRegressor(base_model)
    if name == "ridge":
        from sklearn.linear_model import Ridge

        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0, random_state=training.random_state)),
        ])
    if name == "elastic_net":
        from sklearn.linear_model import ElasticNet, MultiTaskElasticNet

        scaler = StandardScaler()
        if output_dim == 1:
            regressor = ElasticNet(
                alpha=0.05,
                l1_ratio=0.5,
                max_iter=10000,
                selection="cyclic",
                random_state=training.random_state,
            )
        else:
            regressor = MultiTaskElasticNet(
                alpha=0.05,
                l1_ratio=0.3,
                max_iter=10000,
                random_state=training.random_state,
            )
        return Pipeline([
            ("scaler", scaler),
            ("regressor", regressor),
        ])
    if name == "lasso":
        from sklearn.linear_model import Lasso, MultiTaskLasso

        scaler = StandardScaler()
        if output_dim == 1:
            regressor = Lasso(alpha=0.05, max_iter=10000, selection="cyclic", random_state=training.random_state)
        else:
            regressor = MultiTaskLasso(alpha=0.05, max_iter=10000, random_state=training.random_state)
        return Pipeline([
            ("scaler", scaler),
            ("regressor", regressor),
        ])
    if name == "ols":
        from sklearn.linear_model import LinearRegression

        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ])
    raise ValueError(f"Unknown model name: {name}")
