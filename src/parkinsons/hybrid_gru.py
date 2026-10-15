"""Hybrid HGB + bidirectional GRU model for Parkinson's UPDRS prediction.

Architecture
------------
**HGB branch** — unweighted average of two ``TwoStageMLE(smooth=None)`` models
(one shallow, one deep).  Smoothing is deliberately omitted here so the GRU
branch can learn from the unsmoothed signal.

**GRU branch** — a 2-layer bidirectional GRU (hidden=64, dropout=0.1) that
reads each patient's visit sequence sorted by age.  At each time step the
input is a 6-dimensional longitudinal vector:
  [age_norm, off_norm, on_norm, t_on_norm, t_off_norm, hgb_pred_norm]
Missing values are filled with 0 after normalisation (missing = mean, which
maps to 0 under z-score normalisation).  Sequences are zero-padded to the
maximum patient length and a boolean key-padding mask is passed to the GRU.
The GRU produces a per-visit output; its last valid hidden state is projected
to a scalar residual and added to the HGB prediction for that patient.

**Final prediction** — ``smooth_per_patient(0.5 * HGB + 0.5 * GRU, "quad_iso")``
applied to every output, ensuring non-decreasing per-patient predictions.

Leakage contract
----------------
``HybridGRUModel.fit(X, y)`` calibrates ``calibrate_bias_model`` and
standardisation statistics (mean / std of the 5 longitudinal columns) on the
``(X, y)`` it receives only — never on the full training set when called from
inside a CV fold.  ``predict`` reuses the fitted objects stored on ``self``.

``patient_id`` and ``age`` must be columns of ``X``; they are used for
grouping and ordering only and are never passed to any sklearn estimator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted

from parkinsons.features import BIAS_PARAMS, make_cat_dtypes, calibrate_bias_model
from parkinsons.two_stage_mle import TwoStageMLE, smooth_per_patient

# ── Longitudinal columns fed to the GRU (in this order) ──────────────────────
_SEQ_COLS = ["age", "off", "on", "time_since_intake_on", "time_since_intake_off"]
_N_SEQ_FEATURES = len(_SEQ_COLS) + 1   # +1 for the HGB prediction


# ══════════════════════════════════════════════════════════════════════════════
# PyTorch module
# ══════════════════════════════════════════════════════════════════════════════


class _GRUNet(nn.Module):
    """2-layer bidirectional GRU → scalar residual per visit."""

    def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.head = nn.Linear(hidden_size * 2, 1)  # *2 for bidirectional

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Run GRU on a padded batch, return per-visit residuals.

        Parameters
        ----------
        x : (B, T, F) — padded sequence batch
        lengths : (B,) — true sequence length for each sample

        Returns
        -------
        (B, T) — residual scalar per time step (only valid steps matter)
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return self.head(out).squeeze(-1)  # (B, T)


# ══════════════════════════════════════════════════════════════════════════════
# sklearn estimator
# ══════════════════════════════════════════════════════════════════════════════


class HybridGRUModel(BaseEstimator, RegressorMixin):
    """HGB ensemble + bidirectional GRU for UPDRS motor score regression.

    Parameters
    ----------
    cat_dtypes : dict[str, pd.CategoricalDtype] or None
        Shared category mapping from ``make_cat_dtypes``.  Required for
        ``TwoStageMLE`` (``categorical_features="from_dtype"``).
    bias_params : dict or None
        Keyword overrides for ``features.BIAS_PARAMS``.
    hidden_size : int, default 64
        GRU hidden dimension (bidirectional → 2 × hidden_size in output).
    dropout : float, default 0.1
        Dropout applied between GRU layers.
    lr : float, default 3e-3
        Adam learning rate for GRU training.
    epochs : int, default 30
        Training epochs for the GRU.
    batch_size : int, default 64
        Number of *patients* per batch.
    seed : int, default 0
        RNG seed for reproducibility.
    device : str or None, default None
        Torch device (``"cpu"``, ``"cuda"``).  ``None`` → auto-detect.
    """

    def __init__(
        self,
        cat_dtypes=None,
        bias_params=None,
        hidden_size: int = 64,
        dropout: float = 0.1,
        lr: float = 3e-3,
        epochs: int = 30,
        batch_size: int = 64,
        seed: int = 0,
        device: str | None = None,
    ):
        self.cat_dtypes = cat_dtypes
        self.bias_params = bias_params
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = device

    # ── internal helpers ──────────────────────────────────────────────────────

    def _resolve_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _make_hgb_models(self):
        bp = {**BIAS_PARAMS, **(self.bias_params or {})}
        model_a = TwoStageMLE(
            cat_dtypes=self.cat_dtypes,
            inner_n_splits=5,
            smooth=None,  # smoothing applied only at the end, after blending
            bias_params=bp,
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
        )
        model_b = TwoStageMLE(
            cat_dtypes=self.cat_dtypes,
            inner_n_splits=5,
            smooth=None,
            bias_params=bp,
            max_iter=800,
            learning_rate=0.03,
            max_leaf_nodes=63,
            min_samples_leaf=50,
        )
        return model_a, model_b

    def _standardize_seq_cols(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        """Extract and z-score the 5 longitudinal columns (+ slot for hgb later).

        Parameters
        ----------
        X : raw dataframe
        fit : if True, compute and store mean_/std_ on X

        Returns
        -------
        (N, 5) float array, NaN → 0 after z-score
        """
        arr = X[_SEQ_COLS].to_numpy(dtype=float)
        if fit:
            self.seq_mean_ = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
            std[std < 1e-8] = 1.0
            self.seq_std_ = std
        normed = (arr - self.seq_mean_) / self.seq_std_
        normed = np.where(np.isfinite(normed), normed, 0.0)
        return normed

    def _build_sequences(
        self,
        X: pd.DataFrame,
        seq_normed: np.ndarray,
        hgb_pred: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        """Pack rows into per-patient sequences sorted by age.

        Returns
        -------
        seqs : list of (L_p, F) arrays — one per patient, age-sorted
        targets : list of (L_p,) arrays — y values aligned to seqs
        order : (N,) int array — mapping from original row index to sequence
                position (used to scatter predictions back into raw order)
        """
        # sort rows by (patient_id, age) to get stable sequence order
        sort_order = X.sort_values(["patient_id", "age"], kind="stable").index

        # row-level HGB prediction, normalised to zero-mean unit-std
        if not hasattr(self, "hgb_pred_mean_"):
            raise RuntimeError("call _fit_hgb before _build_sequences")
        hp_norm = (hgb_pred - self.hgb_pred_mean_) / max(self.hgb_pred_std_, 1e-8)

        # build the (N, 6) input matrix in sorted order
        sorted_seq = seq_normed[X.index.get_indexer(sort_order)]
        sorted_hp = hp_norm[X.index.get_indexer(sort_order)]
        full = np.concatenate([sorted_seq, sorted_hp[:, None]], axis=1)

        pids = X.loc[sort_order, "patient_id"].to_numpy()
        starts = np.r_[0, np.flatnonzero(pids[1:] != pids[:-1]) + 1, len(pids)]

        seqs = []
        orig_positions = []  # sorted-order positions per patient
        for a, b in zip(starts[:-1], starts[1:]):
            seqs.append(full[a:b])
            orig_positions.append(np.arange(a, b))

        return seqs, orig_positions, sort_order

    def _collate(
        self,
        patient_seqs: list[np.ndarray],
        y_patient: list[np.ndarray],
        device: torch.device,
    ):
        """Pad a list of patient sequences and targets into a batch tensor."""
        lengths = torch.tensor([len(s) for s in patient_seqs], dtype=torch.long)
        T = int(lengths.max().item())
        F = patient_seqs[0].shape[1]
        B = len(patient_seqs)
        x = torch.zeros(B, T, F)
        y = torch.zeros(B, T)
        mask = torch.zeros(B, T, dtype=torch.bool)
        for i, (s, yt) in enumerate(zip(patient_seqs, y_patient)):
            L = len(s)
            x[i, :L] = torch.tensor(s, dtype=torch.float32)
            y[i, :L] = torch.tensor(yt, dtype=torch.float32)
            mask[i, :L] = True
        return x.to(device), y.to(device), mask.to(device), lengths

    # ── fit/predict ───────────────────────────────────────────────────────────

    def _fit_hgb(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Fit both HGB models; return their averaged unsmoothed test predictions
        on train (OOF via inner GroupKFold to avoid leaking y to GRU training)."""
        ma, mb = self._make_hgb_models()
        self.hgb_a_ = clone(ma).fit(X, y)
        self.hgb_b_ = clone(mb).fit(X, y)
        # OOF predictions for GRU input (keep HGB from leaking y to GRU)
        from sklearn.model_selection import GroupKFold
        groups = X["patient_id"].to_numpy()
        oof = np.full(len(y), np.nan)
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups=groups):
            ma_fold = clone(ma).fit(X.iloc[tr], y[tr])
            mb_fold = clone(mb).fit(X.iloc[tr], y[tr])
            oof[te] = 0.5 * (
                ma_fold.predict_unsmoothed(X.iloc[te])
                + mb_fold.predict_unsmoothed(X.iloc[te])
            )
        assert np.all(np.isfinite(oof)), "OOF HGB predictions contain NaN"
        # normalisation statistics for HGB prediction column
        self.hgb_pred_mean_ = float(np.mean(oof))
        self.hgb_pred_std_ = float(np.std(oof))
        return oof

    def _hgb_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Unsmoothed average of the two fitted HGB models."""
        return 0.5 * (
            self.hgb_a_.predict_unsmoothed(X) + self.hgb_b_.predict_unsmoothed(X)
        )

    def fit(self, X: pd.DataFrame, y) -> "HybridGRUModel":
        """Fit both HGB models then train the GRU on their OOF residuals.

        All calibration (``calibrate_bias_model``, z-score statistics) is
        computed exclusively on ``(X, y)`` — never on the full training set
        when called inside a GroupKFold fold.
        """
        y = np.asarray(y, dtype=float)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        dev = self._resolve_device()

        # ── Step 1: bias model calibration (training rows only) ───────────────
        # TwoStageMLE.fit already calls calibrate_bias_model on its own training
        # rows; no explicit call needed here.

        # ── Step 2: fit both HGB models + get OOF predictions ─────────────────
        hgb_oof = self._fit_hgb(X, y)

        # ── Step 3: fit z-score statistics for sequential input ───────────────
        seq_normed = self._standardize_seq_cols(X, fit=True)

        # ── Step 4: build per-patient sequences ───────────────────────────────
        seqs, orig_positions, sort_order = self._build_sequences(
            X, seq_normed, hgb_oof
        )

        # residual that the GRU should learn to predict
        y_sorted = y[X.index.get_indexer(sort_order)]
        hgb_sorted = hgb_oof[X.index.get_indexer(sort_order)]
        residual_sorted = y_sorted - hgb_sorted

        # group residuals per patient (same patient order as seqs)
        pids = X.loc[sort_order, "patient_id"].to_numpy()
        starts = np.r_[0, np.flatnonzero(pids[1:] != pids[:-1]) + 1, len(pids)]
        y_patients = [
            residual_sorted[a:b] for a, b in zip(starts[:-1], starts[1:])
        ]

        # ── Step 5: train GRU ─────────────────────────────────────────────────
        self.gru_net_ = _GRUNet(
            input_size=_N_SEQ_FEATURES,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        ).to(dev)

        optimiser = torch.optim.Adam(self.gru_net_.parameters(), lr=self.lr)
        n_patients = len(seqs)
        rng = np.random.default_rng(self.seed)

        self.gru_net_.train()
        for epoch in range(self.epochs):
            idx = rng.permutation(n_patients)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_patients, self.batch_size):
                batch_idx = idx[start : start + self.batch_size].tolist()
                batch_seqs = [seqs[i] for i in batch_idx]
                batch_y = [y_patients[i] for i in batch_idx]
                xb, yb, maskb, lengths = self._collate(batch_seqs, batch_y, dev)

                optimiser.zero_grad()
                out = self.gru_net_(xb, lengths)  # (B, T)
                loss = ((out - yb) ** 2 * maskb).sum() / maskb.sum()
                loss.backward()
                nn.utils.clip_grad_norm_(self.gru_net_.parameters(), 5.0)
                optimiser.step()
                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  GRU epoch {epoch + 1:3d}/{self.epochs}"
                    f"  loss={epoch_loss / max(n_batches, 1):.4f}",
                    flush=True,
                )

        self.gru_net_.eval()
        self.n_features_in_ = len(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict UPDRS scores: blend HGB and GRU, then apply quad_iso smoothing."""
        check_is_fitted(self, ["hgb_a_", "hgb_b_", "gru_net_"])
        dev = self._resolve_device()

        # ── HGB branch ────────────────────────────────────────────────────────
        hgb_pred = self._hgb_predict(X)

        # ── GRU branch ────────────────────────────────────────────────────────
        seq_normed = self._standardize_seq_cols(X, fit=False)
        seqs, orig_positions, sort_order = self._build_sequences(
            X, seq_normed, hgb_pred
        )

        # dummy y for _collate (shape only; values unused at inference)
        dummy_y = [np.zeros(len(s)) for s in seqs]

        gru_residual_sorted = np.zeros(len(X))
        self.gru_net_.eval()
        with torch.no_grad():
            for start in range(0, len(seqs), self.batch_size):
                batch_seqs = seqs[start : start + self.batch_size]
                batch_y = dummy_y[start : start + self.batch_size]
                positions = orig_positions[start : start + self.batch_size]
                xb, _, _, lengths = self._collate(batch_seqs, batch_y, dev)
                out = self.gru_net_(xb, lengths).cpu().numpy()  # (B, T)
                for i, (pos, length) in enumerate(
                    zip(positions, lengths.tolist())
                ):
                    gru_residual_sorted[pos] = out[i, :length]

        # scatter from sorted order back to original X row order
        gru_residual = np.empty(len(X))
        gru_residual[X.index.get_indexer(sort_order)] = gru_residual_sorted

        # ── Blend ─────────────────────────────────────────────────────────────
        gru_pred = np.clip(hgb_pred + gru_residual, 0.0, 132.0)
        blended = np.clip(0.5 * hgb_pred + 0.5 * gru_pred, 0.0, 132.0)

        # ── Smoothing ─────────────────────────────────────────────────────────
        return smooth_per_patient(X, blended, "quad_iso")
