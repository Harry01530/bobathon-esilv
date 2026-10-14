"""Patient-sequence model: a bidirectional GRU over all the visits of a patient.

Experiment 13. The visit-level HGB sees one visit at a time (plus a few patient
aggregates). The GRU reads the whole visit sequence of a patient at once and
predicts the true OFF at every visit (sequence labelling), so the level of the
patient's curve is estimated from all of their measurements jointly.

Inputs per visit
----------------
* raw block: time to the last visit, since the first visit, since diagnosis,
  age, OFF, ON, LEDD, intake times, a NaN flag for each, sex, cohort, gene;
* engineered block: every numeric column of the 09 feature matrix
  (``build_features`` with the bias calibration), standardised with the
  training means and standard deviations, clipped to ±6, NaN -> 0.

Network
-------
Linear(d_in, h) + GELU -> 2-layer bidirectional GRU(h) -> per visit:
[GRU output, input embedding, masked mean of the GRU outputs over the
patient's visits] -> MLP -> true OFF / 40. Loss: MSE over the real visits
(padding masked). AdamW, one-cycle learning rate, ``epochs`` passes.
Several seeds can be averaged (``n_seeds``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
except ImportError as e:  # pragma: no cover
    raise ImportError("parkinsons.seq_gru needs PyTorch: pip install torch") from e

SCALE = 40.0
_RAW_DUPLICATES = ("sexM", "age_at_diagnosis", "age", "ledd", "time_since_intake_on",
                   "time_since_intake_off", "on", "off")
GENES = ("GBA+", "LRRK2+", "No Mutation", "OTHER+")


def raw_block(X: pd.DataFrame) -> pd.DataFrame:
    """Timing, measurements, NaN flags and covariates of each visit (target-free)."""
    age = X["age"].to_numpy(float)
    g = X.groupby("patient_id")["age"]
    dx = X["age_at_diagnosis"].to_numpy(float)
    out = pd.DataFrame({
        "dt_last": (g.transform("max").to_numpy() - age) / 5,
        "since_first": (age - g.transform("min").to_numpy()) / 5,
        "years_dx": np.nan_to_num((age - dx) / 10), "age_c": (age - 65) / 10,
        "dx_c": np.nan_to_num((dx - 60) / 10), "dx_na": np.isnan(dx) * 1.0,
        "off": X["off"].fillna(0).to_numpy() / SCALE, "off_na": X["off"].isna().to_numpy() * 1.0,
        "on": X["on"].fillna(0).to_numpy() / SCALE, "on_na": X["on"].isna().to_numpy() * 1.0,
        "ledd": np.log1p(X["ledd"].fillna(0).to_numpy()) / 7, "ledd_na": X["ledd"].isna().to_numpy() * 1.0,
        "ton": X["time_since_intake_on"].fillna(0).to_numpy() / 10,
        "ton_na": X["time_since_intake_on"].isna().to_numpy() * 1.0,
        "toff": X["time_since_intake_off"].fillna(0).to_numpy() / 20,
        "toff_na": X["time_since_intake_off"].isna().to_numpy() * 1.0,
        "sexM": X["sexM"].to_numpy(float), "cohortB": (X["cohort"] == "B").to_numpy() * 1.0,
        "n_vis": g.transform("size").to_numpy() / 12,
        "gene_na": X["gene"].isna().to_numpy() * 1.0,
    }, index=X.index)
    for k in GENES:
        out[f"gene_{k}"] = (X["gene"] == k).to_numpy() * 1.0
    return out


class _Net(nn.Module):
    def __init__(self, d_in: int, h: int):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, h), nn.GELU())
        self.gru = nn.GRU(h, h, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(5 * h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, x, m, lens):
        L = x.shape[1]
        z = self.inp(x)
        o, _ = self.gru(pack_padded_sequence(z, lens, batch_first=True, enforce_sorted=False))
        o, _ = pad_packed_sequence(o, batch_first=True, total_length=L)
        ctx = (o * m[..., None]).sum(1) / m.sum(1, keepdim=True)
        ctx = ctx[:, None, :].expand(-1, L, -1)
        return self.head(torch.cat([o, z, ctx], -1)).squeeze(-1)


class SeqGRU:
    """Fit/predict a GRU on (raw visits, 09 feature matrix). Not an sklearn estimator:
    it is used inside ``HybridMLE``, which builds the feature matrix."""

    def __init__(self, hidden: int = 64, epochs: int = 60, batch_size: int = 128, lr: float = 3e-3,
                 weight_decay: float = 1e-4, n_seeds: int = 1, random_state: int = 0):
        self.hidden, self.epochs, self.batch_size = hidden, epochs, batch_size
        self.lr, self.weight_decay, self.n_seeds, self.random_state = lr, weight_decay, n_seeds, random_state

    # ── inputs ────────────────────────────────────────────────────────────────
    def _matrix(self, X, Xf, fit: bool):
        num = Xf.select_dtypes("number").drop(columns=[c for c in _RAW_DUPLICATES if c in Xf.columns])
        if fit:
            self.cols_ = list(num.columns)
            self.mu_ = num.mean()
            self.sd_ = num.std().replace(0, 1).fillna(1)
        num = num[self.cols_]
        eng = ((num - self.mu_) / self.sd_).clip(-6, 6).fillna(0)
        M = pd.concat([raw_block(X).reset_index(drop=True), eng.reset_index(drop=True)], axis=1)
        return M.to_numpy(np.float32)

    @staticmethod
    def _sequences(X, M, y=None):
        age = X["age"].to_numpy(float)
        idx = X["Index"].to_numpy(float) if "Index" in X.columns else np.arange(len(X), dtype=float)
        codes = pd.factorize(X["patient_id"])[0]
        order = np.lexsort((idx, age, codes))
        c = codes[order]
        starts = np.r_[0, np.flatnonzero(c[1:] != c[:-1]) + 1, len(order)]
        lens = np.diff(starts)
        n, L = len(lens), int(lens.max())
        SX = np.zeros((n, L, M.shape[1]), np.float32)
        SY = np.zeros((n, L), np.float32)
        SM = np.zeros((n, L), np.float32)
        SI = np.full((n, L), -1, np.int64)
        for i, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
            rows = order[a:b]
            SX[i, :b - a] = M[rows]
            SM[i, :b - a] = 1
            SI[i, :b - a] = rows
            if y is not None:
                SY[i, :b - a] = y[rows] / SCALE
        return SX, SY, SM, SI, lens.astype(np.int64)

    # ── fit / predict ─────────────────────────────────────────────────────────
    def fit(self, X: pd.DataFrame, Xf: pd.DataFrame, y):
        y = np.asarray(y, dtype=float)
        SX, SY, SM, _, LEN = self._sequences(X, self._matrix(X, Xf, fit=True), y)
        tx, ty, tm, tl = map(torch.from_numpy, (SX, SY, SM, LEN))
        n = len(SX)
        steps = self.epochs * int(np.ceil(n / self.batch_size))
        self.nets_ = []
        for s in range(self.n_seeds):
            seed = self.random_state + s
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            net = _Net(SX.shape[2], self.hidden)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps)
            for _ in range(self.epochs):
                net.train()
                perm = rng.permutation(n)
                for k in range(0, n, self.batch_size):
                    b = torch.from_numpy(perm[k:k + self.batch_size])
                    pred = net(tx[b], tm[b], tl[b])
                    loss = (((pred - ty[b]) ** 2) * tm[b]).sum() / tm[b].sum()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    sched.step()
            net.eval()
            self.nets_.append(net)
        return self

    @torch.no_grad()
    def predict(self, X: pd.DataFrame, Xf: pd.DataFrame) -> np.ndarray:
        SX, _, SM, SI, LEN = self._sequences(X, self._matrix(X, Xf, fit=False))
        tx, tm, tl = map(torch.from_numpy, (SX, SM, LEN))
        P = np.mean([net(tx, tm, tl).numpy() for net in self.nets_], axis=0)
        out = np.full(len(X), np.nan)
        msk = SM > 0
        out[SI[msk]] = P[msk] * SCALE
        return out
