"""Smoke test: _build_pred_features row alignment.

Builds a DataFrame with 3 patients whose IDs are NOT in alphabetical
order in the file (Z1, A1, M1), verifies that the returned feature
DataFrame has:
  - result['pred'].values == pred exactly (no row permutation)
  - result.index == raw.index
"""
import numpy as np
import pandas as pd
import pytest

from parkinsons.two_stage import _build_pred_features


def _make_raw():
    """Return a raw DataFrame with file-order patients Z1, A1, M1."""
    rows = [
        # Z1 visits
        {"patient_id": "Z1", "age": 60.0},
        {"patient_id": "Z1", "age": 61.0},
        {"patient_id": "Z1", "age": 62.0},
        # A1 visits
        {"patient_id": "A1", "age": 55.0},
        {"patient_id": "A1", "age": 56.0},
        # M1 visits
        {"patient_id": "M1", "age": 70.0},
        {"patient_id": "M1", "age": 71.0},
        {"patient_id": "M1", "age": 72.0},
    ]
    return pd.DataFrame(rows)


def test_pred_column_equals_input_pred():
    raw = _make_raw()
    pred = np.arange(1.0, len(raw) + 1)   # [1,2,3,4,5,6,7,8] in raw order

    result = _build_pred_features(pred, raw)

    # result.index must equal raw.index (same labels, same order)
    assert result.index.equals(raw.index), (
        f"Index mismatch:\n  result: {result.index.tolist()}\n  raw: {raw.index.tolist()}"
    )

    # result['pred'] must equal pred in the original raw row order
    np.testing.assert_array_equal(
        result["pred"].values,
        pred,
        err_msg=(
            f"pred column mismatch.\n"
            f"  expected: {pred}\n"
            f"  got:      {result['pred'].values}"
        ),
    )
