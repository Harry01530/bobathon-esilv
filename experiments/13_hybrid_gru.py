# %% [markdown]
# # 13_hybrid_gru
#
# HybridModel: 0.5 × mean(TwoStageMLE_A + TwoStageMLE_B).predict_unsmoothed
#              + 0.5 × SeqGRU(seed=0, n_seeds=1, epochs=60, hidden=64)
# then smooth_per_patient('quad_iso').
#
# Sources: scratch/test13_bob/parkinsons13 (isolated package)
#   seq_gru.py: ZIP test_13, _sequences tie-breaker Index added by Bob
#   ensemble_mle.py: written by Bob 2026-10-14 (no parkinsons.ensemble_mle dep)
#   two_stage_mle.py: commit a608a6f86a, MRO fix + relative imports by Bob
# Data: data/kaggle_official
# Hub workspace: julien.gouel/bobathon-esilv
#
# CLI:
#   python 13_hybrid_gru.py              -- full 5-fold CV + refit + Hub
#   python 13_hybrid_gru.py --resume     -- skip CV, reload scratch/test13_bob/13_cv_report.pkl
#   python 13_hybrid_gru.py --benchmark  -- 2 epochs / 200 patients, timing only, NO CV pickle / CSV / Hub

# %%
# ── Imports: standard library ─────────────────────────────────────────────────
import logging  # logging (console + file)
import os  # (imported but not used in this script)
import pickle  # save / reload the CV report and the final model
import re  # regex used to extract report URLs from console output
import signal  # SIGALRM-based timeout for the Hub upload
import sys  # command-line arguments and sys.path
import time  # timing measurements
from pathlib import Path  # file path handling

# ── Isolated package path (never touches src/) ───────────────────────────────
# Add the isolated scratch package folder to the front of the import path
_PKG13 = Path(__file__).resolve().parents[1] / "scratch" / "test13_bob"
if str(_PKG13) not in sys.path:
    sys.path.insert(0, str(_PKG13))

# ── Imports: data science ─────────────────────────────────────────────────────
import numpy as np  # numerical computing
import pandas as pd  # CSV loading / data manipulation
from sklearn.base import clone, is_regressor  # clone an unfitted estimator / check it is a regressor
from sklearn.model_selection import GroupKFold  # CV that keeps groups (patients) together

# ── Logging configuration ─────────────────────────────────────────────────────
# Log file placed next to the script
_LOG_FILE = Path(__file__).resolve().parent / "13_hybrid_gru_run.log"
logging.basicConfig(
    level=logging.INFO,  # minimum level displayed: INFO
    format="%(asctime)s  %(levelname)-8s  %(message)s",  # timestamp + level + message
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),  # print to console
        logging.FileHandler(_LOG_FILE, mode="w", encoding="utf-8"),  # write to file (overwritten at each run)
    ],
)
log = logging.getLogger(__name__)

# ── Mode flags ────────────────────────────────────────────────────────────────
# Command-line switches that select which path of the script runs
BENCHMARK = "--benchmark" in sys.argv
RESUME    = "--resume"    in sys.argv

# ── Paths ─────────────────────────────────────────────────────────────────────
# Project root = parent of the folder containing this script
PROJECT_ROOT  = Path(__file__).resolve().parents[1]
DATA_DIR      = PROJECT_ROOT / "data" / "kaggle_official"
# Output folder for submissions (created if it does not exist)
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)
SCRATCH_DIR   = _PKG13
CV_PICKLE     = SCRATCH_DIR / "13_cv_report.pkl"      # written only by full run
MODEL_PICKLE  = SCRATCH_DIR / "13_final_model.pkl"    # written only by full run
BENCH_TIMING  = SCRATCH_DIR / "13_benchmark_timing.json"  # written only by benchmark

# ── Load data ─────────────────────────────────────────────────────────────────
# Project helpers: default bias params, categorical dtypes helper, Hub credentials, hybrid model
from parkinsons13.features import BIAS_PARAMS, make_cat_dtypes
from parkinsons13.hub import load_skore_credentials
from parkinsons13.ensemble_mle import HybridModel

# Train / test features and the sample submission file
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
X_test_raw  = pd.read_csv(DATA_DIR / "X_test.csv")
sample_sub  = pd.read_csv(DATA_DIR / "sample_submission.csv")
# Target, reordered to match the row order of X_train_raw
y_full = (
    pd.read_csv(DATA_DIR / "y_train.csv")
    .set_index("Index")["target"]
    .loc[X_train_raw["Index"]]
    .values
)
# Patient id used as the CV group (avoids train/test leakage between patients)
groups_full = X_train_raw["patient_id"].values
# Consistent categorical dtypes across train and test
CAT_DTYPES = make_cat_dtypes(X_train_raw, X_test_raw)
# Best bias parameters: the defaults, with prior_scale set to 0.3
BEST_BIAS_PARAMS = {**BIAS_PARAMS, "prior_scale": 0.3}

log.info("Train: %d rows, %d patients", len(X_train_raw), len(np.unique(groups_full)))

import torch
# Limit PyTorch to 2 CPU threads
torch.set_num_threads(2)


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK PATH — measures GRU training time only, writes benchmark_timing.json
# NO CV pickle, NO production CSV, NO Hub upload.
# ═══════════════════════════════════════════════════════════════════════════════
def run_benchmark():
    # Imports done locally: only needed in benchmark mode
    from parkinsons13.features import build_features, calibrate_bias_model
    from parkinsons13.seq_gru import SeqGRU
    from parkinsons13.two_stage_mle import TwoStageMLE

    # Benchmark size: very small, just enough to measure speed
    BENCH_EPOCHS    = 2
    BENCH_PATIENTS  = 200

    # Randomly pick a subset of patients (fixed seed → reproducible)
    rng = np.random.default_rng(0)
    all_pids = np.unique(groups_full)
    chosen   = rng.choice(all_pids, size=min(BENCH_PATIENTS, len(all_pids)), replace=False)
    # Keep only the rows belonging to the chosen patients
    mask     = np.isin(groups_full, chosen)
    Xb       = X_train_raw.loc[mask].reset_index(drop=True)
    yb       = y_full[mask]
    log.info("BENCHMARK: %d patients, %d rows, %d epochs", len(chosen), len(Xb), BENCH_EPOCHS)

    # Fit member A calibration (train-only on subset)
    # Calibrate the bias model, then build the feature matrix, and time both steps
    t0 = time.monotonic()
    calib = calibrate_bias_model(Xb, yb)
    Xf    = build_features(Xb, CAT_DTYPES,
                           bias_calibration=calib,
                           bias_params=BEST_BIAS_PARAMS)
    t_feat = time.monotonic() - t0
    log.info("  features built in %.1fs", t_feat)

    # Time 1 GRU fit (2 epochs, subset)
    t1  = time.monotonic()
    gru = SeqGRU(hidden=64, epochs=BENCH_EPOCHS, batch_size=128,
                 lr=3e-3, weight_decay=1e-4, n_seeds=1, random_state=0)
    gru.fit(Xb, Xf, yb)
    t_gru_2ep = time.monotonic() - t1
    log.info("  GRU 2 epochs / %d patients: %.1fs", len(chosen), t_gru_2ep)

    # Extrapolate: 60 epochs full (5576 patients)
    # Scale linearly in number of epochs and in number of patients
    scale_ep  = 60 / BENCH_EPOCHS
    scale_pat = len(all_pids) / len(chosen)
    t_gru_full_est = t_gru_2ep * scale_ep * scale_pat
    # Full run: 5 folds × (2 HGB members + 1 GRU) + refit
    # Each fold: ~2×TwoStageMLE fit (say 3min each from exp12 data) + GRU
    t_hgb_fold_est = 6 * 60   # rough 6min per fold for 2 members
    t_per_fold     = t_hgb_fold_est + t_gru_full_est
    t_5cv_est      = 5 * t_per_fold
    t_refit_est    = t_per_fold        # one more for refit

    log.info(
        "ESTIMATE: GRU(60ep,full)=%.0fs  per_fold=%.0fs  5CV=%.0fs(%.1fmin)  +refit=%.0fs",
        t_gru_full_est, t_per_fold, t_5cv_est, t_5cv_est / 60, t_refit_est,
    )
    # Does the whole run (CV + refit) fit in the 75-minute budget?
    deadline_ok = (t_5cv_est + t_refit_est) < 75 * 60  # 75 min headroom before noon
    log.info("Fits before 11h45 deadline: %s", "YES" if deadline_ok else "NO — skip 13")

    import json
    # Record all measurements and estimates in a JSON file
    timing = {
        "benchmark_patients": int(len(chosen)),
        "benchmark_rows": int(len(Xb)),
        "gru_epochs_bench": BENCH_EPOCHS,
        "t_feat_s": round(t_feat, 1),
        "t_gru_2ep_s": round(t_gru_2ep, 1),
        "t_gru_full_est_s": round(t_gru_full_est, 1),
        "t_per_fold_est_s": round(t_per_fold, 1),
        "t_5cv_est_s": round(t_5cv_est, 1),
        "t_5cv_est_min": round(t_5cv_est / 60, 1),
        "t_refit_est_s": round(t_refit_est, 1),
        "fits_before_1145": deadline_ok,
        "note": "NO production artefacts written in benchmark mode",
    }
    BENCH_TIMING.write_text(json.dumps(timing, indent=2))
    log.info("Timing saved → %s", BENCH_TIMING)
    log.info("BENCHMARK DONE — check fits_before_1145 before launching full run.")


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK-EXTENDED PATH — warmup 2ep/200pat then 5ep/1000pat, same process.
# Writes 13_benchmark_extended_timing.json only. NO CV pickle / CSV / Hub.
# ═══════════════════════════════════════════════════════════════════════════════
def run_benchmark_extended():
    import json
    from parkinsons13.features import build_features, calibrate_bias_model
    from parkinsons13.seq_gru import SeqGRU

    # Output file and benchmark sizes (warmup run, then the measured run)
    BENCH_TIMING_EXT = SCRATCH_DIR / "13_benchmark_extended_timing.json"
    WARMUP_PATIENTS  = 200
    WARMUP_EPOCHS    = 2
    MEASURE_PATIENTS = 1000
    MEASURE_EPOCHS   = 5
    GRU_FULL_EPOCHS  = 60
    FULL_PATIENTS    = len(np.unique(groups_full))   # 5576

    # Fixed seed → reproducible patient subsets
    rng = np.random.default_rng(0)
    all_pids = np.unique(groups_full)

    # ── Warmup subset ─────────────────────────────────────────────────────────
    # Small throwaway run so PyTorch/optimizer initialisation does not skew the measurement
    wu_pids = rng.choice(all_pids, size=min(WARMUP_PATIENTS, len(all_pids)), replace=False)
    wu_mask = np.isin(groups_full, wu_pids)
    Xw = X_train_raw.loc[wu_mask].reset_index(drop=True)
    yw = y_full[wu_mask]
    calib_w = calibrate_bias_model(Xw, yw)
    Xfw = build_features(Xw, CAT_DTYPES, bias_calibration=calib_w, bias_params=BEST_BIAS_PARAMS)
    log.info("BENCH-EXT warmup: %d patients, %d rows, %d epochs", len(wu_pids), len(Xw), WARMUP_EPOCHS)
    SeqGRU(hidden=64, epochs=WARMUP_EPOCHS, batch_size=128, lr=3e-3,
           weight_decay=1e-4, n_seeds=1, random_state=0).fit(Xw, Xfw, yw)
    log.info("  warmup done (AdamW/PyTorch initialised)")

    # ── Measure subset ────────────────────────────────────────────────────────
    # Larger subset: this is the run whose timing is actually measured
    ms_pids = rng.choice(all_pids, size=min(MEASURE_PATIENTS, len(all_pids)), replace=False)
    ms_mask = np.isin(groups_full, ms_pids)
    Xm = X_train_raw.loc[ms_mask].reset_index(drop=True)
    ym = y_full[ms_mask]
    # Time the calibration + feature building
    t0 = time.monotonic()
    calib_m = calibrate_bias_model(Xm, ym)
    Xfm = build_features(Xm, CAT_DTYPES, bias_calibration=calib_m, bias_params=BEST_BIAS_PARAMS)
    t_feat = time.monotonic() - t0
    log.info("  measure features: %d patients, %d rows, %.1fs", len(ms_pids), len(Xm), t_feat)

    # Time one GRU fit on the measure subset
    t1 = time.monotonic()
    SeqGRU(hidden=64, epochs=MEASURE_EPOCHS, batch_size=128, lr=3e-3,
           weight_decay=1e-4, n_seeds=1, random_state=0).fit(Xm, Xfm, ym)
    t_gru_meas = time.monotonic() - t1
    log.info("  GRU %dep/%dpat: %.1fs", MEASURE_EPOCHS, len(ms_pids), t_gru_meas)

    # ── Extrapolation ─────────────────────────────────────────────────────────
    # 5 CV folds (80% train ~4460 pat) + 1 refit (full 5576 pat) = 5 GRU fits total
    # Use mean patient count across CV folds: 80% × FULL_PATIENTS
    cv_train_patients = int(0.8 * FULL_PATIENTS)
    # Time cost per epoch per patient, then scaled to full epochs and patient counts
    t_gru_per_ep_per_pat = t_gru_meas / (MEASURE_EPOCHS * len(ms_pids))
    t_gru_cv_fold   = t_gru_per_ep_per_pat * GRU_FULL_EPOCHS * cv_train_patients
    t_gru_refit     = t_gru_per_ep_per_pat * GRU_FULL_EPOCHS * FULL_PATIENTS
    t_gru_total_est = 5 * t_gru_cv_fold + t_gru_refit   # 5 folds + 1 refit

    # HGB: measured from exp.12 — 19m38s for 5-fold CV + refit of 2 members.
    # Exp.13 adds member B (800 iter vs 600) → budget 30 min conservatively.
    t_hgb_total = 30 * 60

    # Total estimated duration of a full run
    t_total_est = t_hgb_total + t_gru_total_est

    # Deadline: 11h45 real wall clock
    # Compute how many seconds remain until the next 11:45
    import datetime
    now = datetime.datetime.now()
    deadline = now.replace(hour=11, minute=45, second=0, microsecond=0)
    if deadline < now:
        deadline += datetime.timedelta(days=1)
    seconds_left = (deadline - now).total_seconds()
    fits_before_1145 = t_total_est < seconds_left

    log.info(
        "ESTIMATE: t_gru_cv_fold=%.0fs t_gru_refit=%.0fs t_gru_total=%.0fs "
        "t_hgb_total=%.0fs t_TOTAL=%.0fs(%.1fmin)  seconds_left=%.0fs  fits=<%s>",
        t_gru_cv_fold, t_gru_refit, t_gru_total_est,
        t_hgb_total, t_total_est, t_total_est / 60,
        seconds_left, "YES" if fits_before_1145 else "NO — skip 13",
    )

    # Record all measurements and estimates in a JSON file
    timing = {
        "warmup_patients": int(len(wu_pids)),
        "warmup_epochs": WARMUP_EPOCHS,
        "measure_patients": int(len(ms_pids)),
        "measure_rows": int(len(Xm)),
        "measure_epochs": MEASURE_EPOCHS,
        "t_feat_s": round(t_feat, 1),
        "t_gru_meas_s": round(t_gru_meas, 1),
        "t_gru_per_ep_per_pat_ms": round(t_gru_per_ep_per_pat * 1000, 4),
        "t_gru_cv_fold_est_s": round(t_gru_cv_fold, 1),
        "t_gru_refit_est_s": round(t_gru_refit, 1),
        "t_gru_total_est_s": round(t_gru_total_est, 1),
        "t_hgb_total_s": t_hgb_total,
        "t_hgb_note": "measured exp.12 19m38s, +30% margin for deeper member B",
        "t_total_est_s": round(t_total_est, 1),
        "t_total_est_min": round(t_total_est / 60, 1),
        "seconds_left_at_benchmark": round(seconds_left, 0),
        "fits_before_1145": fits_before_1145,
        "note": "NO production artefacts written in benchmark-extended mode",
    }
    BENCH_TIMING_EXT.write_text(json.dumps(timing, indent=2))
    log.info("Extended timing saved → %s", BENCH_TIMING_EXT)
    log.info("BENCHMARK-EXTENDED DONE — check fits_before_1145.")


# ═══════════════════════════════════════════════════════════════════════════════
# FULL RUN PATH — 5-fold CV, refit, CSV, Hub upload
# ═══════════════════════════════════════════════════════════════════════════════
def run_full():
    import skore
    from skore import login

    # The hybrid model: TwoStageMLE members + GRU, as described at the top of the file
    model = HybridModel(
        cat_dtypes=CAT_DTYPES,
        bias_params=BEST_BIAS_PARAMS,
        gru_epochs=60,
        gru_hidden=64,
        gru_random_state=0,
    )
    # skore needs sklearn to recognise the model as a regressor
    assert is_regressor(model), "HybridModel not recognised as regressor"

    # 5-fold split where a patient never appears in both train and test
    eval_splits = list(
        GroupKFold(n_splits=5).split(X_train_raw, y_full, groups=groups_full)
    )
    # Safety check: no patient shared between train and test in any fold
    for k, (tr, te) in enumerate(eval_splits):
        assert not (set(groups_full[tr]) & set(groups_full[te])), f"Patient leak fold {k}"

    # ── CV (or resume from pickle) ─────────────────────────────────────────────
    if RESUME and CV_PICKLE.exists():
        # --resume: skip the (long) CV and reload the saved report
        log.info("--resume: loading CV from %s", CV_PICKLE)
        with open(CV_PICKLE, "rb") as f:
            cv = pickle.load(f)
    else:
        # Run the cross-validation and save the report to disk
        t0 = time.monotonic()
        log.info("Running skore.evaluate(HybridModel, GroupKFold5) …")
        cv = skore.evaluate(clone(model), X_train_raw, y_full, splitter=eval_splits)
        log.info("CV done in %.1fs", time.monotonic() - t0)
        with open(CV_PICKLE, "wb") as f:
            pickle.dump(cv, f)
        log.info("CV pickled → %s", CV_PICKLE)

    # Per-fold RMSE on the test side (aggregate=None → one value per fold)
    df_rmse = cv.metrics.rmse(data_source="test", aggregate=None)
    folds   = df_rmse.values.flatten().tolist()
    arr     = np.array(folds)
    log.info(
        "HybridModel  folds=%s  mean=%.4f  std=%.4f",
        [round(v, 3) for v in folds], arr.mean(), arr.std(),
    )

    # ── Refit ─────────────────────────────────────────────────────────────────
    # Retrain on 100% of the training data (fresh clone) and save the model
    log.info("Refitting on all %d rows …", len(X_train_raw))
    final = clone(model).fit(X_train_raw, y_full)
    with open(MODEL_PICKLE, "wb") as f:
        pickle.dump(final, f)
    log.info("Model pickled → %s", MODEL_PICKLE)

    # ── CSV ───────────────────────────────────────────────────────────────────
    # Test predictions clipped to the valid target range [0, 132], then sanity-checked
    preds = np.clip(final.predict(X_test_raw), 0.0, 132.0)
    assert len(preds) == len(sample_sub) and np.all(np.isfinite(preds))
    # Build the submission, reordered to follow sample_submission
    submission = (
        pd.DataFrame({"Index": X_test_raw["Index"].values, "target": preds})
        .set_index("Index").loc[sample_sub["Index"]].reset_index()
    )
    # Format check: expected columns and number of rows
    assert list(submission.columns) == ["Index", "target"] and len(submission) == 11013

    # Default Hub keys and output CSV path
    est_key = "13_fold5_hybrid_gru"
    cv_key  = "13_cv_hybrid_gru"
    csv_path = SUBMISSIONS_DIR / f"{est_key}.csv"
    submission.to_csv(csv_path, index=False)
    log.info("CSV → %s  (%d rows)", csv_path, len(submission))

    # ── Hub upload — alarm covers login/Project/summarize/puts ────────────────
    # Patch HUBClient.__init__ with retry=False BEFORE login/Project
    # (disables automatic retries so the upload cannot exceed the time budget)
    try:
        import skore._plugins.hub.client.client as _hmod
        _Orig = _hmod.HUBClient
        # Only patch once, even if this code runs several times
        if not getattr(_Orig, "_retry_false_patched", False):
            _orig_init = _Orig.__init__
            def _patched_init(self_c, *a, **kw):
                kw["retry"] = False
                _orig_init(self_c, *a, **kw)
            _Orig.__init__ = _patched_init
            _Orig._retry_false_patched = True
        log.info("HUBClient patched: retry=False")
    except Exception as _ep:
        # If the patch fails (e.g. skore version change), carry on without it
        log.warning("HUBClient patch skipped: %s", _ep)

    # Default values used if the upload fails or times out
    est_url = "not-uploaded"
    cv_url  = "not-uploaded"
    est_key_real = est_key
    cv_key_real  = cv_key

    # Custom exception raised by the alarm handler when the budget is exhausted
    class _HubTimeout(Exception):
        pass

    def _alarm(signum, frame):
        raise _HubTimeout("180s Hub budget exhausted")

    # signal.alarm only exists on Unix; skip the timeout mechanism elsewhere (e.g. Windows)
    _has_alarm = hasattr(signal, "alarm")
    if _has_alarm:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(180)   # covers login + Project + summarize + all puts

    try:
        # Load credentials, log in, and open the project on the Hub
        cfg = load_skore_credentials()
        login(mode="hub")
        project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])

        # Keys already in the project, to avoid overwriting existing reports
        try:
            existing_keys: set[str] = set(project.summarize().frame()["key"].tolist())
        except Exception:
            existing_keys = set()

        def free_key(base):
            # Returns `base` if free, otherwise `base_2`, `base_3`, … until an unused key is found
            if base not in existing_keys:
                return base
            i = 2
            while f"{base}_{i}" in existing_keys:
                i += 1
            return f"{base}_{i}"

        # Pick collision-free keys for the two reports
        est_key_real = free_key(est_key)
        cv_key_real  = free_key(cv_key)

        # EstimatorReport first (fold 4), capture URL immediately
        # (the URL is printed to the rich console by project.put, so we capture it)
        skore.console.begin_capture()
        project.put(est_key_real, cv.reports_[4])
        _cap = skore.console.end_capture()
        existing_keys.add(est_key_real)
        _urls = re.findall(r"https://[^\s\]>\"]+", _cap)
        est_url = _urls[-1].rstrip("/") if _urls else "url-not-captured"
        log.info("Est  key=%s  url=%s", est_key_real, est_url)

        # Save metadata immediately after first put
        # (so something is on disk even if the next upload times out)
        _save_meta(folds, arr, est_key_real, cv_key_real, est_url, cv_url, csv_path, BENCH_TIMING)

        # CV report
        # Same capture trick; fall back to the estimator URL if none is found
        skore.console.begin_capture()
        project.put(cv_key_real, cv)
        _cap2 = skore.console.end_capture()
        existing_keys.add(cv_key_real)
        _urls2 = re.findall(r"https://[^\s\]>\"]+", _cap2)
        cv_url = _urls2[-1].rstrip("/") if _urls2 else est_url
        log.info("CV   key=%s  url=%s", cv_key_real, cv_url)

    except _HubTimeout as _ht:
        # Time budget exceeded: keep going, local artefacts are already saved
        log.warning("Hub timeout: %s — artefacts preserved", _ht)
    except Exception as _he:
        # Any other upload error does not stop the script
        log.error("Hub failed: %s: %s — artefacts preserved", type(_he).__name__, _he)
    finally:
        # Always cancel the pending alarm
        if _has_alarm:
            signal.alarm(0)

    # Final metadata (with real cv_url if captured)
    _save_meta(folds, arr, est_key_real, cv_key_real, est_url, cv_url, csv_path, BENCH_TIMING)

    # Final recap framed by lines of "="
    log.info(
        "\n%s\nHybridModel  folds=%s  mean=%.4f  std=%.4f"
        "\nEst  key: %-35s  url: %s"
        "\nCV   key: %-35s  url: %s"
        "\nCSV: %s  (%d rows)\n%s",
        "=" * 66,
        [round(v, 3) for v in folds], arr.mean(), arr.std(),
        est_key_real, est_url, cv_key_real, cv_url,
        csv_path, len(submission), "=" * 66,
    )
    log.info("Done — stop before Kaggle upload.")


def _save_meta(folds, arr, est_key_real, cv_key_real, est_url, cv_url, csv_path, bench_timing_path):
    # Writes the experiment metadata JSON (provenance, CV scores, Hub keys/URLs, CSV path)
    import json
    meta = {
        "experiment": "13_hybrid_gru",
        "provenance": {
            "two_stage_mle": "github.com/Harry01530/bobathon-esilv@a608a6f86a MRO-fixed relative-imports",
            "seq_gru": "scratch/test13_bob/test_13/seq_gru.py (Index tie-breaker added)",
            "ensemble_mle": "scratch/test13_bob/parkinsons13/ensemble_mle.py (Bob 2026-10-14)",
        },
        "gru_epochs": 60,
        "gru_seeds": 1,
        "cv_folds_rmse": [round(v, 4) for v in folds],
        "cv_mean_rmse": round(float(arr.mean()), 4),
        "cv_std_rmse":  round(float(arr.std()),  4),
        "hub_est_key": est_key_real,
        "hub_cv_key":  cv_key_real,
        "hub_est_url": est_url,
        "hub_cv_url":  cv_url,
        "csv_path": str(csv_path),
    }
    # JSON file is written next to the submission CSV
    meta_path = Path(csv_path).parent / "13_hybrid_gru_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))


# Extended-benchmark flag (defined here, after the functions, but before the entry point)
BENCHMARK_EXT = "--benchmark-extended" in sys.argv

# Entry point: pick the mode from the command-line flags
if __name__ == "__main__":
    if BENCHMARK_EXT:
        run_benchmark_extended()
    elif BENCHMARK:
        run_benchmark()
    else:
        run_full()
