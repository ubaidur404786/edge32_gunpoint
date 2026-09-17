"""Load the UCR GunPoint TSV files, map labels to 0..K-1, stratified train/val split.

Run from the project root:  python -m src.dataset
That prints the dataset statistics and writes results/data_stats.json.
"""

import json            # write the measured statistics to results/data_stats.json
import numpy as np     # all array work: loading, statistics, splitting

import config          # paths, seed, validation fraction


def load_tsv(path):
    """Read one UCR .tsv file.

    Returns X of shape (rows, length) as float32 and y of shape (rows,) as int64 with
    the ORIGINAL labels (1, 2, ...). Column 0 is the label, the rest is the series.
    """
    raw = np.loadtxt(path, delimiter="\t", dtype=np.float64)
    y_raw = raw[:, 0].astype(np.int64)
    X = raw[:, 1:].astype(np.float32)
    return X, y_raw


def map_labels(y_raw, classes):
    """Map original labels (e.g. 1, 2) to 0..K-1 in sorted order. `classes` is the sorted
    array of original labels found in TRAIN; TEST must not contain anything else."""
    lookup = {int(c): i for i, c in enumerate(classes)}
    unknown = set(np.unique(y_raw).tolist()) - set(lookup)
    if unknown:
        raise ValueError(f"labels {unknown} not present in TRAIN")
    return np.array([lookup[int(v)] for v in y_raw], dtype=np.int64)


def stratified_split(X, y, val_fraction, seed):
    """Hold out `val_fraction` of each class for validation, with a fixed seed.

    Returns (X_tr, y_tr, X_val, y_val). Done per class so both parts keep the class ratio.
    """
    rng = np.random.default_rng(seed)
    tr_idx, val_idx = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n_val = int(round(val_fraction * len(idx)))
        val_idx.append(idx[:n_val])
        tr_idx.append(idx[n_val:])
    tr_idx = np.concatenate(tr_idx)
    val_idx = np.concatenate(val_idx)
    rng.shuffle(tr_idx)   # so batches are not sorted by class
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


def to_conv1d(X):
    """(rows, length) -> (rows, 1, length): the (batch, channels, length) layout that
    PyTorch's Conv1d expects. One channel because each time step is a single value."""
    return X[:, None, :].astype(np.float32)

def to_channels_last(X):
    """(rows, length) -> (rows, length, 1): the (batch, length, channels) layout of the
    TFLite model. Same data as to_conv1d, only the axis order differs."""
    return X[:, :, None].astype(np.float32)

def load_all():
    """Everything the other scripts need, in one call.

    Returns a dict with X_train/y_train (the part we fit on), X_val/y_val (model selection),
    X_test/y_test (the official TEST split, evaluated ONCE at the end), plus series_length.
    """
    X_full, y_full_raw = load_tsv(config.TRAIN_TSV)
    X_test, y_test_raw = load_tsv(config.TEST_TSV)
    classes = np.unique(y_full_raw)
    y_full = map_labels(y_full_raw, classes)
    y_test = map_labels(y_test_raw, classes)
    X_tr, y_tr, X_val, y_val = stratified_split(X_full, y_full, config.VAL_FRACTION, config.SEED)
    return {
        "X_train": X_tr, "y_train": y_tr,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "series_length": int(X_full.shape[1]),
        "original_labels": classes.tolist(),
    }


def _describe(name, X, y_raw, y):
    """Print and return the statistics for one split."""
    per_series_mean = X.mean(axis=1)
    per_series_std = X.std(axis=1)
    labels, counts = np.unique(y_raw, return_counts=True)
    stats = {
        "rows": int(X.shape[0]),
        "series_length": int(X.shape[1]),
        "class_counts_original_labels": {int(l): int(c) for l, c in zip(labels, counts)},
        "class_counts_mapped": {int(l): int(c) for l, c in zip(*np.unique(y, return_counts=True))},
        "global_mean": float(X.mean()),
        "global_std": float(X.std()),
        "global_min": float(X.min()),
        "global_max": float(X.max()),
        "per_series_mean_min": float(per_series_mean.min()),
        "per_series_mean_max": float(per_series_mean.max()),
        "per_series_std_min": float(per_series_std.min()),
        "per_series_std_max": float(per_series_std.max()),
    }
    print(f"\n=== {name} ===")
    for k, v in stats.items():
        print(f"  {k:32s} {v}")
    return stats


if __name__ == "__main__":
    X_full, y_full_raw = load_tsv(config.TRAIN_TSV)
    X_test, y_test_raw = load_tsv(config.TEST_TSV)
    classes = np.unique(y_full_raw)
    y_full = map_labels(y_full_raw, classes)
    y_test = map_labels(y_test_raw, classes)

    print("label mapping (original -> ours):", {int(c): i for i, c in enumerate(classes)})
    stats = {
        "train": _describe("TRAIN (official file)", X_full, y_full_raw, y_full),
        "test": _describe("TEST (official file)", X_test, y_test_raw, y_test),
    }

    # One example row: label and the first 10 values, so the scale of the data is visible.
    print("\nexample: TRAIN row 0, original label", int(y_full_raw[0]), "-> mapped", int(y_full[0]))
    print("  first 10 values:", np.array2string(X_full[0, :10], precision=4, separator=", "))
    print("  row mean / std :", f"{X_full[0].mean():.6f} / {X_full[0].std():.6f}")

    # Stratified split of TRAIN into train/val with the fixed seed.
    X_tr, y_tr, X_val, y_val = stratified_split(X_full, y_full, config.VAL_FRACTION, config.SEED)
    split = {
        "seed": config.SEED,
        "val_fraction": config.VAL_FRACTION,
        "train_rows": int(len(y_tr)),
        "train_class_counts": {int(l): int(c) for l, c in zip(*np.unique(y_tr, return_counts=True))},
        "val_rows": int(len(y_val)),
        "val_class_counts": {int(l): int(c) for l, c in zip(*np.unique(y_val, return_counts=True))},
    }
    print("\n=== stratified split of TRAIN ===")
    for k, v in split.items():
        print(f"  {k:32s} {v}")
    stats["split"] = split

    # Shape the model will see.
    print("\nConv1d input shape (batch, channels, length):", to_conv1d(X_tr).shape)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    out = config.RESULTS_DIR / "data_stats.json"
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    print("\nwrote", out)