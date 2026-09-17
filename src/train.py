"""Train on the train split, select on the val split, evaluate TEST exactly once.

Run from the project root:  python -m src.train
Writes models/model_fp32.pt and results/train_metrics.json.
"""

import copy            # deep-copy the best state_dict so later epochs cannot overwrite it
import json            # write results/train_metrics.json
import numpy as np     # seeding and accuracy arithmetic
import torch           # tensors, optimizer, loss, checkpoint saving
import torch.nn as nn  # CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader  # mini-batching with a seeded shuffle

import config                                   # seed, epochs, batch size, lr, patience, paths
from src.dataset import load_all, to_conv1d     # data splits and the (N,1,L) shape helper
from src.model import GunPointCNN, count_parameters  # the model and its parameter count


def to_tensors(X, y):
    """numpy (N,L) float32 + (N,) int64 -> torch (N,1,L) float32 + (N,) int64."""
    return torch.from_numpy(to_conv1d(X)), torch.from_numpy(y)


@torch.no_grad()
def evaluate(model, X, y, loss_fn):
    """Mean loss and accuracy on a whole split in one forward pass (the splits are tiny)."""
    model.eval()
    logits = model(X)
    loss = loss_fn(logits, y).item()
    acc = (logits.argmax(dim=1) == y).float().mean().item()
    return loss, acc


def main():
    # Fixed seeds: same split (dataset.py), same init, same shuffle order every run.
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    d = load_all()
    L = d["series_length"]
    X_tr, y_tr = to_tensors(d["X_train"], d["y_train"])
    X_val, y_val = to_tensors(d["X_val"], d["y_val"])
    X_te, y_te = to_tensors(d["X_test"], d["y_test"])

    model = GunPointCNN(L)
    _, n_params = count_parameters(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Seeded shuffling so the batch order is reproducible.
    gen = torch.Generator().manual_seed(config.SEED)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=config.BATCH_SIZE, shuffle=True, generator=gen)

    print(f"train {len(y_tr)} rows, val {len(y_val)} rows, test {len(y_te)} rows (test is NOT used until the end)")
    print(f"params {n_params}, epochs <= {config.EPOCHS}, batch {config.BATCH_SIZE}, lr {config.LEARNING_RATE}, patience {config.PATIENCE}\n")
    print(f"{'epoch':>5s} {'train_loss':>10s} {'val_loss':>9s} {'val_acc':>8s}")

    history = []
    best = {"val_loss": float("inf"), "epoch": -1, "state": None}
    epochs_without_improvement = 0

    for epoch in range(1, config.EPOCHS + 1):
        # ---- one pass over the training rows --------------------------------
        model.train()
        running, seen = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(yb)   # weight by batch size (last batch is smaller)
            seen += len(yb)
        train_loss = running / seen

        # ---- validation: selection signal only -------------------------------
        val_loss, val_acc = evaluate(model, X_val, y_val, loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})
        marker = ""
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict()), "val_acc": val_acc}
            epochs_without_improvement = 0
            marker = " *"   # new best
        else:
            epochs_without_improvement += 1
        print(f"{epoch:5d} {train_loss:10.4f} {val_loss:9.4f} {val_acc:8.3f}{marker}")

        if epochs_without_improvement >= config.PATIENCE:
            print(f"\nearly stop: no val-loss improvement for {config.PATIENCE} epochs")
            break

    # ---- restore the best-val weights ----------------------------------------
    model.load_state_dict(best["state"])
    print(f"best epoch {best['epoch']}: val_loss {best['val_loss']:.4f}, val_acc {best['val_acc']:.3f}")

    # ---- the ONE evaluation on the official TEST split -----------------------
    test_loss, test_acc = evaluate(model, X_te, y_te, loss_fn)
    print(f"\nFINAL TEST (150 rows, evaluated once): loss {test_loss:.4f}, accuracy {test_acc:.4f}")

    # ---- save checkpoint and metrics ------------------------------------------
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    ckpt_path = config.MODELS_DIR / "model_fp32.pt"
    torch.save({"state_dict": model.state_dict(), "series_length": L, "num_classes": config.NUM_CLASSES}, ckpt_path)

    metrics = {
        "seed": config.SEED,
        "params": n_params,
        "epochs_run": len(history),
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "best_val_acc": best["val_acc"],
        "test_loss": test_loss,
        "test_acc": test_acc,
        "config": {"epochs": config.EPOCHS, "batch_size": config.BATCH_SIZE,
                   "lr": config.LEARNING_RATE, "patience": config.PATIENCE},
        "history": history,
    }
    with open(config.RESULTS_DIR / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("wrote", ckpt_path, "and", config.RESULTS_DIR / "train_metrics.json")


if __name__ == "__main__":
    main()