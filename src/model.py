"""The PyTorch model for GunPoint. Nothing else lives here.

Conv1d(1->8,k3) -> ReLU -> MaxPool1d(2) -> Conv1d(8->16,k3) -> ReLU -> MaxPool1d(2)
-> Flatten -> Linear(->2)

Run from the project root to print shapes and parameter counts:  python -m src.model
"""

import torch           # tensors and the dummy forward pass in __main__
import torch.nn as nn  # the layer classes and nn.Module base class

import config          # channel counts, kernel/pool sizes, number of classes


def conv_out_len(length, kernel_size):
    """Length after a Conv1d with stride 1 and no padding ("valid")."""
    return length - kernel_size + 1


def pool_out_len(length, pool_size):
    """Length after MaxPool1d(pool_size): floor division, leftovers are dropped."""
    return length // pool_size


def flat_features(series_length):
    """Number of inputs to the Linear layer, derived step by step from the series length."""
    n = conv_out_len(series_length, config.KERNEL_SIZE)  # conv1
    n = pool_out_len(n, config.POOL_SIZE)                # pool1
    n = conv_out_len(n, config.KERNEL_SIZE)              # conv2
    n = pool_out_len(n, config.POOL_SIZE)                # pool2
    return config.CONV2_CHANNELS * n


class GunPointCNN(nn.Module):
    """Two small conv blocks and one linear layer. Output = 2 raw logits (no Softmax)."""

    def __init__(self, series_length):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, config.CONV1_CHANNELS, config.KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool1d(config.POOL_SIZE),
            nn.Conv1d(config.CONV1_CHANNELS, config.CONV2_CHANNELS, config.KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool1d(config.POOL_SIZE),
        )
        self.flatten = nn.Flatten()   # (batch, 16, L') -> (batch, 16*L'); channel-major order
        self.classifier = nn.Linear(flat_features(series_length), config.NUM_CLASSES)

    def forward(self, x):
        # x: (batch, 1, series_length)
        return self.classifier(self.flatten(self.features(x)))


def count_parameters(model):
    """Trainable parameters, per named tensor and in total."""
    rows = [(name, tuple(p.shape), p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    return rows, sum(r[2] for r in rows)


if __name__ == "__main__":
    from src.dataset import load_all   # only here: the series length must come from the data, not be typed in

    L = load_all()["series_length"]
    model = GunPointCNN(L).eval()

    # Print the output shape of every leaf layer with a forward hook.
    def make_hook(name):
        def hook(module, inputs, output):
            print(f"  {name:28s} -> {tuple(output.shape)}")
        return hook

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:   # leaf layers only
            module.register_forward_hook(make_hook(name))

    print(f"series length from data: {L}")
    print("shape after every layer (batch of 1):")
    print(f"  {'input':28s} -> {(1, 1, L)}")
    with torch.no_grad():
        out = model(torch.zeros(1, 1, L))

    # The arithmetic behind the Linear layer's input size, step by step.
    k, p, c2 = config.KERNEL_SIZE, config.POOL_SIZE, config.CONV2_CHANNELS
    n1 = conv_out_len(L, k); n2 = pool_out_len(n1, p); n3 = conv_out_len(n2, k); n4 = pool_out_len(n3, p)
    print("\nLinear input size arithmetic:")
    print(f"  conv1: {L} - {k} + 1 = {n1}")
    print(f"  pool1: {n1} // {p}     = {n2}")
    print(f"  conv2: {n2} - {k} + 1  = {n3}")
    print(f"  pool2: {n3} // {p}      = {n4}")
    print(f"  flatten: {c2} channels x {n4} = {c2 * n4}  -> Linear({c2 * n4}, {config.NUM_CLASSES})")

    rows, total = count_parameters(model)
    print("\ntrainable parameters:")
    for name, shape, n in rows:
        print(f"  {name:28s} {str(shape):16s} {n}")
    print(f"  {'TOTAL':28s} {'':16s} {total}")
    print("\noutput logits shape:", tuple(out.shape), "(raw logits, no Softmax)")