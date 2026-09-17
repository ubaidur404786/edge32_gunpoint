"""Every project setting in one place. All scripts import this file.

Run scripts from the project root as modules, e.g.  python -m src.dataset
"""

from pathlib import Path  # build OS-independent paths relative to this file, not to the current directory

# ---- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "GunPoint"
TRAIN_TSV = DATA_DIR / "GunPoint_TRAIN.tsv"
TEST_TSV = DATA_DIR / "GunPoint_TEST.tsv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"
FIRMWARE_DIR = ROOT / "firmware" / "gunpoint_tflm"

# ---- Data ------------------------------------------------------------------
SEED = 42            # one seed for the split, weight init and shuffling
VAL_FRACTION = 0.2   # fraction of the official TRAIN split held out for validation
CLASS_NAMES = ["Gun", "Point"]  # UCR GunPoint: label 1 = Gun, label 2 = Point -> we map to 0, 1

# ---- Model (Step 2) --------------------------------------------------------
CONV1_CHANNELS = 8
CONV2_CHANNELS = 16
KERNEL_SIZE = 3
POOL_SIZE = 2
NUM_CLASSES = 2

# ---- Training (Step 3) -----------------------------------------------------
EPOCHS = 200         # upper bound; early stopping usually ends earlier
BATCH_SIZE = 8       # small dataset -> small batches
LEARNING_RATE = 1e-3
PATIENCE = 20        # stop when val loss has not improved for this many epochs

# ---- Board / serial protocol (Steps 7-10) ----------------------------------
SERIAL_PORT = "COM11"
BAUD = 115200
N_SAMPLES_FIRST_RUN = 5   # Step 8 exports only this many test samples first