"""
operator_classifier.py
----------------------
Extends the base digit-recognition CNN (trained on MNIST) into a 16-class model:
    Classes 0-9  : digits '0' through '9'
    Class  10    : '+'
    Class  11    : '-'
    Class  12    : '*'
    Class  13    : '÷'  (division)
    Class  14    : '('
    Class  15    : ')'

Workflow
--------
1. generate_operator_dataset()
       Synthetically draws each operator symbol onto 28×28 images with random
       augmentations (position jitter, thickness variation, rotation, noise)
       so the model generalises to messy handwriting.

2. OperatorCNN  (nn.Module)
       A lightweight CNN that accepts 28×28 single-channel input —
       identical input spec as the MNIST digit models your team already trained.

3. train_combined_model()
       Loads MNIST digits (via torchvision) + the synthetic operator images,
       trains the 16-class CNN, and saves the weights to `model_combined.pth`.

4. predict_character(image_28x28) -> (label_char, confidence)
       Loads the saved model and returns the predicted character + its
       softmax confidence for a single 28×28 grayscale image.

Usage
-----
    # One-time training (takes ~2-3 min on CPU):
        python operator_classifier.py

    # From another script:
        from operator_classifier import predict_character
        char, conf = predict_character(some_28x28_numpy_array)
"""

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# ---------------------------------------------------------------
# GLOBAL CONFIG
# ---------------------------------------------------------------
MODEL_PATH        = 'model_combined.pth'   # saved weights
SYNTH_DATA_DIR    = 'synthetic_operators'  # generated operator images
SAMPLES_PER_CLASS = 5000                   # synthetic samples per operator
NUM_CLASSES       = 16                     # 0-9 digits + 6 operators
EPOCHS            = 15
BATCH_SIZE        = 128
LEARNING_RATE     = 0.001
DEVICE            = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Maps class index -> display character
INDEX_TO_CHAR = {
    0:'0', 1:'1', 2:'2', 3:'3', 4:'4',
    5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
    10:'+', 11:'-', 12:'*', 13:'÷', 14:'(', 15:')'
}

# Reverse map: character -> class index
CHAR_TO_INDEX = {v: k for k, v in INDEX_TO_CHAR.items()}


# ---------------------------------------------------------------
# PART 1 — SYNTHETIC OPERATOR DATASET GENERATION
# ---------------------------------------------------------------
def _draw_plus(img, cx, cy, size, thickness):
    """Draw a '+' sign centered at (cx, cy)."""
    arm = size // 2
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), 255, thickness)  # horizontal
    cv2.line(img, (cx, cy - arm), (cx, cy + arm), 255, thickness)  # vertical

def _draw_minus(img, cx, cy, size, thickness):
    """Draw a '-' sign (horizontal bar)."""
    arm = size // 2
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), 255, thickness)

def _draw_multiply(img, cx, cy, size, thickness):
    """Draw a '×' sign (two diagonal lines)."""
    arm = size // 2
    cv2.line(img, (cx - arm, cy - arm), (cx + arm, cy + arm), 255, thickness)
    cv2.line(img, (cx - arm, cy + arm), (cx + arm, cy - arm), 255, thickness)

def _draw_divide(img, cx, cy, size, thickness):
    """Draw a '÷' sign (horizontal bar + two dots)."""
    arm = size // 2
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), 255, thickness)  # bar
    dot_r = max(thickness - 1, 2)
    cv2.circle(img, (cx, cy - size // 3), dot_r, 255, -1)  # top dot
    cv2.circle(img, (cx, cy + size // 3), dot_r, 255, -1)  # bottom dot

def _draw_left_paren(img, cx, cy, size, thickness):
    """Draw '(' using an ellipse arc on the left side."""
    w = size // 3
    h = size // 2
    # cv2.ellipse: center, axes, angle, start_angle, end_angle
    cv2.ellipse(img, (cx, cy), (w, h), 0, 100, 260, 255, thickness)

def _draw_right_paren(img, cx, cy, size, thickness):
    """Draw ')' — mirror of left paren."""
    w = size // 3
    h = size // 2
    cv2.ellipse(img, (cx, cy), (w, h), 0, -80, 80, 255, thickness)


# Dispatch table: operator symbol -> drawing function
DRAW_FN = {
    '+': _draw_plus,
    '-': _draw_minus,
    '*': _draw_multiply,
    '÷': _draw_divide,
    '(': _draw_left_paren,
    ')': _draw_right_paren,
}


def generate_operator_dataset():
    """
    Creates SYNTH_DATA_DIR/<symbol>/ with SAMPLES_PER_CLASS augmented 28×28 PNGs
    for each of the 6 operator symbols.
    """
    os.makedirs(SYNTH_DATA_DIR, exist_ok=True)

    for symbol, draw_fn in DRAW_FN.items():
        # Use safe folder names (avoid filesystem issues with special chars)
        safe_name = {'+':'plus', '-':'minus', '*':'mul',
                     '÷':'div', '(':'lparen', ')':'rparen'}[symbol]
        folder = os.path.join(SYNTH_DATA_DIR, safe_name)
        os.makedirs(folder, exist_ok=True)

        print(f"[GEN] Generating {SAMPLES_PER_CLASS} samples for '{symbol}' → {folder}")

        for i in range(SAMPLES_PER_CLASS):
            img = np.zeros((28, 28), dtype=np.uint8)

            # --- random augmentation parameters ---
            cx = 14 + np.random.randint(-3, 4)        # center jitter
            cy = 14 + np.random.randint(-3, 4)
            size = np.random.randint(10, 20)           # symbol size
            thickness = np.random.randint(1, 4)        # stroke thickness

            draw_fn(img, cx, cy, size, thickness)

            # Random rotation (-15 to +15 degrees)
            angle = np.random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
            img = cv2.warpAffine(img, M, (28, 28))

            # Add slight Gaussian noise
            noise = np.random.normal(0, 12, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Random global brightness shift
            shift = np.random.randint(-20, 20)
            img = np.clip(img.astype(np.int16) + shift, 0, 255).astype(np.uint8)

            cv2.imwrite(os.path.join(folder, f'{i:05d}.png'), img)

    print("[GEN] Synthetic operator dataset generation complete.")


# ---------------------------------------------------------------
# PART 2 — PYTORCH DATASET WRAPPERS
# ---------------------------------------------------------------
class SyntheticOperatorDataset(Dataset):
    """
    Loads the synthetically generated operator images from disk.
    Maps each operator folder to its correct class index (10-15).
    """
    # Folder name -> class index mapping
    FOLDER_TO_CLASS = {
        'plus': 10, 'minus': 11, 'mul': 12,
        'div': 13, 'lparen': 14, 'rparen': 15
    }

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []  # list of (image_path, class_index)

        for folder_name, class_idx in self.FOLDER_TO_CLASS.items():
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                print(f"[WARN] Folder not found: {folder_path}, skipping.")
                continue
            for fname in sorted(os.listdir(folder_path)):
                if fname.endswith('.png'):
                    self.samples.append((os.path.join(folder_path, fname), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # Load as grayscale, normalize to [0,1]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0  # shape: (1,28,28)

        if self.transform:
            img = self.transform(img)

        return img, label


class MNISTasFloat(Dataset):
    """
    Thin wrapper around torchvision MNIST that ensures output dtype is float32
    and labels are plain ints (compatible with our combined DataLoader).
    """
    def __init__(self, mnist_dataset):
        self.dataset = mnist_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # torchvision already returns (1,28,28) float tensor via default transform
        return img.float(), int(label)


# ---------------------------------------------------------------
# PART 3 — CNN MODEL ARCHITECTURE
# ---------------------------------------------------------------
class OperatorCNN(nn.Module):
    """
    Simple CNN — same input spec as MNIST models (1×28×28 grayscale).
    Architecture: Conv→ReLU→Pool → Conv→ReLU→Pool → FC→ReLU→Dropout → FC(16)

    Kept intentionally small so it trains fast on CPU (~2 min).
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super(OperatorCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28×28 → 28×28
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → 14×14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14×14 → 14×14
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → 7×7

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 7×7 → 7×7
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → 3×3
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # flatten
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------
# PART 4 — TRAINING
# ---------------------------------------------------------------
def train_combined_model():
    """
    1. Generates synthetic operator data (if not already present).
    2. Loads MNIST + synthetic operators into one combined DataLoader.
    3. Trains the 16-class CNN.
    4. Saves weights to MODEL_PATH.
    """
    # --- Step 1: make sure synthetic data exists ---
    if not os.path.isdir(SYNTH_DATA_DIR) or len(os.listdir(SYNTH_DATA_DIR)) < 6:
        generate_operator_dataset()
    else:
        print("[INFO] Synthetic data already exists, skipping generation.")

    # --- Step 2: load datasets ---
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),   # PIL → (1,28,28) float [0,1]
    ])

    print("[INFO] Downloading / loading MNIST...")
    mnist_train_raw = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
    mnist_train = MNISTasFloat(mnist_train_raw)

    print("[INFO] Loading synthetic operator dataset...")
    operator_train = SyntheticOperatorDataset(SYNTH_DATA_DIR)

    # Concatenate: MNIST (60 k) + operators (30 k) = 90 k samples
    combined_dataset = ConcatDataset([mnist_train, operator_train])
    train_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # --- Step 3: model + training loop ---
    model = OperatorCNN(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"[TRAIN] Starting training for {EPOCHS} epochs on {DEVICE}...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct   = 0
        total     = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100.0 * correct / total
        print(f"  Epoch [{epoch+1:2d}/{EPOCHS}]  Loss: {total_loss/len(train_loader):.4f}  Acc: {acc:.2f}%")

    # --- Step 4: save ---
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[TRAIN] Model saved to '{MODEL_PATH}'")


# ---------------------------------------------------------------
# PART 5 — INFERENCE HELPER
# ---------------------------------------------------------------
# Module-level cache so the model is loaded only once per process
_cached_model = None

def _load_model():
    """Load and cache the trained 16-class CNN."""
    global _cached_model
    if _cached_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file '{MODEL_PATH}' not found. Run `train_combined_model()` first."
            )
        model = OperatorCNN(num_classes=NUM_CLASSES).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        _cached_model = model
    return _cached_model


def predict_character(image_28x28):
    model = _load_model()

    # Normalise to [0,1] float tensor of shape (1, 1, 28, 28)
    tensor = torch.tensor(image_28x28, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        logits     = model(tensor)
        probs      = torch.softmax(logits, dim=1)
        confidence, pred_idx = probs.max(1)

    char = INDEX_TO_CHAR[pred_idx.item()]
    return char, confidence.item()


# ---------------------------------------------------------------
# ENTRY POINT — run this file directly to train the model
# ---------------------------------------------------------------
if __name__ == "__main__":
    train_combined_model()
