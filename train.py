"""Download a small Quick, Draw! subset and train ResNet-18, VGG-11, MobileNetV3-Small.

Usage:
    python train.py
Optional environment variables:
    QD_CLASSES="cat,apple,airplane,fish,house,tree,star,bicycle"
    QD_SAMPLES_PER_CLASS=2500
    QD_EPOCHS=5
    QD_BATCH_SIZE=64
"""
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import models, transforms
from PIL import Image

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
WEIGHTS_DIR = ROOT / "weights"
DATA_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR.mkdir(exist_ok=True)

CLASSES = [x.strip().replace("_", " ") for x in os.getenv(
    "QD_CLASSES", "apple,cat,airplane,fish,house,tree,star,bicycle"
).split(",") if x.strip()]
SAMPLES = int(os.getenv("QD_SAMPLES_PER_CLASS", "2500"))
EPOCHS = int(os.getenv("QD_EPOCHS", "5"))
BATCH_SIZE = int(os.getenv("QD_BATCH_SIZE", "64"))
SIZE = 96
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
(ROOT / "classes.txt").write_text("\n".join(CLASSES) + "\n")

def fetch_class(name):
    filename = name.replace(" ", "_") + ".npy"
    path = DATA_DIR / filename
    if not path.exists():
        url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{filename}"
        print(f"Downloading {name}: {url}")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download class '{name}'. Check the category spelling and internet access: {url}"
            ) from exc
    return np.load(path, mmap_mode="r")

def load_data():
    xs, ys = [], []
    rng = np.random.default_rng(42)
    for label, name in enumerate(CLASSES):
        arr = fetch_class(name)
        take = min(SAMPLES, len(arr))
        indices = rng.choice(len(arr), size=take, replace=False)
        # QuickDraw numpy bitmaps are 28x28 grayscale; resize and convert to 3 channels.
        for i in indices:
            image = Image.fromarray(arr[i].reshape(28, 28).astype(np.uint8), mode="L")
            image = image.resize((SIZE, SIZE), Image.Resampling.BILINEAR).convert("RGB")
            xs.append(np.asarray(image, dtype=np.uint8))
            ys.append(label)
        print(f"Loaded {take:,} examples for {name}")
    x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).float() / 255.0
    # Standardize to [-1, 1], matching the app preprocessing.
    x = (x - 0.5) / 0.5
    y = torch.tensor(ys, dtype=torch.long)
    return TensorDataset(x, y)

def create_model(arch, num_classes):
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "vgg11":
        model = models.vgg11(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model

def train_one(arch, filename, dataset):
    model = create_model(arch, len(CLASSES)).to(DEVICE)
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)
        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += len(yb)
        acc = correct / max(total, 1)
        print(f"{arch} | epoch {epoch+1}/{EPOCHS} | loss {running_loss/max(n_train,1):.4f} | val acc {acc:.3%}")
        if acc >= best_acc:
            best_acc = acc
            torch.save({"model_state": model.state_dict(), "classes": CLASSES, "val_accuracy": best_acc}, WEIGHTS_DIR / filename)
    print(f"Saved {filename}; best validation accuracy {best_acc:.3%}")

if __name__ == "__main__":
    print(f"Device: {DEVICE}; classes: {CLASSES}; samples/class: up to {SAMPLES}")
    data = load_data()
    # Sequential training avoids loading all three large networks into memory simultaneously.
    train_one("resnet18", "resnet18.pt", data)
    train_one("vgg11", "vgg11.pt", data)
    train_one("mobilenet_v3_small", "mobilenet_v3_small.pt", data)
    print("Training complete. Start the game with: streamlit run app.py")
