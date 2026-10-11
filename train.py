"""Download a Quick, Draw! subset and train ResNet-18, VGG-16, and MobileNetV3-Small.

Usage:
    python train.py

Optional environment variables:
    QD_CLASSES="apple,bicycle,car,cat,fish,house,star,sun,tree,umbrella"
    QD_SAMPLES_PER_CLASS=3000
    QD_EPOCHS=5
    QD_BATCH_SIZE=32

The class order is shared with classes.txt and the app's prediction labels.
The script trains from scratch and saves checkpoints under weights/.
"""

import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import models


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
WEIGHTS_DIR = ROOT / "weights"
CLASSES_FILE = ROOT / "classes.txt"

DATA_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CLASSES = (
    "apple,bicycle,car,cat,fish,house,star,sun,tree,umbrella"
)
CLASSES = [
    name.strip().replace("_", " ")
    for name in os.getenv("QD_CLASSES", DEFAULT_CLASSES).split(",")
    if name.strip()
]
SAMPLES = int(os.getenv("QD_SAMPLES_PER_CLASS", "3000"))
EPOCHS = int(os.getenv("QD_EPOCHS", "5"))
BATCH_SIZE = int(os.getenv("QD_BATCH_SIZE", "32"))
IMAGE_SIZE = 96
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not CLASSES:
    raise ValueError("No classes configured. Set QD_CLASSES to a comma-separated list.")
if len(set(CLASSES)) != len(CLASSES):
    raise ValueError("QD_CLASSES contains duplicate class names.")
if SAMPLES < 1 or EPOCHS < 1 or BATCH_SIZE < 1:
    raise ValueError("Samples, epochs, and batch size must all be positive integers.")

# Save the exact class order used by all three models.
CLASSES_FILE.write_text("\n".join(CLASSES) + "\n", encoding="utf-8")


def fetch_class(name: str) -> np.ndarray:
    """Download and memory-map the Quick, Draw! bitmap file for one class."""
    filename = name.replace(" ", "_") + ".npy"
    path = DATA_DIR / filename

    if not path.exists():
        url = (
            "https://storage.googleapis.com/quickdraw_dataset/full/"
            f"numpy_bitmap/{filename}"
        )
        print(f"Downloading {name}: {url}")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as exc:
            # Remove a partial download so the next run can retry cleanly.
            if path.exists():
                path.unlink()
            raise RuntimeError(
                f"Could not download class '{name}'. Check the category spelling "
                f"and internet access. URL: {url}"
            ) from exc

    return np.load(path, mmap_mode="r")


def load_data() -> TensorDataset:
    """Sample up to SAMPLES drawings per class and build a labelled dataset."""
    images = []
    labels = []
    rng = np.random.default_rng(42)

    for label, class_name in enumerate(CLASSES):
        drawings = fetch_class(class_name)
        count = min(SAMPLES, len(drawings))
        indices = rng.choice(len(drawings), size=count, replace=False)

        for index in indices:
            bitmap = drawings[index].reshape(28, 28).astype(np.uint8)
            image = Image.fromarray(bitmap)
            image = image.resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
            ).convert("RGB")
            images.append(np.asarray(image, dtype=np.uint8))
            labels.append(label)

        print(f"Loaded {count:,} examples for {class_name}")

    x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255.0
    # Match app.py preprocessing: normalize RGB channels to [-1, 1].
    x = (x - 0.5) / 0.5
    y = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(x, y)


def create_model(architecture: str, num_classes: int) -> nn.Module:
    """Create a model whose output layer matches the configured class count."""
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "vgg16":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif architecture == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model


def train_one(
    architecture: str,
    checkpoint_filename: str,
    dataset: TensorDataset,
) -> None:
    """Train one architecture and save its best validation checkpoint."""
    model = create_model(architecture, len(CLASSES)).to(DEVICE)

    validation_size = max(1, int(0.1 * len(dataset)))
    training_size = len(dataset) - validation_size
    train_dataset, validation_dataset = random_split(
        dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=1e-4
    )
    loss_function = nn.CrossEntropyLoss()
    best_accuracy = 0.0
    checkpoint_path = WEIGHTS_DIR / checkpoint_filename

    for epoch in range(EPOCHS):
        model.train()
        total_training_loss = 0.0

        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_images), batch_labels)
            loss.backward()
            optimizer.step()
            total_training_loss += loss.item() * len(batch_images)

        model.eval()
        correct = 0
        total = 0

        with torch.inference_mode():
            for batch_images, batch_labels in validation_loader:
                batch_images = batch_images.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                predictions = model(batch_images).argmax(dim=1)
                correct += (predictions == batch_labels).sum().item()
                total += len(batch_labels)

        accuracy = correct / max(total, 1)
        average_loss = total_training_loss / max(training_size, 1)
        print(
            f"{architecture} | epoch {epoch + 1}/{EPOCHS} | "
            f"loss {average_loss:.4f} | validation accuracy {accuracy:.3%}"
        )

        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": CLASSES,
                    "architecture": architecture,
                    "val_accuracy": best_accuracy,
                },
                checkpoint_path,
            )

    print(
        f"Saved {checkpoint_path.name}; "
        f"best validation accuracy {best_accuracy:.3%}"
    )


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Classes ({len(CLASSES)}): {CLASSES}")
    print(f"Samples per class: up to {SAMPLES:,}")
    print(f"Maximum dataset size: {len(CLASSES) * SAMPLES:,} drawings")
    print(f"Epochs: {EPOCHS}; batch size: {BATCH_SIZE}")
    print(f"Class list written to: {CLASSES_FILE}")

    dataset = load_data()

    # Train sequentially so only one large model is active at a time.
    train_one("resnet18", "resnet18.pt", dataset)
    train_one("vgg16", "vgg16.pt", dataset)
    train_one("mobilenet_v3_small", "mobilenet_v3_small.pt", dataset)

    print("Training complete. Start the game with: streamlit run app.py")
