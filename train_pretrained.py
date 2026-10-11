"""Fine-tune ImageNet-pretrained models on Quick, Draw! data.

Run from the Drawasaurus project folder:
    python train_pretrained.py

Optional environment variables:
    QD_CLASSES="apple,bicycle,car,cat,fish,house,star,sun,tree,umbrella"
    QD_SAMPLES_PER_CLASS=3000
    QD_EPOCHS=5
    QD_BATCH_SIZE=16
    QD_ARCHS="resnet18,mobilenet_v3_small,vgg16"

This script saves separate *_pretrained.pt checkpoints, so it does not
overwrite checkpoints created by train.py. Check app.py's model-loading code
and use ImageNet normalization when loading these pretrained checkpoints.
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

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
WEIGHTS_DIR = ROOT / "weights"
DATA_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CLASSES = (
    "apple,bicycle,car,cat,fish,house,star,sun,tree,umbrella"
)
CLASSES = [
    item.strip().replace("_", " ")
    for item in os.getenv("QD_CLASSES", DEFAULT_CLASSES).split(",")
    if item.strip()
]
SAMPLES = int(os.getenv("QD_SAMPLES_PER_CLASS", "3000"))
EPOCHS = int(os.getenv("QD_EPOCHS", "5"))
BATCH_SIZE = int(os.getenv("QD_BATCH_SIZE", "16"))
ARCHS = [
    item.strip().lower()
    for item in os.getenv(
        "QD_ARCHS", "resnet18,mobilenet_v3_small,vgg16"
    ).split(",")
    if item.strip()
]
SIZE = 96

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SUPPORTED_ARCHS = {"resnet18", "vgg16", "mobilenet_v3_small"}

(ROOT / "classes.txt").write_text("\n".join(CLASSES) + "\n")


def fetch_class(class_name):
    """Download a QuickDraw class file if needed and recover from partial files."""
    filename = class_name.replace(" ", "_") + ".npy"
    path = DATA_DIR / filename
    url = (
        "https://storage.googleapis.com/quickdraw_dataset/full/"
        f"numpy_bitmap/{filename}"
    )

    if path.exists():
        try:
            # Validate the .npy header and array shape before using the file.
            arr = np.load(path, mmap_mode="r")
            if arr.ndim != 2 or arr.shape[1] != 784:
                raise ValueError(f"Unexpected array shape: {arr.shape}")
            return arr
        except Exception as exc:
            print(f"Removing invalid/incomplete file {path}: {exc}")
            path.unlink(missing_ok=True)

    print(f"Downloading {class_name}: {url}")
    temporary_path = path.with_suffix(".npy.download")
    try:
        urllib.request.urlretrieve(url, temporary_path)
        arr = np.load(temporary_path, mmap_mode="r")
        if arr.ndim != 2 or arr.shape[1] != 784:
            raise ValueError(f"Downloaded array has unexpected shape: {arr.shape}")
        temporary_path.replace(path)
        return np.load(path, mmap_mode="r")
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download/validate '{class_name}'. Check the class name "
            f"and internet connection. URL: {url}"
        ) from exc


def load_data():
    """Load a balanced subset and apply ImageNet normalization."""
    images = []
    labels = []
    rng = np.random.default_rng(42)

    for label, class_name in enumerate(CLASSES):
        drawings = fetch_class(class_name)
        count = min(SAMPLES, len(drawings))
        indices = rng.choice(len(drawings), size=count, replace=False)

        for index in indices:
            image = Image.fromarray(
                drawings[index].reshape(28, 28).astype(np.uint8), mode="L"
            )
            image = image.resize((SIZE, SIZE), Image.Resampling.BILINEAR).convert("RGB")
            images.append(np.asarray(image, dtype=np.uint8))
            labels.append(label)

        print(f"Loaded {count:,} examples for {class_name}")

    # Convert to N,C,H,W float tensors in [0,1], then ImageNet-normalize.
    x = torch.from_numpy(np.stack(images).copy()).permute(0, 3, 1, 2).float()
    x.div_(255.0)
    x.sub_(MEAN).div_(STD)
    y = torch.tensor(labels, dtype=torch.long)

    print(f"Total loaded: {len(y):,} drawings")
    return TensorDataset(x, y)


def create_model(architecture, num_classes):
    """Load pretrained ImageNet weights and replace the output head."""
    if architecture == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif architecture == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        # With 96x96 inputs, VGG16's final feature map is 3x3.
        # Identity avoids the unsupported non-divisible adaptive pool on Apple MPS.
        # The first classifier layer must be resized to accept 512*3*3 features.
        model.avgpool = nn.Identity()
        model.classifier[0] = nn.Linear(512 * 3 * 3, 4096)
        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features, num_classes
        )

    elif architecture == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features, num_classes
        )

    else:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. "
            f"Choose from: {', '.join(sorted(SUPPORTED_ARCHS))}"
        )

    return model


def train_one(architecture, dataset):
    """Fine-tune one model and save its best validation checkpoint."""
    print(
        f"\n{'=' * 60}\n"
        f"Training ImageNet-pretrained {architecture} on {DEVICE}\n"
        f"{'=' * 60}"
    )
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

    if architecture == "resnet18":
        head_parameters = list(model.fc.parameters())
    else:
        head_parameters = list(model.classifier.parameters())

    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [
        parameter for parameter in model.parameters()
        if id(parameter) not in head_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": 1e-5},
            {"params": head_parameters, "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )
    loss_function = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    checkpoint_path = WEIGHTS_DIR / f"{architecture}_pretrained.pt"

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_images), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_images)

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
        mean_loss = total_loss / max(training_size, 1)
        print(
            f"{architecture} | epoch {epoch + 1}/{EPOCHS} "
            f"| train loss {mean_loss:.4f} "
            f"| validation accuracy {accuracy:.3%}"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": CLASSES,
                    "architecture": architecture,
                    "val_accuracy": best_accuracy,
                    "pretrained": True,
                    "normalization": "imagenet",
                    "image_size": SIZE,
                },
                checkpoint_path,
            )
            print(f"  Saved best checkpoint: {checkpoint_path}")

    print(
        f"Finished {architecture}; best validation accuracy: "
        f"{best_accuracy:.3%}"
    )


if __name__ == "__main__":
    if not CLASSES:
        raise ValueError("QD_CLASSES must contain at least one class.")
    if len(set(CLASSES)) != len(CLASSES):
        raise ValueError("QD_CLASSES contains duplicate class names.")
    if SAMPLES < 1 or EPOCHS < 1 or BATCH_SIZE < 1:
        raise ValueError("Samples, epochs, and batch size must all be positive.")
    invalid_archs = sorted(set(ARCHS) - SUPPORTED_ARCHS)
    if invalid_archs:
        raise ValueError(f"Unsupported architectures: {invalid_archs}")

    print(f"Device: {DEVICE}")
    print(f"Classes ({len(CLASSES)}): {CLASSES}")
    print(f"Samples per class: up to {SAMPLES:,}")
    print(f"Maximum dataset size: {len(CLASSES) * SAMPLES:,} drawings")
    print(f"Epochs: {EPOCHS}; batch size: {BATCH_SIZE}")
    print(f"Architectures to train: {ARCHS}")
    print(f"Class list written to: {ROOT / 'classes.txt'}")

    dataset = load_data()
    for architecture in ARCHS:
        train_one(architecture, dataset)

    print("\nPretrained-model training complete.")
    print("Use ImageNet mean/std normalization when running inference.")
