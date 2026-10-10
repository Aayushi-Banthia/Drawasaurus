# AI Draw Battle — Streamlit

A Drawasaurus-inspired sketch game with three independently trained judges:
**ResNet-18, VGG-11, and MobileNetV3-Small**.

## 1. Create an environment

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

If PyTorch installation fails, install the appropriate build from https://pytorch.org/get-started/locally/ for your operating system and hardware, then install the remaining requirements.

## 2. Train the judges

```bash
python train.py
```

The default subset has eight categories: `apple`, `cat`, `airplane`, `fish`, `house`, `tree`, `star`, and `bicycle`. The training script downloads each category's `.npy` bitmap file from the official Quick, Draw! Google Cloud Storage location, samples up to 2,500 drawings per category, and trains all three networks.

Training three models can take a while. A CUDA-capable GPU is recommended. For a quick smoke test, use fewer samples and epochs:

**Windows PowerShell**
```powershell
$env:QD_SAMPLES_PER_CLASS="200"
$env:QD_EPOCHS="1"
python train.py
```

**macOS/Linux**
```bash
QD_SAMPLES_PER_CLASS=200 QD_EPOCHS=1 python train.py
```

You can choose categories with `QD_CLASSES`, for example:
```bash
QD_CLASSES="cat,dog,apple,fish" QD_SAMPLES_PER_CLASS=1500 QD_EPOCHS=5 python train.py
```

The selected class list is written to `classes.txt`; trained weights are saved under `weights/`.

## 3. Launch the game

```bash
streamlit run app.py
```

## Scoring

Each model's top-1 prediction is compared with the prompt. Every correct judge awards 10 points, for up to 30 points per round. Confidence scores are displayed but do not affect points.

## Important notes

- The app requires trained weights. It does not silently substitute pretrained ImageNet classifiers, because those models are not trained on Quick, Draw! categories.
- The drawing is converted to a dark-background, bright-stroke image before inference. If predictions look consistently poor, compare a saved canvas preprocessing preview against sample images from the downloaded `.npy` files and adjust polarity/resizing as needed.
- For a fair comparison, all three models use the same class set and sampled dataset.
- The initial implementation uses a standard Streamlit canvas. It supports drawing, clearing, and color/brush controls from the component toolbar.
- Dataset credit: Google Creative Lab and Quick, Draw! contributors. The dataset is licensed CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
