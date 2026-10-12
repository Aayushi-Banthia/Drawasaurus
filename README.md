# 🎨 Drawasaurus — AI Draw Battle

**Draw it. Submit it. Challenge four AI models.**

Drawasaurus is an interactive AI-powered drawing game built with **Python, Streamlit, and PyTorch**. Players sketch randomly assigned objects on a digital canvas and challenge four deep-learning models to recognize their drawings.

The game features four AI judges — **ResNet-18, VGG-16, MobileNetV3-Small, and Pico** — each independently predicting the object represented by a submitted sketch. Players earn points for every model that correctly identifies the target.

By combining interactive gameplay with computer vision and deep-learning model comparison, Drawasaurus provides a practical demonstration of how different CNN architectures interpret human-drawn sketches.

---

## ✨ Key Features

- **Interactive Drawing Canvas:** Sketch objects freehand using a digital whiteboard.
- **Customizable Drawing Tools:** Choose pen colors, adjust brush sizes, erase strokes, and clear the canvas.
- **Randomized Drawing Prompts:** Receive a target object for each round.
- **Four Independent AI Judges:** Evaluate drawings using ResNet-18, VGG-16, MobileNetV3-Small, and Pico.
- **Prediction Comparison:** Examine each model's prediction independently.
- **Confidence Scores:** View prediction confidence when available.
- **Dynamic Scoring:** Earn points for each correctly identifying model.
- **Five-Round Gameplay:** Track your score across a complete game.
- **Pretrained Checkpoints:** Load available trained weights without requiring local training.
- **Cached Model Loading:** Avoid repeatedly loading model checkpoints during a session.
- **CPU and GPU Support:** Use CUDA-compatible GPUs when available, with CPU fallback.

## 🕹️ How to Play

1. Launch Drawasaurus and start a game.
2. View the target object assigned for the round.
3. Sketch the object using the drawing canvas and available tools.
4. Submit your drawing for evaluation.
5. Let all four AI judges analyze the sketch.
6. Compare their predictions and confidence scores, where available.
7. Earn points for every judge that correctly identifies the target.
8. Continue through all five rounds.
9. Review your final score and see how well your drawing performed against the AI models.

## 🏆 Scoring System

Each AI judge awards **10 points** when its top prediction matches the target object.

| Correct AI predictions | Points earned |
|:---:|---:|
| 0 / 4 | 0 |
| 1 / 4 | 10 |
| 2 / 4 | 20 |
| 3 / 4 | 30 |
| 4 / 4 | 40 |

**Maximum score: 200 points** across five rounds.

Scoring is based on top-prediction correctness. Confidence scores are displayed for reference and do not directly affect the points awarded.

## 🧠 Meet the AI Judges

Drawasaurus compares four model architectures to explore how different neural networks recognize hand-drawn objects.

| Model | Architecture | Key characteristics |
|---|---|---|
| ResNet-18 | Residual CNN | Uses residual connections to support feature learning in deeper networks. |
| VGG-16 | Deep CNN | Uses stacked convolutional layers to learn hierarchical visual features. |
| MobileNetV3-Small | Lightweight CNN | Designed for computational efficiency and reduced model complexity. |
| Pico | Custom CNN architecture | Provides an additional model for architectural comparison. |

Each model processes the submitted sketch and generates predictions over the supported object categories. The game evaluates each model separately against the target label.

**Important:** The models act as independent judges rather than combining their predictions into a single ensemble output.

## 📊 Dataset

Drawasaurus uses a selected subset of Google's **Quick, Draw!** dataset, a large collection of human-drawn sketches organized into object categories.

- **Dataset repository:** [Google Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset)
- **NumPy bitmap files:** [Quick, Draw! Bitmap Dataset](https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/)

### Supported Categories

The default configuration includes ten object classes:

-apple
-bicycle
-car
-cat
-fish
-house
-star
-sun
-tree
-umbrella

### Training Data

The training pipeline supports configurable sample counts per category. The earlier configuration allows up to 2,500 drawings per class, subject to dataset availability and the selected training settings.

The `classes.txt` file defines the supported labels. Its contents and ordering must remain consistent with the class configuration used by the corresponding model checkpoints.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application logic and model training |
| Streamlit | Web interface and gameplay |
| Streamlit Drawable Canvas | Interactive sketching interface |
| PyTorch | Deep-learning training and inference |
| Torchvision | ResNet-18, VGG-16, and MobileNetV3-Small architectures |
| NumPy | Numerical and image-array processing |
| Pillow | Image manipulation and resizing |
| Quick, Draw! | Training dataset |
| Git | Version control |
| Git LFS | Large model-checkpoint storage |

## 📁 Project Structure

```text
Drawasaurus/
├── app.py                          # Main game application
├── pico.py                         # Pico model architecture
├── train.py                        # Original training pipeline
├── train_pretrained.py             # Pretrained-model training pipeline
├── requirements.txt                # Python dependencies
├── classes.txt                     # Supported object categories
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore rules
├── .gitattributes                  # Git LFS configuration
├── data/                           # Downloaded training data
└── weights/                        # Trained model checkpoints
    ├── resnet18.pt
    ├── vgg16_pretrained.pt
    ├── mobilenet_v3_small_pretrained.pt
    └── best_pico_weights.pt
```

The `data/` directory may be generated locally during training. The `weights/` directory contains checkpoints required for inference. Large checkpoint files may be tracked using Git LFS.

*Note: The exact repository contents may vary by version. Verify checkpoint filenames against the current application configuration.*

## ⚙️ Installation and Setup

### Prerequisites

- Python installed on your system
- Git
- Git LFS for repositories that use it to distribute model checkpoints
- Sufficient memory and storage for the dependencies and model weights

### 1. Clone the Repository

```bash
git clone https://github.com/Aayushi-Banthia/Drawasaurus.git
cd Drawasaurus
```

### 2. Download Model Checkpoints

If the repository uses Git LFS for model weights, install Git LFS using the [official installation guide](https://git-lfs.com/).

Initialize Git LFS and retrieve the tracked files:

```bash
git lfs install
git lfs pull
```

If the required checkpoints are already available locally, proceed to the next step.

### 3. Create a Virtual Environment

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PyTorch or Torchvision installation fails, consult the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) and select the appropriate configuration for your operating system and hardware.

### 5. Verify Model Weights

Before launching the application, check that the required model checkpoints exist and are compatible with their respective architectures.

Missing or incompatible checkpoints may prevent individual AI judges from loading correctly.

## 🎮 Run the Application

Once the dependencies and required checkpoints are available, start the game:

```bash
python -m streamlit run app.py
```

Open the local URL printed in your terminal. By default, Streamlit typically uses:

```text
http://localhost:8501
```

You can now start drawing and challenge the four AI judges.

**You do not need to train the models just to play** if the required compatible checkpoints are already available.

## 🧪 Training the Models (Optional)

Drawasaurus includes training scripts for experimenting with model training and initialization strategies.

### Original Training Pipeline

```bash
python train.py
```

The original pipeline prepares the selected Quick, Draw! dataset, preprocesses the images, trains the supported models, and saves checkpoints according to the implementation.

### ImageNet-Pretrained Training Pipeline

```bash
python train_pretrained.py
```

This pipeline uses ImageNet-pretrained initialization for the supported ResNet-18, VGG, and MobileNet architectures before training on the selected drawing categories.

### Quick Training Test

For a smaller experimental run, reduce the sample count and number of epochs.

**Windows PowerShell**

```powershell
$env:QD_SAMPLES_PER_CLASS="200"
$env:QD_EPOCHS="1"
python train.py
```

**macOS / Linux**

```bash
QD_SAMPLES_PER_CLASS=200 QD_EPOCHS=1 python train.py
```

These settings are intended to test the training pipeline, not to produce production-quality models.

**Note:** Pico has a separate model implementation. Do not assume that either training script retrains Pico unless its training procedure is explicitly implemented in that script.

Training time depends on the dataset size, architecture, hardware, and training configuration. A compatible CUDA-enabled GPU can accelerate training when supported by the installed PyTorch environment.

## 🔍 Image Preprocessing and Inference

When a player submits a drawing, the application converts the canvas output into an image representation suitable for model inference.

The general workflow is:

1. Capture the submitted canvas.
2. Process the image using the application's preprocessing pipeline.
3. Resize and convert the image into the required tensor format.
4. Pass the processed input to each loaded model.
5. Obtain the predicted class and confidence information, where available.
6. Compare each top prediction with the target category.
7. Calculate the round score and update the cumulative score.

Preprocessing, input dimensions, normalization, and class-label ordering must remain compatible with the corresponding model checkpoints.

## ⚠️ Limitations

- Recognition is restricted to the categories represented in the trained models.
- Ambiguous, incomplete, or unconventional sketches may be misclassified.
- Prediction quality depends on the training data, preprocessing pipeline, and checkpoint compatibility.
- Training may require considerable time and computational resources.
- Confidence scores do not influence the current scoring system.
- Gameplay follows a fixed five-round format.
- Persistent leaderboards, user profiles, and real-time multiplayer are not part of the current core implementation.
- Checkpoint availability depends on repository contents and Git LFS configuration.

## 🚀 Future Enhancements

Potential directions for expanding Drawasaurus include:

- Timed rounds and countdowns.
- Persistent leaderboards and player profiles.
- Real-time multiplayer drawing battles.
- Difficulty levels and adaptive drawing prompts.
- Model accuracy comparison dashboards.
- Animated AI verdicts and more detailed prediction feedback.
- Optional confidence-based bonus scoring.
- Improved mobile responsiveness and accessibility.
- Additional drawing categories and model architectures.

## 📜 Dataset Attribution

Drawasaurus uses data from Google's Quick, Draw! dataset.

- [Quick, Draw! Dataset Repository](https://github.com/googlecreativelab/quickdraw-dataset)
- [Quick, Draw! NumPy Bitmap Data](https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/)

Consult the dataset's official documentation and applicable licensing terms before redistributing dataset contents or using them in other projects.

## 👥 Contributors

Drawasaurus is a collaborative project.

---

**Built with Python, Streamlit, PyTorch, and a little artistic creativity. 🎨🤖**
