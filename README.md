# 🎨 Drawasaurus — AI Draw Battle

**Drawasaurus: AI Draw Battle** is an interactive, whiteboard-based drawing game inspired by Drawasaurus, built using Streamlit and deep learning.

The objective is simple: players are given the name of an object and must draw it on a digital whiteboard. Once submitted, the drawing is evaluated by three independently trained AI classifiers — **ResNet, VGG, and MobileNet** — that attempt to recognize the object.

The more AI judges a player can convince, the more points they earn!

The project combines an engaging drawing-game experience with practical applications of computer vision, image classification, and deep learning.

## ✨ Features

- **Interactive Drawing Whiteboard:** Draw objects freehand using the Streamlit canvas, with options to clear and redraw.
- **Randomized Drawing Prompts:** Receive an object name to draw in each round.
- **Three AI Judges:** Every submitted drawing is evaluated by three separate deep-learning models.
- **Independent Predictions:** Each model predicts the most likely object class independently.
- **Confidence Scores:** View the confidence score associated with each judge's prediction.
- **Dynamic Scoring System:** Earn points for every AI judge that correctly identifies your drawing.
- **Multi-Round Gameplay:** Continue playing with new drawing prompts and accumulate points.
- **Quick, Draw! Dataset Integration:** Train the classifiers using a selected subset of Google's Quick, Draw! dataset.
- **Custom Model Training:** Train ResNet, VGG, and MobileNet on the same selected object categories.
- **CPU and GPU Support:** Use CUDA when available, with CPU fallback through PyTorch.
- **Cached Model Loading:** Reuse loaded models during gameplay instead of reloading them for every interaction.

## 🕹️ How to Play

1. Start the game and view the target object displayed on the screen.
2. Draw the object on the digital whiteboard.
3. Submit your drawing for evaluation.
4. Let all three AI judges analyze your sketch.
5. Review their predictions and confidence scores.
6. Earn points for every judge that correctly recognizes the target.
7. Move to the next round and try to convince all three judges!

### 🏆 Scoring System

Each correct judge awards **10 points**.

| Correct AI Judges | Points Earned |
|:---:|:---:|
| 0 out of 3 | 0 |
| 1 out of 3 | 10 |
| 2 out of 3 | 20 |
| 3 out of 3 | 30 |

Scoring is based on each model's top prediction. Confidence scores are displayed for reference but do not currently affect the points awarded.

## 🧠 Meet the AI Judges

The game uses three different convolutional neural network architectures to classify player drawings.

| Model | Architecture | Purpose |
|---|---|---|
| **ResNet** | ResNet-18 | Classifies drawings using residual connections. |
| **VGG** | VGG-11 | Classifies drawings using stacked convolutional layers. |
| **MobileNet** | MobileNetV3-Small | Provides an efficient image-classification model designed with lightweight computation in mind. |

Each model is trained separately using the same selected dataset categories. Their predictions are compared against the target object to determine the player's score.

## 📊 Dataset

This project uses a subset of Google's **Quick, Draw! dataset**, which contains millions of human-drawn doodles organized into object categories.

**Dataset:** [Google Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset)

The training script downloads rasterized bitmap files from the dataset and uses a configurable number of examples from each selected category.

### Default Categories

- Apple
- Cat
- Airplane
- Fish
- House
- Tree
- Star
- Bicycle

The default configuration samples up to 2,500 drawings per category.

## 🛠️ Tech Stack

| Technology | Usage |
|---|---|
| Python | Core programming language |
| Streamlit | Web interface and game logic |
| Streamlit Drawable Canvas | Interactive drawing whiteboard |
| PyTorch | Deep-learning model training and inference |
| Torchvision | ResNet, VGG, and MobileNet architectures |
| NumPy | Dataset and array processing |
| Pillow | Image preprocessing and resizing |
| Quick, Draw! Dataset | Training data for the classifiers |

## 📁 Project Structure

```text
Drawasaurus/
├── app.py              # Streamlit game interface
├── train.py            # Dataset download and model training
├── requirements.txt    # Project dependencies
├── classes.txt         # Selected model classes
├── README.md           # Project documentation
├── .gitignore          # Files excluded from Git
├── data/               # Downloaded dataset files
└── weights/            # Trained model weights
    ├── resnet18.pt
    ├── vgg11.pt
    └── mobilenet_v3_small.pt
```

The `data/` and `weights/` directories are generated locally and are excluded from Git by default.

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Aayushi-Banthia/Drawasaurus.git
cd Drawasaurus
```

### 2. Create a Virtual Environment

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PyTorch installation encounters an issue, refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

## 🚀 Training the AI Judges

Before playing, train the three models using the selected Quick, Draw! categories.

Run:

```bash
python train.py
```

The training pipeline:

1. Downloads the required dataset files.
2. Selects a subset of drawings from each category.
3. Preprocesses the images for model input.
4. Trains ResNet-18, VGG-11, and MobileNetV3-Small independently.
5. Saves the trained weights in the `weights/` directory.

Training all three models can take considerable time on a CPU. A CUDA-capable GPU is recommended for faster training.

### Quick Training Test

To test the pipeline with fewer examples and one epoch, use:

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

These settings are intended for a quick test, not for achieving strong classification performance.

## 🎮 Running the Game

Once training has completed successfully and the three model weight files have been generated, launch the application:

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually:

`http://localhost:8501`

You can now draw objects, submit your sketches, and see how many AI judges you can convince!

## ⚠️ Current Limitations

- The game requires trained model weights, which are not included in the repository by default.
- Recognition is limited to the object categories used during training.
- The models may misclassify ambiguous, incomplete, or unusual drawings.
- The current scoring system uses top-1 predictions and does not incorporate confidence into scoring.
- Persistent leaderboards, multiplayer gameplay, and countdown timers are not currently implemented.

## 🔮 Future Enhancements

Potential improvements include:

- ⏱️ Timed drawing rounds.
- 🏅 Persistent leaderboards and player profiles.
- 👥 Multiplayer rooms and real-time competitions.
- 🎯 Difficulty levels and adaptive prompts.
- 📈 Model performance comparisons and evaluation metrics.
- ✨ Animated AI judge verdicts and enhanced visual feedback.
- 🧮 Optional confidence-based bonus scoring.
- 📱 A more responsive interface for mobile devices.

## 📜 Dataset Attribution

This project uses the Google Quick, Draw! dataset.

- **Dataset repository:** https://github.com/googlecreativelab/quickdraw-dataset
- **Bitmap data source:** https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/
- **License:** [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

Please follow the dataset's attribution requirements when using or redistributing its contents.

---

**Built with Python, Streamlit, PyTorch, and a little artistic creativity.** 🎨🤖
