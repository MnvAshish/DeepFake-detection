# 🔍 DeepFake Detector

An ensemble deep learning system for detecting deepfake images and videos, with an interactive Streamlit web app for real-time analysis and explainability.

The system combines **ResNet50**, **VGG16**, and **InceptionV3** with **MediaPipe** face detection and a soft-voting ensemble to classify media as REAL or FAKE, and uses **Grad-CAM** to visualize which facial regions influenced each prediction.

---

## ✨ Features

- **Image & Video detection** — upload a single image or a video file and get a REAL/FAKE verdict with a confidence score.
- **Ensemble of 3 CNNs** — ResNet50, VGG16, and InceptionV3 vote together (soft voting, weighted average, or hard voting) for a more robust prediction than any single model.
- **Automatic face detection** — faces are located and cropped with MediaPipe before classification.
- **Grad-CAM explanations** — heatmaps show *why* the model flagged an image as fake.
- **Frame-level timeline** — for videos, see the fake-probability trend across sampled frames.
- **Interactive Streamlit UI** — adjustable frame interval, max frames, face margin, ensemble method, and fake threshold, plus a demo mode with synthetic results (no models required).
- **Per-model breakdown** — compare each model's individual vote alongside the ensemble result.

---

## 🗂️ Project Structure

```
DeepFake-detection/
├── app.py              # Streamlit web app (image + video detection UI)
├── config/             # Configuration files (thresholds, ensemble settings, etc.)
├── src/                # Core source code
│   ├── data/            # Dataset download & preprocessing scripts
│   ├── training/         # Model training pipeline
│   ├── inference/        # Predictor, ensemble logic
│   └── utils/            # Config loader, device helpers, etc.
├── outputs/            # Model checkpoints / generated artifacts
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 🧠 Pipeline

```
Upload (Image or Video)
      │
      ▼
Face Detection (MediaPipe)
      │
      ▼
Preprocessing (resize, normalize)
      │
      ├── ResNet50 (224×224)
      ├── VGG16 (224×224)
      └── InceptionV3 (299×299)
      │
      ▼
Soft Voting Ensemble
      │
      ▼
REAL / FAKE + Confidence
      │
      ▼
Grad-CAM Explanation (image mode)
```

---

## 📦 Supported Datasets

| Dataset | Type | Size | Access |
|---|---|---|---|
| **Celeb-DF v2** | Video | 590 real + 5639 fake | Public (Google Drive) |
| **FaceForensics++** | Video | ~1,000 each | Request form |
| **DFDC** | Video | ~100,000 | Kaggle |
| **Custom** | Video/Image | Any | Your own dataset |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/MnvAshish/DeepFake-detection.git
cd DeepFake-detection
```

### 2. Install dependencies

It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Download a dataset

```bash
python src/data/dataset_downloader.py --dataset celebdf_v2
```

### 4. Preprocess the data

```bash
python src/data/real_dataset_prep.py
```

### 5. Train the models

```bash
python src/training/train_pipeline.py
```

### 6. Launch the web app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`) in your browser.

> 💡 No trained models yet? Use the **Demo** tab in the app to explore the UI with synthetic FAKE/REAL results — no training required.

---

## 🛠️ Tech Stack

- **Deep Learning:** PyTorch, TorchVision
- **Face Detection:** MediaPipe
- **Web UI:** Streamlit, Plotly
- **Computer Vision:** OpenCV, Pillow, imageio
- **Data Science:** NumPy, Pandas, scikit-learn, Matplotlib, Seaborn
- **Other:** Albumentations (augmentation), PyYAML, tqdm, Loguru, TensorBoard

See [`requirements.txt`](./requirements.txt) for exact versions.

---

## ⚙️ Configuration

Runtime behavior (frame extraction interval, max frames per video, face margin, ensemble method, and the fake/real confidence threshold) can be tuned directly from the sidebar in the Streamlit app, or via the config files in [`config/`](./config).

---

## 📊 Using the App

- **Image Detection tab** — upload a JPG/PNG/BMP/WebP image, run analysis, and view the verdict card, per-model gauge/bar charts, and Grad-CAM overlays.
- **Video Detection tab** — upload an MP4/AVI/MOV/MKV video, preview sampled frames, and run analysis to get an aggregated verdict plus a frame-by-frame fake-probability timeline.
- **Demo tab** — instantly view example FAKE/REAL result cards without needing trained models.
- **About tab** — pipeline overview and dataset reference, all from within the app.

---

## ⚠️ Disclaimer

This tool is intended for research and educational purposes. Deepfake detection is an active research area, and no model is guaranteed to be 100% accurate. Predictions should not be used as the sole basis for high-stakes decisions (e.g., legal, journalistic, or identity-verification purposes) without human review.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.

## 📄 License

No license has been specified for this repository yet. Please contact the repository owner for usage terms.
