# Speaker Identification System

Diploma project: Closed-set speaker identification comparing classical ML baselines (GMM-UBM, SVM) with deep learning approaches (CNN, ECAPA-TDNN).

## Project Structure

```
voice-classification/
├── configs/          # YAML configuration files
├── src/              # Source code
│   ├── data/         # Data loading, features, augmentation
│   ├── models/       # GMM, SVM, CNN, ECAPA-TDNN
│   ├── training/     # Training loop, losses, metrics
│   └── evaluation/   # Evaluation, embeddings, visualization
├── scripts/          # CLI training and evaluation scripts
├── notebooks/        # Jupyter notebooks for analysis and thesis figures
├── tests/            # Unit tests
├── data/             # Dataset (not in git)
├── checkpoints/      # Saved models (not in git)
└── results/          # Figures and metrics (not in git)
```

## Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Dataset

The project supports **VoxCeleb1** (download from https://mm.kaist.ac.kr/datasets/voxceleb/). Place the data in `data/raw/voxceleb1/`.

If VoxCeleb1 is not available, a **synthetic dataset** is generated automatically for development and testing.

## Training

### 1. Classical Baselines (GMM, SVM)

```bash
python scripts/train_baseline.py --config configs/baseline_gmm.yaml
python scripts/train_baseline.py --config configs/baseline_svm.yaml
```

### 2. CNN

```bash
python scripts/train_cnn.py --config configs/cnn.yaml
```

### 3. ECAPA-TDNN

```bash
python scripts/train_ecapa.py --config configs/ecapa_tdnn.yaml
```

## Evaluation

Compare all trained models on the test set:

```bash
python scripts/evaluate_all.py
```

## Inference

Identify speaker from a single audio file:

```bash
python scripts/infer.py --audio path/to/audio.wav --model checkpoints/ecapa_tdnn_best.pt --config configs/ecapa_tdnn.yaml
```

## Models

| Model | Features | Approach |
|-------|----------|----------|
| GMM-UBM | MFCCs | Universal Background Model + per-speaker adaptation |
| SVM | MFCC statistics | RBF kernel SVM on mean+std supervectors |
| CNN | Mel-spectrograms | VGG-style 2D CNN with temporal average pooling |
| ECAPA-TDNN | Mel-spectrograms | SE-Res2Net blocks + attentive statistics pooling + AAM-Softmax |

## Notebooks

1. `01_data_exploration.ipynb` - Dataset statistics and visualization
2. `02_feature_analysis.ipynb` - MFCC vs Mel-spectrogram comparison
3. `03_baseline_experiments.ipynb` - GMM and SVM training
4. `04_cnn_experiments.ipynb` - CNN training and analysis
5. `05_ecapa_experiments.ipynb` - ECAPA-TDNN training and analysis
6. `06_model_comparison.ipynb` - Side-by-side model comparison
7. `07_embedding_analysis.ipynb` - t-SNE and cosine similarity analysis

## Tests

```bash
python -m pytest tests/ -v
```

## Key References

- Desplanques, B., Thienpondt, J., & Demuynck, K. (2020). "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification." *Interspeech 2020.*
- Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). "ArcFace: Additive Angular Margin Loss for Deep Face Recognition." *CVPR 2019.*
- Park, D. S., et al. (2019). "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition." *Interspeech 2019.*
