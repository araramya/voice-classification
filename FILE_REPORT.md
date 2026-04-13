# What Each File Does

## configs/ — Settings for each experiment

| File | What it does |
|------|-------------|
| `base.yaml` | Default settings shared by all experiments: 16kHz audio, 80 mel bands, 50 speakers, 70/15/15 split, batch size 64. Every other config inherits from this. |
| `cnn.yaml` | CNN settings: use mel-spectrograms, 80 epochs, cross-entropy loss, SpecAugment enabled, embedding size 192. |
| `ecapa_tdnn.yaml` | ECAPA-TDNN settings: use mel-spectrograms, 100 epochs, 512 channels, scale 8, noise augmentation enabled. |
| `baseline_gmm.yaml` | GMM settings: use MFCCs, no augmentation, model type "gmm". |
| `baseline_svm.yaml` | SVM settings: use MFCCs, no augmentation, model type "svm". |

---

## src/ — All the code that makes the system work

### Core files

| File | What it does |
|------|-------------|
| `config.py` | Reads YAML config files and converts them into Python objects. Handles config inheritance — when cnn.yaml says `_base_: "base.yaml"`, this file loads base.yaml first and merges the CNN overrides on top. |
| `utils.py` | Small helper functions: `set_seed()` makes experiments reproducible, `get_device()` detects if GPU is available, `setup_logging()` creates log files, `count_parameters()` counts how many numbers a model has to learn. |
| `inference.py` | The "use the model" file. Takes an audio file + trained model, loads the audio, extracts features, runs the model, returns who's speaking with confidence scores. This is what you'd use in a real application. |

### src/data/ — Getting data ready for training

| File | What it does |
|------|-------------|
| `download.py` | Prepares the dataset. Checks if VoxCeleb1 exists — if not, generates synthetic audio (50 speakers, 30 recordings each, unique frequencies per speaker). Scans all WAV files and creates a metadata table (which file belongs to which speaker). |
| `features.py` | Converts raw audio into numbers the models can understand. Two options: **mel-spectrogram** (80-band frequency heatmap, used by CNN and ECAPA-TDNN) or **MFCC** (40 compact coefficients, used by GMM and SVM). |
| `augmentation.py` | Makes training data more varied to prevent overfitting. **AudioAugmentor** adds noise and changes volume on raw audio. **SpecAugmentor** masks random strips of the spectrogram (SpecAugment technique from Google). Only used during training, never during testing. |
| `dataset.py` | Connects everything for PyTorch. When the model asks for sample #42, this file loads the WAV file, converts to mono, crops/pads to 3 seconds, applies augmentation, extracts features, and returns a ready-to-use tensor. Two versions: `SpeakerDataset` for deep learning (returns tensors), `SpeakerDatasetForBaseline` for classical ML (returns numpy arrays). |
| `splits.py` | Divides data into training (70%), validation (15%), and test (15%) sets. Uses stratified splitting so every speaker appears in all three sets. Saves the split to JSON so every experiment uses the same division. |

### src/models/ — The 4 speaker identification models

| File | What it does |
|------|-------------|
| `baseline_gmm.py` | **GMM-UBM model.** First trains a Universal Background Model (256 Gaussians) on all speakers combined to learn "what speech sounds like in general." Then adapts this model slightly for each individual speaker. To identify someone, checks which speaker's model best explains the test audio. |
| `baseline_svm.py` | **SVM model.** Compresses each recording into 80 numbers (mean + standard deviation of 40 MFCCs), normalizes them, then trains a Support Vector Machine to find decision boundaries between speakers. |
| `cnn.py` | **CNN model.** Four convolutional blocks (64→128→256→512 filters) that scan the spectrogram like an image, detecting patterns at increasing scales. Pools over time to handle variable lengths, then produces a 192-number speaker embedding. A final layer maps the embedding to speaker scores. |
| `ecapa_tdnn.py` | **ECAPA-TDNN model.** The best model. Uses three SE-Res2Net blocks with increasing dilation (2,3,4) to capture patterns at multiple scales. Combines outputs from all three blocks (multi-layer feature aggregation). Uses attention to focus on the most informative parts of the audio. Produces a 192-number speaker embedding. |
| `layers.py` | Building blocks shared by CNN and ECAPA-TDNN. **SEBlock**: learns which channels are important and amplifies them. **Res2Conv1dBlock**: splits channels into groups and processes them hierarchically for multi-scale patterns. **SERes2NetBlock**: combines SE + Res2Net + residual connection. **AttentiveStatisticsPooling**: learns to weight time frames by importance, computes weighted mean and standard deviation. **TemporalAveragePooling**: simple averaging over time. |

### src/training/ — How models learn

| File | What it does |
|------|-------------|
| `trainer.py` | The main training loop. For each epoch: feeds batches through the model, computes loss, updates weights, checks validation performance, saves the best model, stops early if no improvement for 15-20 epochs. Supports mixed precision (faster on GPU) and gradient clipping (prevents training explosions). Logs everything to TensorBoard. |
| `losses.py` | **AAM-Softmax (ArcFace) loss.** A specialized loss that forces speaker embeddings to be well-separated on a unit sphere by adding angular margins between classes. Makes embeddings more discriminative than standard cross-entropy. Designed for real data — didn't work on synthetic data. |
| `schedulers.py` | Controls learning rate over time. **Cosine warmup**: starts small (5 epochs warmup), then gradually decreases following a cosine curve. This helps the model learn quickly at first, then fine-tune carefully. Alternative: **step decay** (halve LR every 20 epochs). |
| `metrics.py` | Computes evaluation numbers. **Accuracy**: % correct predictions. **Top-5 accuracy**: is the correct answer in top 5? **EER (Equal Error Rate)**: the balanced error rate where false accepts = false rejects (standard metric in speaker recognition). **Confusion matrix**: table showing predictions vs truth for every speaker. |

### src/evaluation/ — Measuring how good the models are

| File | What it does |
|------|-------------|
| `embeddings.py` | Passes all test audio through a trained model and collects the 192-number speaker embeddings. These vectors represent each speaker's "voice fingerprint" and are used for visualization and EER computation. |
| `evaluate.py` | Runs the complete evaluation: computes accuracy, top-5 accuracy, confusion matrix, extracts embeddings, computes EER. Also has `compare_models()` which takes results from all 4 models and creates a comparison dictionary. |
| `visualization.py` | Generates thesis-quality figures (300 DPI). Training curves (loss/accuracy over epochs), confusion matrix heatmaps, t-SNE scatter plots of speaker embeddings, DET curves for EER, model comparison bar charts, spectrogram examples. |

---

## scripts/ — Run these to train and evaluate

| File | What it does |
|------|-------------|
| `train_baseline.py` | Trains GMM or SVM. Loads config, prepares data, extracts MFCC features grouped by speaker, trains the model, evaluates on test set, saves model (.pkl) and results (.json). Run with: `python scripts/train_baseline.py --config configs/baseline_gmm.yaml` |
| `train_cnn.py` | Trains CNN. Loads config, creates datasets with augmentation, builds CNN model, creates Trainer, runs training loop with early stopping and checkpointing. Run with: `python scripts/train_cnn.py --config configs/cnn.yaml` |
| `train_ecapa.py` | Trains ECAPA-TDNN. Same flow as train_cnn.py but creates ECAPA-TDNN model with 512 channels and scale 8. Run with: `python scripts/train_ecapa.py --config configs/ecapa_tdnn.yaml` |
| `evaluate_all.py` | Loads all 4 trained models, evaluates each on the test set, computes comparison metrics, generates comparison bar chart. Saves everything to `results/comparison/`. Run with: `python scripts/evaluate_all.py` |
| `infer.py` | Identifies the speaker in a single audio file. Loads a trained model, processes the audio, prints: predicted speaker, confidence %, top-5 predictions, embedding shape. Run with: `python scripts/infer.py --audio path/to/file.wav --model checkpoints/ecapa_tdnn_best.pt --config configs/ecapa_tdnn.yaml --label-encoder data/splits/label_encoder.pkl` |

---

## tests/ — Verify the code works

| File | What it does |
|------|-------------|
| `test_features.py` | 4 tests. Checks that mel-spectrogram output has the right shape (1, 80, time), MFCC output has the right shape (40, time), no NaN/Inf values appear, and 1D input is handled correctly. |
| `test_dataset.py` | 3 tests. Creates a tiny synthetic dataset (3 speakers, 5 utterances each), verifies the metadata has correct columns, and checks that the PyTorch dataset returns tensors with correct shapes and label types. |
| `test_models.py` | 5 tests. Feeds a test batch through CNN and ECAPA-TDNN, verifies output shapes match (batch, num_speakers) for classification and (batch, 192) for embeddings. Also tests that ECAPA-TDNN handles different audio lengths. |

Run all 12 tests with: `python -m pytest tests/ -v`

---

## notebooks/ — Interactive analysis and thesis figures

| File | What it does |
|------|-------------|
| `01_data_exploration.ipynb` | Shows dataset statistics: how many utterances per speaker (bar chart), duration distribution (histogram), waveform visualizations for different speakers, and audio playback. |
| `02_feature_analysis.ipynb` | Compares mel-spectrogram vs MFCC side by side for the same audio. Demonstrates SpecAugment by showing original and augmented spectrograms. |
| `03_baseline_experiments.ipynb` | Trains and evaluates GMM and SVM interactively. Shows confusion matrices inline. |
| `04_cnn_experiments.ipynb` | Analyzes CNN training: plots loss and accuracy curves over epochs, generates confusion matrix heatmap. |
| `05_ecapa_experiments.ipynb` | Same as 04 but for ECAPA-TDNN. Also compares loss functions if multiple were tried. |
| `06_model_comparison.ipynb` | The main results notebook. Loads results from all 4 models, creates side-by-side comparison bar charts for accuracy and EER. |
| `07_embedding_analysis.ipynb` | Visualizes what the models learned. Uses t-SNE to project 192-dim embeddings into 2D scatter plots (each color = one speaker). Shows cosine similarity distributions between same-speaker and different-speaker pairs. |

Run with: `cd notebooks && jupyter notebook`

---

## Generated at runtime (not code — created when you run the system)

### data/

| Path | What it is |
|------|-----------|
| `data/raw/voxceleb1/id10001/synthetic/*.wav` | The actual audio files (50 speakers x 30 files) |
| `data/processed/metadata.csv` | Table listing every audio file with speaker ID, file path, and duration |
| `data/splits/splits.json` | Which files go into train/val/test (so experiments are reproducible) |
| `data/splits/selected_speakers.json` | List of the 50 speakers used |
| `data/splits/label_encoder.pkl` | Mapping: "id10001"→0, "id10002"→1, etc. (needed for inference) |

### checkpoints/

| File | What it is |
|------|-----------|
| `baseline_gmm.pkl` | Trained GMM model (8 MB) |
| `baseline_svm.pkl` | Trained SVM model (0.9 MB) |
| `cnn_mel_spectrogram_best.pt` | Best CNN model from training (59.5 MB) |
| `cnn_mel_spectrogram_latest.pt` | Last CNN model saved (59.5 MB) |
| `ecapa_tdnn_best.pt` | Best ECAPA-TDNN model from training (64.4 MB) |
| `ecapa_tdnn_latest.pt` | Last ECAPA-TDNN model saved (64.4 MB) |

### results/

| File | What it is |
|------|-----------|
| `results/comparison/comparison.json` | All 4 models compared: accuracy and EER |
| `results/comparison/comparison.png` | Bar chart visualization of the comparison |
| `results/baseline_gmm/metrics.json` | GMM evaluation results |
| `results/baseline_svm/metrics.json` | SVM evaluation results |
| `results/cnn_mel_spectrogram/metrics.json` | CNN evaluation results with confusion matrix |
| `results/ecapa_tdnn/metrics.json` | ECAPA-TDNN evaluation results with confusion matrix |
| `results/ecapa_tdnn/training_history.json` | Epoch-by-epoch loss and accuracy |
| `results/*.log` | Training logs with timestamps |

### runs/

| Path | What it is |
|------|-----------|
| `runs/cnn_mel_spectrogram/events.out.tfevents.*` | TensorBoard log for CNN training |
| `runs/ecapa_tdnn/events.out.tfevents.*` | TensorBoard log for ECAPA-TDNN training |

View with: `tensorboard --logdir runs` → open `http://localhost:6006`
