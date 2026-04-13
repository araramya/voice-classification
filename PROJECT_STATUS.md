# Speaker Identification — Project Status and Plan

---

## PART 1: WHAT WAS DONE

### 1.1 Built the Complete System

We built a full speaker identification system from scratch. The system takes an audio recording as input and determines which person is speaking. It includes everything needed for a diploma: data processing, 4 different models, training, evaluation, and visualization.

**The system has 4 models that we compare:**

| Model | Type | Era | What it does |
|-------|------|-----|-------------|
| GMM-UBM | Classical ML | 2000s | Models each speaker's voice as a statistical probability distribution |
| SVM | Classical ML | 2000s | Finds decision boundaries between speakers using summary statistics |
| CNN | Deep Learning | 2017 | Treats voice spectrograms as images, learns visual patterns |
| ECAPA-TDNN | Deep Learning | 2020 | State-of-the-art architecture with attention and multi-scale analysis |

The whole point is to show how speaker identification evolved from basic statistics to modern neural networks.

### 1.2 Generated Synthetic Test Data

Since real data (VoxCeleb1) requires download and registration, we created a synthetic data generator that produces artificial voices:

- **50 speakers**, each with a unique frequency signature
- **30 recordings per speaker** = 1,500 audio files total
- Each file is 3 seconds long at 16kHz
- Located at: `data/raw/voxceleb1/id10001-id10050/synthetic/*.wav`

This synthetic data is intentionally simple — it's meant to **verify the system works**, not to produce meaningful research results.

### 1.3 Trained All 4 Models

Every model was trained on the synthetic data. Here is what we have:

**Saved model files** (in `checkpoints/`):

| File | Size | What it is |
|------|------|-----------|
| `baseline_gmm.pkl` | 8.0 MB | Trained GMM-UBM model |
| `baseline_svm.pkl` | 0.9 MB | Trained SVM model |
| `cnn_mel_spectrogram_best.pt` | 59.5 MB | Best CNN model (saved at epoch 32) |
| `ecapa_tdnn_best.pt` | 64.4 MB | Best ECAPA-TDNN model (saved at epoch 43) |

**Training details:**

| Model | Epochs | Time | Notes |
|-------|--------|------|-------|
| GMM-UBM | N/A (single pass) | ~2 min | Trained UBM with 256 components, then adapted per speaker |
| SVM | N/A (single pass) | ~1 min | RBF kernel SVM on 80-dim supervectors |
| CNN | 37 out of 80 (early stopped) | ~45 min | Reached 100% val accuracy by epoch 3, kept improving loss until epoch 32 |
| ECAPA-TDNN | 63 out of 100 (early stopped) | ~48 min | Reached 100% val accuracy by epoch 15, best loss at epoch 43 |

Both deep learning models stopped early — the system detected that the models had converged (validation loss stopped improving for 15-20 epochs) and automatically stopped training to prevent wasting time.

### 1.4 Evaluated and Compared All Models

**Results** (in `results/comparison/comparison.json`):

| Model | Accuracy | Top-5 Accuracy | EER |
|-------|----------|---------------|-----|
| GMM-UBM | 100.0% | — | 0.0% |
| SVM | 100.0% | — | 0.0% |
| CNN | 100.0% | 100.0% | 0.0% |
| ECAPA-TDNN | 100.0% | 100.0% | 0.0% |

**Why all 100%:** Synthetic data has very distinct frequency signatures per speaker — any reasonable model can perfectly separate them. These results prove the system works correctly, but they don't show meaningful differences between models. Real differences will appear with VoxCeleb1 data.

**What the metrics mean:**
- **Accuracy** — percentage of test recordings correctly identified
- **Top-5 Accuracy** — percentage where the correct speaker is in the top 5 predictions
- **EER (Equal Error Rate)** — the balanced error rate where false acceptance = false rejection. 0% means perfect. This metric is only meaningful when models make mistakes.

### 1.5 What Files Were Produced

**Results you can show:**

| File | What it shows |
|------|--------------|
| `results/comparison/comparison.png` | Bar chart comparing all 4 models |
| `results/comparison/comparison.json` | Raw numbers: accuracy and EER per model |
| `results/baseline_gmm/metrics.json` | GMM results: 100% accuracy, 50 speakers, 225 test samples |
| `results/baseline_svm/metrics.json` | SVM results: 100% accuracy |
| `results/cnn_mel_spectrogram/metrics.json` | CNN results: accuracy, EER, full 50x50 confusion matrix |
| `results/ecapa_tdnn/metrics.json` | ECAPA-TDNN results: same as CNN |
| `results/ecapa_tdnn/training_history.json` | Epoch-by-epoch loss and accuracy for ECAPA-TDNN training |

**Training logs:**

| File | What it shows |
|------|--------------|
| `results/baseline_gmm.log` | GMM training log with timestamps |
| `results/baseline_svm.log` | SVM training log |
| `results/cnn_mel_spectrogram.log` | CNN training log |
| `results/ecapa_tdnn.log` | ECAPA-TDNN training log |

**TensorBoard** (interactive training dashboard):
Run `tensorboard --logdir runs` and open `http://localhost:6006` — shows interactive loss/accuracy charts.

**Saved data:**

| File | What it is |
|------|-----------|
| `data/processed/metadata.csv` | Index of all 1,500 audio files with speaker IDs and durations |
| `data/splits/splits.json` | Exact train/val/test split (which files go where) |
| `data/splits/selected_speakers.json` | List of the 50 selected speakers |
| `data/splits/label_encoder.pkl` | Mapping from speaker names to numbers (needed for inference) |

### 1.6 Tested Inference

The inference pipeline works — you can give it any audio file and it identifies the speaker:

```
Input:  data/raw/voxceleb1/id10001/synthetic/00000.wav
Output: Predicted Speaker: id10001, Confidence: 100.0%
```

This was tested with both the CNN and ECAPA-TDNN models on different speakers.

### 1.7 Issues Found and Fixed

| Problem | What happened | How we fixed it |
|---------|-------------|----------------|
| Audio loading crash | torchaudio 2.11 requires FFmpeg on Windows, which is not installed | Replaced with soundfile library for reading WAV files |
| Windows multiprocessing hang | DataLoader with num_workers>0 freezes on Windows | Set num_workers=0 |
| Notebook paths wrong | Notebooks couldn't find the data folder | Fixed relative paths |
| ECAPA-TDNN didn't learn | AAM-Softmax (ArcFace) loss couldn't converge on synthetic data | Switched to cross-entropy loss for synthetic experiments |
| Test assertion error | Label type was numpy.int64 instead of Python int | Updated the test to accept both types |

---

## PART 2: WHAT WE STILL NEED TO DO

### Phase 1: Get Real Data (VoxCeleb1) — IN PROGRESS

**Status:** You received the download links by email. Links expire in 1 week.

**What to download:**
- VoxCeleb1 Dev Part A — click Download
- VoxCeleb1 Dev Part B — click Download
- VoxCeleb1 Dev Part C — click Download
- VoxCeleb1 Dev Part D — click Download (had connection error, retry needed)

**After downloading all 4 parts, combine them:**
```bash
cat vox1_dev_wav_parta* vox1_dev_wav_partb* vox1_dev_wav_partc* vox1_dev_wav_partd* > vox1_dev_wav.zip
```

**Total size:** ~30 GB

**What's inside:** Real celebrity voice recordings — 1,251 speakers, ~150,000 utterances extracted from YouTube interviews. This is the standard benchmark dataset used in speaker recognition research.

### Phase 2: Set Up Google Colab

Training deep learning models on real data requires a GPU. Your laptop only has a CPU, which is too slow. Google Colab provides a free GPU (NVIDIA T4, 16GB).

**What needs to be done:**
1. Upload the project folder to Google Drive
2. Upload VoxCeleb1 zip to Google Drive
3. Create a Colab notebook that:
   - Mounts your Drive
   - Installs dependencies
   - Extracts VoxCeleb1
   - Trains all 4 models with GPU acceleration
   - Saves results back to Drive

**Config changes needed for real data:**
- `num_speakers`: increase from 50 to 100-200 (VoxCeleb1 has 1,251 speakers)
- `num_workers`: change from 0 to 2 (Colab is Linux, multiprocessing works)
- ECAPA-TDNN loss: change from `cross_entropy` back to `aam_softmax` (works on real data)

### Phase 3: Train on Real Data

Run training for all 4 models on VoxCeleb1. Expected times on Colab GPU:

| Model | Expected Time | Expected Accuracy |
|-------|--------------|-------------------|
| GMM-UBM | 10-30 min | 70-85% |
| SVM | 5-15 min | 75-90% |
| CNN | 30-60 min | 85-95% |
| ECAPA-TDNN | 60-120 min | 95-99% |

**This is where the interesting results come.** Unlike synthetic data where everything is 100%, real data will show clear differences between models:
- Classical methods (GMM, SVM) will struggle with similar-sounding speakers and noisy recordings
- CNN will do better by learning patterns automatically
- ECAPA-TDNN will be the best because of its attention mechanisms and multi-scale analysis

### Phase 4: Generate Thesis Figures

Run the 7 Jupyter notebooks to produce publication-quality figures:

| Notebook | What it generates | Why it matters for the thesis |
|----------|-------------------|------------------------------|
| 01 - Data Exploration | Speaker distribution chart, waveform plots | Shows the dataset characteristics |
| 02 - Feature Analysis | Mel-spectrogram vs MFCC comparison, SpecAugment demo | Explains the feature extraction process visually |
| 03 - Baseline Experiments | GMM & SVM confusion matrices | Shows classical model performance |
| 04 - CNN Experiments | Training curves (loss/accuracy over epochs), confusion matrix | Shows how CNN learns |
| 05 - ECAPA Experiments | Same as CNN but for ECAPA-TDNN | Shows state-of-the-art performance |
| 06 - Model Comparison | Side-by-side bar chart of all 4 models | The key result figure for the thesis |
| 07 - Embedding Analysis | t-SNE plots showing speaker clusters, cosine similarity distributions | Visualizes what the models actually learned |

**The most important figures for the thesis:**
- Training curves (04, 05) — show the model actually learned something
- Model comparison chart (06) — the main result
- t-SNE plots (07) — show that ECAPA-TDNN creates better speaker clusters than CNN
- Confusion matrices (03, 04, 05) — show where each model makes mistakes

### Phase 5: Write the Thesis

With all results and figures, the thesis structure would be:

1. **Introduction** — what is speaker identification, why it matters
2. **Literature Review** — history from GMM-UBM (2000s) to ECAPA-TDNN (2020)
3. **Methodology** — feature extraction, data augmentation, all 4 model architectures
4. **Experimental Setup** — VoxCeleb1 dataset, training configuration, evaluation metrics
5. **Results** — accuracy tables, confusion matrices, training curves, EER comparison
6. **Discussion** — why deep learning outperforms classical methods, what ECAPA-TDNN does differently
7. **Conclusion** — summary of findings, future work

---

## SUMMARY

| Phase | Status | What |
|-------|--------|------|
| Build the system | DONE | All code, 4 models, training, evaluation, inference |
| Test with synthetic data | DONE | All models trained, 100% accuracy, pipeline validated |
| Fix bugs | DONE | 5 issues found and resolved |
| Get VoxCeleb1 access | DONE | Download links received via email |
| Download VoxCeleb1 | IN PROGRESS | 4 parts to download (~30GB total), Part D had connection error |
| Set up Google Colab | TODO | Need to create Colab notebook for GPU training |
| Train on real data | TODO | The main experiment — will show real differences between models |
| Generate thesis figures | TODO | 7 notebooks to run for publication-quality plots |
| Write thesis | TODO | Using results and figures from all experiments |
