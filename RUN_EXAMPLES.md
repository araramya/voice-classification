# How to Run Every Model — Exact Commands

## 1. Test Inference (Identify a Speaker)

### With ECAPA-TDNN (best model):
```bash
python scripts/infer.py --audio data/raw/voxceleb1/id10001/synthetic/00000.wav --model checkpoints/ecapa_tdnn_best.pt --config configs/ecapa_tdnn.yaml --label-encoder data/splits/label_encoder.pkl
```

### With CNN:
```bash
python scripts/infer.py --audio data/raw/voxceleb1/id10001/synthetic/00000.wav --model checkpoints/cnn_mel_spectrogram_best.pt --config configs/cnn.yaml --label-encoder data/splits/label_encoder.pkl
```

### Try different speakers (change id10001 to any speaker):
```bash
python scripts/infer.py --audio data/raw/voxceleb1/id10005/synthetic/00015.wav --model checkpoints/ecapa_tdnn_best.pt --config configs/ecapa_tdnn.yaml --label-encoder data/splits/label_encoder.pkl

python scripts/infer.py --audio data/raw/voxceleb1/id10025/synthetic/00010.wav --model checkpoints/ecapa_tdnn_best.pt --config configs/ecapa_tdnn.yaml --label-encoder data/splits/label_encoder.pkl

python scripts/infer.py --audio data/raw/voxceleb1/id10050/synthetic/00000.wav --model checkpoints/cnn_mel_spectrogram_best.pt --config configs/cnn.yaml --label-encoder data/splits/label_encoder.pkl
```

**Expected output for each:**
```
Predicted Speaker: id10025
Confidence: 100.0%

Top-5 Predictions:
  id10025: 100.0%
  id10015: 0.0%
  id10017: 0.0%
  id10026: 0.0%
  id10028: 0.0%

Embedding shape: (192,)
```

The predicted speaker should match the folder name in the audio path.

---

## 2. Train Models (from scratch)

### Generate synthetic dataset first:
```bash
python -c "from src.data.download import prepare_dataset; prepare_dataset('data', 50, 20)"
```

### Train GMM-UBM (~2 minutes):
```bash
python scripts/train_baseline.py --config configs/baseline_gmm.yaml
```

### Train SVM (~1 minute):
```bash
python scripts/train_baseline.py --config configs/baseline_svm.yaml
```

### Train CNN (~45 minutes on CPU):
```bash
python scripts/train_cnn.py --config configs/cnn.yaml
```

### Train ECAPA-TDNN (~48 minutes on CPU):
```bash
python scripts/train_ecapa.py --config configs/ecapa_tdnn.yaml
```

---

## 3. Evaluate All Models

### Compare all 4 models at once:
```bash
python scripts/evaluate_all.py
```

Results saved to:
- `results/comparison/comparison.json` — numbers
- `results/comparison/comparison.png` — chart

---

## 4. Run Tests (verify code works)

```bash
python -m pytest tests/ -v
```

Expected: 12 tests, all pass.

---

## 5. View Training Dashboard

```bash
tensorboard --logdir runs
```

Then open http://localhost:6006 in your browser.

---

## 6. Run Jupyter Notebooks (generate thesis figures)

```bash
cd notebooks
jupyter notebook
```

Run notebooks 01 through 07 in order.

---

## 7. Quick Demo Script

Copy and paste this entire block to run a full demo:

```bash
echo "=== Testing ECAPA-TDNN on 3 different speakers ==="

echo ""
echo "--- Speaker id10001 ---"
python scripts/infer.py --audio data/raw/voxceleb1/id10001/synthetic/00000.wav --model checkpoints/ecapa_tdnn_best.pt --config configs/ecapa_tdnn.yaml --label-encoder data/splits/label_encoder.pkl

echo ""
echo "--- Speaker id10025 ---"
python scripts/infer.py --audio data/raw/voxceleb1/id10025/synthetic/00010.wav --model checkpoints/ecapa_tdnn_best.pt --config configs/ecapa_tdnn.yaml --label-encoder data/splits/label_encoder.pkl

echo ""
echo "--- Speaker id10050 ---"
python scripts/infer.py --audio data/raw/voxceleb1/id10050/synthetic/00000.wav --model checkpoints/ecapa_tdnn_best.pt --config configs/ecapa_tdnn.yaml --label-encoder data/splits/label_encoder.pkl

echo ""
echo "=== Testing CNN on the same 3 speakers ==="

echo ""
echo "--- Speaker id10001 ---"
python scripts/infer.py --audio data/raw/voxceleb1/id10001/synthetic/00000.wav --model checkpoints/cnn_mel_spectrogram_best.pt --config configs/cnn.yaml --label-encoder data/splits/label_encoder.pkl

echo ""
echo "--- Speaker id10025 ---"
python scripts/infer.py --audio data/raw/voxceleb1/id10025/synthetic/00010.wav --model checkpoints/cnn_mel_spectrogram_best.pt --config configs/cnn.yaml --label-encoder data/splits/label_encoder.pkl

echo ""
echo "--- Speaker id10050 ---"
python scripts/infer.py --audio data/raw/voxceleb1/id10050/synthetic/00000.wav --model checkpoints/cnn_mel_spectrogram_best.pt --config configs/cnn.yaml --label-encoder data/splits/label_encoder.pkl

echo ""
echo "=== Done! Both models should correctly identify all 3 speakers ==="
```

**Note:** GMM and SVM don't use the infer.py script — they were evaluated through evaluate_all.py and their results are in `results/baseline_gmm/metrics.json` and `results/baseline_svm/metrics.json`.
