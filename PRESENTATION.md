# Speaker Identification — Project Presentation

---

## What Is This Project About?

This project is about **recognizing people by their voice**. We built a system that takes a short audio recording — just 3 seconds — and determines which person is speaking. Think of it like a fingerprint scanner, but for voices.

The scientific goal is to **compare old and new approaches**. We implemented 4 different models: two classical methods from the 2000s and two modern deep learning methods from 2020. By running them all on the same data, we can clearly see how the technology has evolved and why modern approaches are better.

---

## How Does Voice Identification Work?

Every person's voice is unique. The shape of your throat, mouth, and nasal cavities creates a unique "filter" that shapes the sound you produce. Even if two people say the exact same word, their voices will have different frequency patterns, different pitch ranges, and different characteristics.

Our system works in three steps:

**Step 1 — Record**: We start with a raw audio file. It's just numbers — 16,000 numbers per second representing the pressure waves captured by the microphone.

**Step 2 — Extract Features**: Raw audio is too noisy and too detailed to work with directly. We convert it into a **spectrogram** — a visual representation that shows which frequencies are present at each moment in time. It looks like a heatmap: time goes left to right, frequency goes bottom to top, and brightness shows intensity. Different speakers create different patterns in this heatmap.

**Step 3 — Identify**: We feed the spectrogram into a trained model. The model has previously learned what each speaker's spectrogram patterns look like. It compares the input against all known speakers and returns an answer: "This is Speaker X, and I am 95% confident."

---

## What Did We Build?

### The Complete System

We didn't just build 4 models — we built everything around them too:

**Data pipeline** — A system that takes raw audio files, converts them to spectrograms, splits them into training and testing sets, and applies augmentation (adding noise, masking parts of the spectrogram) to make the models more robust. This is 5 files working together in `src/data/`.

**4 speaker identification models:**

1. **GMM-UBM** (Gaussian Mixture Model) — The old-school approach from the 2000s. It builds a statistical model of "what speech sounds like in general" and then slightly adjusts it for each individual speaker. When it hears new audio, it checks which speaker's model fits best. It's simple, fast, and was the standard for over a decade.

2. **SVM** (Support Vector Machine) — Another classical approach. It takes each recording and compresses it into just 80 numbers (the average characteristics of the voice). Then it draws decision boundaries between speakers. Fast and effective for simple cases, but it throws away a lot of information during the compression step.

3. **CNN** (Convolutional Neural Network) — A deep learning approach. It treats the spectrogram like a photograph and uses the same technology that powers image recognition. It scans the spectrogram with small filters that detect patterns — edges, shapes, energy distributions — and builds up from simple patterns to complex ones through 4 layers. It produces a 192-number "voice fingerprint" for each speaker.

4. **ECAPA-TDNN** — The state-of-the-art from 2020. This is the most sophisticated model. It has three key innovations. First, it analyzes the audio at multiple zoom levels simultaneously — capturing both fine details (individual sounds) and broad patterns (speaking rhythm). Second, it has an attention mechanism that learns to focus on the most informative parts of the audio and ignore silence or noise. Third, it combines information from all its processing layers, not just the last one, giving it the richest possible understanding of the voice. It also produces a 192-number voice fingerprint, but a much higher quality one.

**Training infrastructure** — The system that teaches the models. For deep learning, this means repeatedly showing examples, measuring mistakes, and adjusting the model's internal numbers. It includes learning rate scheduling (controlling how fast the model adjusts), early stopping (detecting when the model has finished learning), gradient clipping (preventing training instability), and checkpointing (saving the best version of the model).

**Evaluation system** — Computes accuracy (what percentage of test recordings are correctly identified), Equal Error Rate or EER (the standard metric in speaker recognition — the point where false accepts equal false rejections), confusion matrices (a table showing which speakers get confused with each other), and generates comparison charts.

**Inference pipeline** — The "use it in practice" part. Give it any audio file and a trained model, and it tells you who is speaking with a confidence score and a top-5 list of most likely speakers.

### The Configuration System

Everything in the system is controlled through simple text files (YAML format). Want to change the number of speakers from 50 to 200? Change one number. Want to try a different learning rate? Change one number. Want to switch from mel-spectrograms to MFCCs? Change one word. No code changes needed for any experiment.

---

## What Did We Test So Far?

### Synthetic Data Experiment

Since the real dataset (VoxCeleb1 — recordings of celebrities from YouTube) is 30 GB and requires registration, we first tested everything with **synthetic data** — artificially generated audio.

We created 50 fake speakers, each with a unique frequency signature. Speaker 1 has a base frequency of 80 Hz, Speaker 2 has 100 Hz, Speaker 3 has 120 Hz, and so on. Each speaker also has unique harmonics and formant patterns, plus small random variations per recording to simulate natural speech variation.

This produced 1,500 audio files (50 speakers x 30 recordings each), split into:
- 1,050 for training (70%)
- 225 for validation (15%)
- 225 for testing (15%)

### Results

| Model | Accuracy | Training Time |
|-------|----------|---------------|
| GMM-UBM | 100% | 2 minutes |
| SVM | 100% | 1 minute |
| CNN | 100% | 45 minutes |
| ECAPA-TDNN | 100% | 48 minutes |

All models achieved perfect accuracy. This is intentional and expected — synthetic speakers have very distinct frequency patterns that any model can easily separate. The purpose of this experiment was not to compare the models (they're all perfect here), but to **verify that the entire system works correctly** from end to end: data loading, feature extraction, training, evaluation, and inference all function without errors.

We also successfully tested the inference pipeline — giving the trained models audio files and confirming they correctly identify the speaker with 100% confidence.

### An Interesting Problem We Encountered

The ECAPA-TDNN model was originally configured with a special loss function called **AAM-Softmax (ArcFace)** — a state-of-the-art technique that forces the model to create very well-separated speaker embeddings by adding angular margins on a unit sphere.

This loss function completely failed on synthetic data — 0% accuracy after 38 epochs. The reason: synthetic voices are so simple that the model's internal representation is also very simple, and the angular margin constraint becomes impossible to satisfy. It's like trying to park a truck in a parking spot designed for a motorcycle — the constraint is too tight for the space.

We switched to standard cross-entropy loss, and the model immediately worked. For real data (VoxCeleb1), AAM-Softmax will be re-enabled — it is designed for complex, real-world voices where the angular margin actually helps by forcing better separation between similar-sounding speakers.

---

## What Do We Still Need To Do?

### Phase 1: Download Real Data

We have received access to **VoxCeleb1** — a dataset of celebrity voice recordings extracted from YouTube interviews. It contains over 1,000 speakers and about 150,000 recordings in real-world conditions: varying noise, different microphones, different rooms, different emotions.

This is the standard benchmark dataset used in speaker recognition research worldwide. We received the download links and are in the process of downloading the files (about 30 GB total, split into 4 parts).

### Phase 2: Train on Real Data Using Google Colab

Deep learning models need a GPU (graphics processing unit) for fast training. Our laptop only has a CPU, which makes training very slow. Google Colab provides a free GPU (NVIDIA T4 with 16 GB of memory) that will speed up training by 10-20 times.

We need to:
- Upload the project and dataset to Google Drive
- Create a Colab notebook that connects everything
- Train all 4 models with GPU acceleration
- Save the results back to Drive

### Phase 3: Get Real Results

This is the most important phase — where we actually see the differences between models.

On real data, we expect:

| Model | Expected Accuracy | Why |
|-------|-------------------|-----|
| GMM-UBM | 70-85% | Cannot capture how voice patterns change over time. Treats each moment independently — misses the rhythm and flow of speech. |
| SVM | 75-90% | Compresses each recording into just 80 numbers, throwing away most of the information. Better than GMM at finding boundaries, but limited by its input. |
| CNN | 85-95% | Learns useful patterns automatically from spectrograms. Much better than handcrafted features. But its architecture is relatively simple — it reduces resolution aggressively and doesn't focus on what matters most. |
| ECAPA-TDNN | 95-99% | The full package: multi-scale analysis captures patterns at all levels, attention focuses on the most informative parts, and multi-layer aggregation combines information from all processing stages. This is why it's the state-of-the-art. |

The gap between 70% (GMM) and 98% (ECAPA-TDNN) will clearly demonstrate how much the field has improved over 20 years.

We will also re-enable the AAM-Softmax loss for ECAPA-TDNN — on real data with complex voice patterns, this angular margin loss will produce even more discriminative embeddings.

### Phase 4: Generate Thesis Figures

We have 7 Jupyter notebooks ready to produce publication-quality figures:

- **Dataset analysis**: How many recordings per speaker, duration distribution, example waveforms
- **Feature comparison**: What mel-spectrograms and MFCCs look like, how SpecAugment modifies them
- **Training curves**: Loss and accuracy over training epochs — showing how each model learns
- **Confusion matrices**: Heatmaps showing which speakers get confused with each other — revealing the models' weaknesses
- **Model comparison chart**: Side-by-side accuracy and EER bars for all 4 models — the key result figure
- **Embedding visualization**: t-SNE plots that project the 192-dimensional voice fingerprints into 2D scatter plots. Each dot is a recording, each color is a speaker. Good models create tight, well-separated clusters. This visually proves that ECAPA-TDNN creates better speaker representations than CNN.

### Phase 5: Write the Thesis

With all results and figures ready, the thesis follows this structure:

1. **Introduction** — What is speaker identification, why it matters, what we're comparing
2. **Literature Review** — The history from GMM-UBM (2000s) through x-vectors to ECAPA-TDNN (2020)
3. **Methodology** — How we extract features, how each model works, how we train and evaluate
4. **Experimental Setup** — The VoxCeleb1 dataset, our specific configuration choices, evaluation metrics
5. **Results** — Accuracy tables, training curves, confusion matrices, EER comparison, t-SNE visualizations
6. **Discussion** — Why deep learning wins, what ECAPA-TDNN does better, when classical methods are still useful
7. **Conclusion** — Summary of findings, contribution, future work

---

## Summary

| Phase | Status | What |
|-------|--------|------|
| Build the complete system | DONE | 4 models, data pipeline, training, evaluation, inference — all working |
| Test on synthetic data | DONE | All models trained and evaluated, 100% accuracy, pipeline validated |
| Fix technical issues | DONE | 5 bugs found and resolved |
| Get VoxCeleb1 access | DONE | Download links received |
| Download VoxCeleb1 | IN PROGRESS | 30 GB, downloading in 4 parts |
| Set up Google Colab | TODO | Create notebook for GPU training |
| Train on real data | TODO | The main experiment — will show real model differences |
| Generate thesis figures | TODO | 7 notebooks ready, waiting for real data results |
| Write thesis | TODO | Structure planned, waiting for results and figures |
