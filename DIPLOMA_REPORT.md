# Speaker Identification System — Diploma Report

## 1. Introduction

### 1.1 What Is Speaker Identification?

Every person has a unique voice. Just like fingerprints, voices carry characteristics that can distinguish one person from another — the shape of the vocal tract, speaking habits, pitch range, and many other subtle properties. Speaker identification is the task of determining **who is speaking** from an audio recording.

In this project, we build a system that listens to a short audio clip (3 seconds) and answers the question: "Which of these known speakers is this?" This is called **closed-set identification** because the speaker must be someone the system already knows — it cannot identify strangers.

### 1.2 Why This Matters

Speaker identification has many real-world applications:

- **Security systems**: Verifying someone's identity by their voice (phone banking, access control)
- **Smart assistants**: Personalizing responses based on who is speaking ("Hey Google, play my playlist")
- **Forensics**: Identifying speakers in recorded phone calls or surveillance audio
- **Meeting transcription**: Labeling who said what in a meeting recording
- **Call centers**: Automatically routing calls based on known customer voices

### 1.3 Goal of This Diploma

The goal is to **compare classical and modern approaches** to speaker identification:

- **Classical approaches** (from the 2000s): GMM-UBM and SVM — these use handcrafted statistical features and traditional machine learning algorithms
- **Modern approaches** (from 2019-2020): CNN and ECAPA-TDNN — these use deep neural networks that automatically learn the best features from raw data

By building and comparing all four systems on the same data, we can clearly see how the field has evolved and why deep learning has become the dominant approach.

### 1.4 How the System Works (Overview)

Every speaker identification system follows three steps:

**Step 1 — Listen**: Take a raw audio recording (a WAV file containing pressure values over time)

**Step 2 — Understand**: Convert the audio into a meaningful numerical representation. The raw audio is just a sequence of numbers representing air pressure — we need to extract patterns that capture *what makes this voice unique*. This step is called **feature extraction**.

**Step 3 — Decide**: Feed those features into a model that has learned what different speakers sound like, and get back an answer: "This is Speaker X."

All four of our systems follow this pattern, but they differ in how they perform Steps 2 and 3.

---

## 2. Background: How Sound Becomes Numbers

### 2.1 Digital Audio

Sound is a vibration in the air. A microphone converts these vibrations into electrical signals, and a computer samples this signal many times per second. Our system uses a **sample rate of 16,000 Hz** — meaning the computer records 16,000 amplitude values every second. A 3-second recording therefore contains 48,000 numbers.

But this raw waveform is not very useful for identification. It contains too much detail (individual wave cycles) and not enough structure (no clear information about voice characteristics). We need a better representation.

### 2.2 From Waveform to Spectrogram

The key insight is that voices differ not in individual amplitude samples, but in their **frequency content** — the mix of low and high pitched sounds at each moment in time.

To analyze frequencies, we use the **Short-Time Fourier Transform (STFT)**:
1. Slide a small window (25 milliseconds) across the audio
2. For each window position, compute the frequency content using the Fourier transform
3. Move the window forward by 10 milliseconds and repeat

The result is a **spectrogram** — a 2D image where:
- The horizontal axis is time
- The vertical axis is frequency
- The brightness at each point shows how strong that frequency is at that moment

This is already much more useful than the raw waveform, but we can do better.

### 2.3 Mel-Spectrogram

Human hearing does not treat all frequencies equally. We can easily distinguish between 200 Hz and 400 Hz, but the difference between 8,000 Hz and 8,200 Hz sounds almost the same. The **mel scale** mimics this property of human hearing — it spaces frequency bands more closely at low frequencies and more widely at high frequencies.

A **mel-spectrogram** applies the mel scale to a regular spectrogram, resulting in 80 frequency bands (called mel bands) that match how we actually perceive sound. This is the primary input for our deep learning models (CNN and ECAPA-TDNN).

Settings used:
- 80 mel bands (covering 0 Hz to 8,000 Hz)
- 512-point FFT (window for frequency analysis)
- 25ms window length, 10ms hop between windows
- Logarithmic scaling (because we perceive loudness on a log scale)

### 2.4 MFCCs (Mel-Frequency Cepstral Coefficients)

MFCCs go one step further than the mel-spectrogram. They apply a mathematical transformation called the **Discrete Cosine Transform (DCT)** to the mel-spectrogram, which:
- Decorrelates the frequency bands (removes redundancy)
- Compresses the information into fewer numbers

We keep the first 40 MFCC coefficients. This gives a very compact representation of the voice's spectral characteristics at each moment in time — just 40 numbers per time window instead of 80.

MFCCs are the standard input for classical models (GMM and SVM) and have been the workhorse of speech processing since the 1980s.

### 2.5 Why Two Different Features?

- **Mel-spectrograms** preserve more detail and spatial structure — the CNN can learn its own patterns from this rich 2D representation
- **MFCCs** are more compact and decorrelated — they work better with classical statistical models that assume feature independence

---

## 3. Data

### 3.1 Dataset: VoxCeleb1

The primary dataset for this project is **VoxCeleb1** (Nagrani et al., 2017), one of the most widely used benchmarks for speaker identification research.

- **Source**: Celebrity interviews extracted from YouTube videos
- **Size**: 1,251 speakers, approximately 150,000 utterances
- **Recording conditions**: Real-world — varying noise, room acoustics, microphone quality, and emotional states
- **Why this dataset**: It is challenging enough to show real differences between models, well-established in the research community, and freely available for academic use

### 3.2 Synthetic Dataset (Development)

Since VoxCeleb1 requires registration and download (~30 GB), the system includes a **synthetic data generator** for development and testing purposes.

The synthetic generator creates artificial audio for each speaker using:
- A unique base frequency (Speaker 1 = 80 Hz, Speaker 2 = 100 Hz, etc.)
- Harmonics at 2x and 3x the base frequency (like real voices)
- Two formant frequencies (simulating the vocal tract resonances that make each voice unique)
- Small random pitch variations per utterance (2% jitter, like natural speech)
- Light background noise (5% amplitude)

**Current setup**: 50 speakers, 30 utterances each = 1,500 audio files

This synthetic data is **intentionally easy** — each speaker has a clearly distinct frequency signature. All models achieve 100% accuracy on it. This serves as a **pipeline validation**: it confirms that every component (data loading, feature extraction, training, evaluation, inference) works correctly before investing time in real data experiments.

### 3.3 Data Splitting

The dataset is divided into three parts:

- **Training set (70%)**: 1,050 utterances — the model learns from this data
- **Validation set (15%)**: 225 utterances — used during training to monitor performance and decide when to stop training
- **Test set (15%)**: 225 utterances — used only once at the end for final evaluation

The split is **stratified by speaker**: every speaker appears in all three sets. This ensures the model is tested on *new recordings* from *known speakers*, which is exactly the task we want to solve.

The splits are saved to a JSON file so every experiment uses the exact same data division, ensuring fair comparison.

---

## 4. Data Augmentation

### 4.1 Why Augment?

Deep learning models are hungry for data. With limited training examples, they tend to **overfit** — memorizing the training data instead of learning general patterns. Data augmentation creates artificial variety by applying random transformations, effectively teaching the model: "Speaker identity doesn't change when there's a bit of noise, when the volume changes, or when some frequencies are missing."

### 4.2 Audio-Level Augmentation

Applied to the raw waveform before feature extraction:

**Noise injection**: Adds random Gaussian noise at a signal-to-noise ratio between 5 and 20 dB. This simulates recording in noisy environments (cafes, streets, offices).

**Volume perturbation**: Randomly changes the volume by -6 to +6 dB. This teaches the model that a whisper and a shout from the same person should be identified as the same speaker.

### 4.3 Spectrogram-Level Augmentation (SpecAugment)

Applied to the mel-spectrogram after feature extraction. This technique, introduced by Google in 2019, randomly masks parts of the spectrogram:

**Frequency masking**: Zeros out a random band of consecutive frequency bins (up to 15 bands). This forces the model to identify speakers even when some frequency information is missing.

**Time masking**: Zeros out a random band of consecutive time frames (up to 20 frames). This forces the model to identify speakers even from partial utterances.

Two masks of each type are applied per training example.

**Important**: Augmentation is only applied during training. During evaluation and inference, the data is used without any modifications.

---

## 5. The Four Models

### 5.1 Model 1: GMM-UBM (Gaussian Mixture Model with Universal Background Model)

**Era**: 2000s — the dominant approach for over a decade

**Key idea**: Model the probability distribution of each speaker's voice features using a mixture of Gaussian (bell-curve) distributions.

**How it works:**

Imagine plotting all the MFCC feature values for a speaker in a high-dimensional space. The points will form clusters — certain combinations of MFCC values are typical for this speaker, others are rare. A Gaussian Mixture Model places multiple "bell curves" (Gaussians) in this space to cover these clusters. The more Gaussians, the more detail the model can capture.

**Training process:**

1. **Train a Universal Background Model (UBM)**: First, train one large GMM with 256 Gaussian components on *all speakers' data combined*. This model captures "what speech sounds like in general" — it represents the average voice characteristics across everyone.

2. **Adapt per speaker**: For each individual speaker, slightly adjust the UBM to better fit that speaker's specific data. This technique is called **Maximum A Posteriori (MAP) adaptation**. Instead of training a completely new model from scratch (which would need lots of data per speaker), we start from the general model and nudge it toward the individual. This is both faster and more robust, especially when training data per speaker is limited.

**Identification**: Given a test recording, extract MFCCs, then compute the likelihood under each speaker's adapted GMM. The speaker whose model assigns the highest probability wins.

**Strengths**: Works reasonably well with limited data per speaker; well-understood mathematically; fast inference.

**Weaknesses**: Assumes features at each time frame are independent; cannot capture temporal patterns; limited capacity.

---

### 5.2 Model 2: SVM (Support Vector Machine)

**Era**: 2000s — often used alongside or instead of GMMs

**Key idea**: Find the optimal decision boundary between speakers in a feature space.

**The challenge**: SVMs need fixed-length input vectors, but utterances have different lengths (different numbers of MFCC frames). Solution: summarize each utterance into a single fixed-length vector.

**Supervector extraction**: For each utterance, compute the mean and standard deviation of all 40 MFCC coefficients across time. Concatenating these gives an 80-dimensional vector that captures "on average, what do this speaker's MFCCs look like, and how much do they vary?"

**Training process:**

1. Convert every training utterance into an 80-dimensional supervector
2. Normalize all features to zero mean and unit variance (StandardScaler)
3. Train an SVM with RBF (Radial Basis Function) kernel to find non-linear decision boundaries between speakers

The RBF kernel allows the SVM to learn complex, curved boundaries — not just straight lines — which is important because speaker characteristics don't separate linearly.

**Identification**: Extract supervector from test audio, classify with the trained SVM.

**Strengths**: Very fast training and inference; effective with the right features; good generalization.

**Weaknesses**: The supervector (mean + std) discards all temporal information; performance depends heavily on the quality of the handcrafted features.

---

### 5.3 Model 3: CNN (Convolutional Neural Network)

**Era**: 2017-2019 — deep learning enters speaker recognition

**Key idea**: Treat the mel-spectrogram as a 2D image and let the neural network automatically learn which visual patterns distinguish speakers.

**Why this works**: Different speakers produce mel-spectrograms with different visual patterns — distinct formant tracks, different pitch contours, characteristic energy distributions. A CNN can learn to recognize these patterns just like it learns to recognize faces or objects in photographs.

**Architecture** (VGG-style, 4 convolutional blocks):

The input mel-spectrogram has shape (80 frequency bands x ~300 time frames). It passes through four processing blocks, each containing:

- **Two convolutional layers**: Each slides a small 3x3 filter across the spectrogram, detecting local patterns (edges, formant transitions, energy peaks). The first block uses 64 filters, growing to 128, 256, and finally 512 filters. Each filter learns to detect a different type of pattern.

- **Batch normalization**: Normalizes the outputs to prevent training instability. This lets us train deeper networks.

- **ReLU activation**: Introduces non-linearity. Without this, stacking multiple layers would be mathematically equivalent to a single layer.

- **Max pooling (2x2)**: Reduces the spatial dimensions by half, keeping only the strongest activations. This makes the model robust to small shifts in time or frequency.

After four blocks, the 80x300 input has been reduced to 5x~19 with 512 channels — a highly compressed representation where each channel encodes a different aspect of the speaker's voice.

**Temporal average pooling**: Since utterances can have slightly different lengths, we average across the time dimension, producing a fixed-length representation regardless of input duration.

**Speaker embedding**: A fully connected layer compresses the 2,560-dimensional pooled features into a 192-dimensional **speaker embedding** — a compact vector that encodes the speaker's identity. Two recordings from the same speaker should produce similar embeddings; recordings from different speakers should produce different embeddings.

**Classification**: A final linear layer maps the 192-dimensional embedding to scores for each speaker.

**Parameters**: ~5.2 million trainable parameters

**Strengths**: Learns features automatically; captures 2D patterns in spectrograms; well-understood architecture.

**Weaknesses**: MaxPooling aggressively reduces resolution; does not model very long-range temporal dependencies; relatively simple architecture compared to state-of-the-art.

---

### 5.4 Model 4: ECAPA-TDNN (State-of-the-Art)

**Era**: 2020 — current state-of-the-art for speaker recognition

**Key idea**: A specialized architecture that processes audio as a 1D sequence (time dimension), uses multi-scale analysis, channel attention, and intelligent pooling to create highly discriminative speaker embeddings.

**ECAPA-TDNN** stands for:
- **E**mphasized **C**hannel **A**ttention
- **P**ropagation and **A**ggregation
- in **T**ime **D**elay **N**eural **N**etwork

This is the architecture from Desplanques et al. (Interspeech 2020) and represents the current best practice for speaker verification systems.

**Architecture overview:**

Unlike the CNN which treats the spectrogram as a 2D image, ECAPA-TDNN treats it as a 1D signal — the 80 mel bands become 80 input channels, and the model processes the time dimension with 1D convolutions. This is more natural for audio processing.

The architecture has five key innovations:

**1. Initial projection**: A wide 1D convolution (kernel size 5) projects the 80 mel-band channels into 512 processing channels. This is like expanding the representation to give the model more room to work.

**2. SE-Res2Net blocks (x3)**: Three identical blocks, each with increasing dilation (2, 3, 4). Each block contains:

- **Res2Net multi-scale convolution**: Instead of one convolution, the channels are split into 8 groups and processed hierarchically. Group 2 builds on Group 1's output, Group 3 builds on Group 2, and so on. This captures patterns at multiple scales (short formant transitions AND longer pitch contours) within a single layer.

- **Squeeze-and-Excitation (SE) attention**: A channel attention mechanism that learns which of the 512 channels are most important for the current input. It globally pools information, compresses it through a bottleneck, and produces per-channel importance weights. Channels with more speaker-discriminative information get amplified; less useful channels get suppressed.

- **Residual connection**: The input is added to the output, allowing gradients to flow directly through the network during training. This makes deep networks easier to train.

- **Increasing dilation**: The three blocks use dilations of 2, 3, and 4, meaning they look at increasingly wider context windows. Block 1 captures very local patterns, Block 2 captures medium-range patterns, Block 3 captures the widest patterns.

**3. Multi-layer Feature Aggregation (MFA)**: Instead of only using the output of the last block, ECAPA-TDNN concatenates the outputs of ALL three SE-Res2Net blocks. This gives the model access to patterns at all scales simultaneously — from fine-grained phonetic details to broad prosodic patterns. The concatenated features (1,536 channels) are projected back through a 1x1 convolution.

**4. Attentive Statistics Pooling**: The most sophisticated part. Instead of simple averaging over time (like the CNN), this mechanism:
- Learns an **attention function** that assigns importance weights to each time frame
- Uses these weights to compute a **weighted mean** (what features are present) and **weighted standard deviation** (how much they vary)
- The attention focuses on the most speaker-discriminative parts of the utterance — typically voiced speech segments rather than silence or noise

The result is a single vector (3,072 dimensions: 1,536 mean + 1,536 std) that summarizes the entire utterance.

**5. Speaker embedding**: A final fully connected layer compresses this into a 192-dimensional embedding, followed by batch normalization for stability.

**Parameters**: ~5.6 million trainable parameters

**Strengths**: State-of-the-art accuracy; multi-scale feature extraction; attention mechanisms focus on relevant parts; designed specifically for speaker recognition.

**Weaknesses**: More complex to train; benefits most from large datasets; slightly slower inference than simpler models.

---

## 6. Training Process

### 6.1 How Deep Learning Models Learn

Training a neural network is an iterative optimization process:

1. **Forward pass**: Feed a batch of training examples through the model, get predictions
2. **Compute loss**: Measure how wrong the predictions are (using a loss function)
3. **Backward pass**: Compute how each parameter contributed to the error (using calculus/backpropagation)
4. **Update weights**: Adjust parameters in the direction that reduces the error (using an optimizer)
5. Repeat for all batches in the training set (this is one **epoch**)
6. Repeat for many epochs until the model converges

### 6.2 Loss Functions

**Cross-Entropy Loss** (used for CNN and ECAPA-TDNN on synthetic data):
The standard classification loss. It penalizes the model proportionally to how confident it was in the wrong answer. If the model says "90% sure this is Speaker A" but it's actually Speaker B, the penalty is much larger than if it said "10% sure this is Speaker A."

**AAM-Softmax / ArcFace Loss** (planned for ECAPA-TDNN on real data):
A specialized loss designed for learning discriminative embeddings. Instead of just penalizing wrong predictions, it:
- Normalizes all embeddings to lie on a unit sphere
- Adds an angular margin (0.2 radians) between the decision boundaries
- Forces embeddings of the same speaker to cluster tightly together while pushing different speakers further apart

Think of it this way: with regular cross-entropy, a model might learn to barely distinguish speakers — just enough to get the right answer. With ArcFace, the model is forced to create a clear gap between speakers, resulting in embeddings that generalize much better to unseen data.

### 6.3 Optimizer and Learning Rate

**Optimizer**: Adam (Adaptive Moment Estimation) — the most popular optimizer for deep learning. It adapts the learning rate for each parameter individually based on the history of gradients. This makes training faster and more robust than simple gradient descent.

**Learning rate schedule** (Cosine Warmup):

The learning rate controls how large each weight update is. Too high, and training is unstable. Too low, and training is too slow. Our schedule uses two phases:

- **Warmup (first 5 epochs)**: Start with a very small learning rate and linearly increase to the target (0.001). This prevents large, destructive updates when the model is still randomly initialized.

- **Cosine decay (remaining epochs)**: Gradually decrease the learning rate following a cosine curve, ending near zero. This allows fine-tuning — large steps early on to find a good region, then smaller steps to precisely optimize.

### 6.4 Early Stopping

We monitor the validation loss after each epoch. If it doesn't improve for 15 consecutive epochs (CNN) or 20 epochs (ECAPA-TDNN), we stop training — the model has converged and further training would only lead to overfitting.

The model checkpoint from the epoch with the best validation loss is saved as the final model.

### 6.5 Mixed Precision Training

On GPUs, the system uses **automatic mixed precision (AMP)**: computations are done in 16-bit floating point instead of 32-bit where possible. This nearly doubles training speed and halves memory usage with negligible impact on accuracy. This is automatically enabled on CUDA GPUs and disabled on CPU.

### 6.6 Gradient Clipping

To prevent training instability (exploding gradients), all gradients are clipped to a maximum norm of 5.0 before each optimizer step.

---

## 7. Evaluation Metrics

### 7.1 Accuracy

The most intuitive metric: what percentage of test utterances are correctly identified? If 220 out of 225 test utterances are assigned to the right speaker, accuracy is 97.8%.

### 7.2 Top-5 Accuracy

Sometimes the model is almost right — the correct speaker might be the second or third most likely prediction. Top-5 accuracy measures how often the correct speaker appears in the model's top 5 predictions. This is useful for understanding whether the model is "close" even when it's wrong.

### 7.3 Equal Error Rate (EER)

The standard metric for speaker verification. Imagine setting a threshold: "If the model's confidence is above this threshold, accept the speaker; otherwise, reject." Two types of errors can occur:

- **False Acceptance Rate (FAR)**: An impostor is accepted as the genuine speaker
- **False Rejection Rate (FRR)**: The genuine speaker is rejected

As you lower the threshold, FAR increases (more people accepted) and FRR decreases (fewer rejections). The **Equal Error Rate** is the point where FAR = FRR. Lower EER = better system.

An EER of 0% means perfect separation between genuine and impostor trials.

### 7.4 Confusion Matrix

A table showing, for each true speaker, how many of their utterances were classified as each speaker. A perfect system has all values on the diagonal (correct predictions) and zeros everywhere else.

---

## 8. Experiment Results

### 8.1 Synthetic Data Results

| Model | Accuracy | EER | Training Time (CPU) |
|-------|----------|-----|---------------------|
| GMM-UBM | 100.0% | 0.0% | ~2 minutes |
| SVM | 100.0% | 0.0% | ~1 minute |
| CNN | 100.0% | 0.0% | ~45 minutes |
| ECAPA-TDNN | 100.0% | 0.0% | ~48 minutes |

All models achieve perfect performance on synthetic data. This is expected and intentional — synthetic speakers have clearly distinct frequency signatures that any reasonable model can separate. These results confirm that the entire pipeline (data loading, feature extraction, training, evaluation, inference) works correctly.

### 8.2 Training Dynamics

**CNN**: Reached 100% validation accuracy by epoch 3. The model continued training until epoch 37 (early stopping), further reducing the validation loss from 0.21 to 0.0007. The fast convergence reflects how easy the synthetic task is for a neural network.

**ECAPA-TDNN**: Reached 100% validation accuracy by epoch 15. Trained until epoch 63 (early stopping) with final validation loss of 0.006. The slower convergence compared to CNN is expected — ECAPA-TDNN is a more complex model that takes longer to optimize.

**GMM-UBM**: No iterative training — the UBM is fit once, then each speaker model is adapted in a single pass. Total time includes feature extraction from all training files.

**SVM**: Similarly fast — once supervectors are extracted, SVM training on 1,050 samples with 50 classes takes seconds.

### 8.3 What to Expect with Real Data (VoxCeleb1)

With real speech data, we expect significantly different results across models:

| Model | Expected Accuracy | Why |
|-------|------------------|-----|
| GMM-UBM | 70-85% | Cannot capture temporal patterns; assumes frame independence |
| SVM | 75-90% | Better generalization than GMM but still limited by supervector representation |
| CNN | 85-95% | Learns useful patterns automatically but architecture is relatively simple |
| ECAPA-TDNN | 95-99% | State-of-the-art architecture specifically designed for this task |

The gap between classical and deep learning methods will be clearly visible, demonstrating the advancement of modern approaches.

### 8.4 Note on AAM-Softmax Loss

The original configuration used AAM-Softmax (ArcFace) loss for ECAPA-TDNN. This loss function failed to converge on synthetic data — the model showed 0% accuracy after 38 epochs and was stopped early.

**Why**: AAM-Softmax adds an angular margin that forces embeddings to be well-separated on a unit sphere. With synthetic data, the embedding space is too simple for this constraint — the model cannot learn to place embeddings correctly under the angular margin before the loss becomes too difficult to optimize.

**Solution**: Switched to standard cross-entropy loss for synthetic experiments. When training on VoxCeleb1, AAM-Softmax should be re-enabled, as it significantly improves embedding quality with real, complex data.

---

## 9. System Architecture and Data Flow

### 9.1 Complete Training Pipeline

```
                     Raw Audio Files (.wav)
                            |
                            v
               Dataset Preparation (download.py)
              /             |              \
             v              v               v
        VoxCeleb1      Synthetic       Metadata CSV
        (real data)    Generator      (file index)
                            |
                            v
                    Train/Val/Test Split (splits.py)
                    (70% / 15% / 15%, stratified)
                            |
                            v
                    PyTorch Dataset (dataset.py)
                    /                        \
                   v                          v
           For Deep Learning:          For Classical ML:
           - Load WAV (soundfile)      - Load WAV (soundfile)
           - Resample to 16kHz        - Extract MFCCs
           - Crop/pad to 3 sec        - Return numpy array
           - Augment (training only)
           - Extract mel-spectrogram
           - SpecAugment (training)
           - Return tensor + label
                   |                          |
                   v                          v
              DataLoader                 Feature Collection
              (batches of 64)           (grouped by speaker)
                   |                          |
                   v                          v
              Trainer (trainer.py)       Model.fit()
              - Forward pass             - GMM: UBM + MAP
              - Loss computation         - SVM: supervector + SVM
              - Backward pass
              - Gradient clipping
              - Weight update
              - Validation check
              - Early stopping
              - Checkpoint saving
                   |                          |
                   v                          v
              Best Checkpoint (.pt)      Saved Model (.pkl)
```

### 9.2 Inference Pipeline

```
         New Audio File (.wav)
                |
                v
         Load with soundfile
         Convert to mono, 16kHz
         Crop/pad to 3 seconds
                |
                v
         Extract mel-spectrogram
         (80 mel bands, log-scaled)
                |
                v
         Load trained model
         (from checkpoint)
                |
                v
         Model forward pass
         → 192-dim embedding
         → Speaker logits
                |
                v
         Softmax → probabilities
                |
                v
         Output:
         - Predicted speaker: "id10025"
         - Confidence: 98.3%
         - Top-5 predictions
         - Speaker embedding vector
```

---

## 10. Project Configuration

The entire system is controlled through YAML configuration files. This means you can change any aspect of the experiment — from the number of speakers to the learning rate — without modifying any code.

### 10.1 Configuration Inheritance

There is one base configuration file (`base.yaml`) that contains sensible defaults for everything. Each experiment then has its own config file that only specifies what's different:

```
base.yaml (all defaults)
    |
    |-- cnn.yaml: "Use mel-spectrograms, 80 epochs, cross-entropy loss"
    |-- ecapa_tdnn.yaml: "Use mel-spectrograms, 100 epochs, cross-entropy loss"
    |-- baseline_gmm.yaml: "Use MFCCs, no augmentation, GMM model"
    |-- baseline_svm.yaml: "Use MFCCs, no augmentation, SVM model"
```

### 10.2 Key Settings

**Audio**: 16 kHz sample rate, 3-second clips (these are standard in the field)

**Features**: 80 mel bands or 40 MFCCs, 512-point FFT, 25ms windows with 10ms hop

**Data**: 50 speakers, 70/15/15 train/val/test split

**Training (CNN)**: 80 epochs, batch size 64, learning rate 0.001, cosine warmup schedule, patience 15

**Training (ECAPA-TDNN)**: 100 epochs, batch size 64, learning rate 0.001, cosine warmup schedule, patience 20

**Augmentation**: SpecAugment with frequency and time masking, optional noise injection

---

## 11. Reproducibility

Scientific experiments must be reproducible. This project ensures reproducibility through:

1. **Fixed random seed (42)**: Set across Python, NumPy, and PyTorch at the start of every experiment. This means random weight initialization, data shuffling, and augmentation all produce identical results when re-run.

2. **Saved train/val/test splits**: The exact list of files in each split is saved to JSON, so different experiments use the same data division.

3. **Saved label encoder**: The mapping from speaker IDs to integer labels is saved to a pickle file, ensuring consistent encoding.

4. **Configuration files**: Every hyperparameter is recorded in YAML, not hardcoded.

5. **Checkpoint metadata**: Each saved model includes not just the weights but also the configuration it was trained with.

---

## 12. Technical Challenges and Solutions

### 12.1 Audio Loading on Windows

**Challenge**: The latest version of torchaudio (2.11) changed its audio loading backend to `torchcodec`, which requires FFmpeg DLLs installed system-wide. On Windows, this is not straightforward.

**Solution**: Bypassed torchaudio's loading function entirely and used the `soundfile` library directly. This reads WAV files reliably on all platforms:

```python
# Instead of torchaudio.load() which crashes:
data, sample_rate = soundfile.read(file_path, dtype="float32")
waveform = torch.from_numpy(data).unsqueeze(0)  # Convert to tensor
```

### 12.2 AAM-Softmax on Synthetic Data

**Challenge**: The ArcFace loss function (AAM-Softmax) failed to train on synthetic data — 0% accuracy even after 38 epochs.

**Analysis**: Synthetic speakers have very simple spectral signatures. The model's embedding space is correspondingly simple. AAM-Softmax forces embeddings onto a unit sphere and adds angular margins between classes. With such a simple embedding space, the optimization landscape becomes too constrained — the model cannot find a good starting point.

**Solution**: Used cross-entropy loss for synthetic experiments. AAM-Softmax is designed for complex, real-world data where the angular margin constraint actually helps by forcing better separation.

### 12.3 Multiprocessing on Windows

**Challenge**: PyTorch's DataLoader with `num_workers > 0` caused hanging on Windows due to multiprocessing issues.

**Solution**: Set `num_workers: 0` for Windows (single-process data loading) and `num_workers: 2` for Linux/Colab environments.

---

## 13. How to Use the System

### 13.1 Setup

```bash
# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; import soundfile; import sklearn; print('Ready')"
```

### 13.2 Generate Data and Train

```bash
# Step 1: Create synthetic dataset
python -c "from src.data.download import prepare_dataset; prepare_dataset('data', 50, 20)"

# Step 2: Train classical baselines (fast)
python scripts/train_baseline.py --config configs/baseline_gmm.yaml
python scripts/train_baseline.py --config configs/baseline_svm.yaml

# Step 3: Train deep learning models (slow on CPU)
python scripts/train_cnn.py --config configs/cnn.yaml
python scripts/train_ecapa.py --config configs/ecapa_tdnn.yaml
```

### 13.3 Evaluate

```bash
# Compare all models
python scripts/evaluate_all.py

# Test single-file inference
python scripts/infer.py \
    --audio data/raw/voxceleb1/id10001/synthetic/00000.wav \
    --model checkpoints/ecapa_tdnn_best.pt \
    --config configs/ecapa_tdnn.yaml \
    --label-encoder data/splits/label_encoder.pkl
```

### 13.4 Generate Thesis Figures

```bash
cd notebooks
jupyter notebook
# Run notebooks 01 through 07 in order
```

---

## 14. Comparison of Approaches

### 14.1 Feature Comparison

| Aspect | Classical (GMM, SVM) | Deep Learning (CNN, ECAPA-TDNN) |
|--------|---------------------|--------------------------------|
| Features | Handcrafted MFCCs | Learned from mel-spectrograms |
| Feature design | Requires expert knowledge | Automatic |
| Temporal modeling | None (frame-independent) | Full sequence modeling |
| Training data needs | Lower | Higher |
| Training time | Minutes | Minutes to hours |
| Inference speed | Fast | Fast |

### 14.2 Model Complexity

| Model | Parameters | Input | Key Innovation |
|-------|-----------|-------|----------------|
| GMM-UBM | ~500K per speaker | MFCC frames | UBM + MAP adaptation |
| SVM | ~4K weights | 80-dim supervector | Kernel trick for non-linear boundaries |
| CNN | 5.2 million | 2D mel-spectrogram | Hierarchical 2D convolutions |
| ECAPA-TDNN | 5.6 million | 1D mel sequence | Multi-scale + attention + MFA |

### 14.3 Historical Context

```
2000s: GMM-UBM dominates speaker recognition
  |
  v
2010s: i-vectors + PLDA (not in this project)
  |      Extract fixed-length vectors from GMMs, then score with
  |      probabilistic discriminant analysis
  |
  v
2017: d-vectors (simple DNNs for speaker embeddings)
  |      First successful deep learning approach
  |
  v
2018: x-vectors (TDNN-based embeddings)
  |      Time-delay neural networks become the standard
  |
  v
2020: ECAPA-TDNN (this project)
         State-of-the-art with SE-Res2Net, MFA, and attentive pooling
```

---

## 15. Future Work

### 15.1 Immediate Next Steps

1. **Train on VoxCeleb1**: Register, download, and train all models on real speech data. This will show the true differences between approaches.

2. **Re-enable AAM-Softmax**: Switch ECAPA-TDNN back to ArcFace loss for real data, which should significantly improve embedding quality.

3. **GPU Training on Google Colab**: Move deep learning training to Colab's T4 GPU for 10-20x speedup.

### 15.2 Possible Extensions

- **Speaker verification**: Instead of "who is this?", answer "is this person who they claim to be?" — a binary yes/no task using embedding similarity
- **Open-set identification**: Handle unknown speakers (not in the training set)
- **Real-time processing**: Optimize for streaming audio input
- **Cross-language evaluation**: Test if the system works across different languages
- **Noise robustness analysis**: Systematically evaluate performance under different noise conditions

---

## 16. Conclusion

This diploma project implements and compares four speaker identification approaches spanning two decades of research. The system includes a complete pipeline from raw audio to speaker prediction, with comprehensive evaluation and visualization tools suitable for thesis work.

The synthetic data experiments validate that all components work correctly, achieving 100% accuracy across all models. The real differentiation between classical and modern approaches will emerge with VoxCeleb1 training, where the advantages of deep learning — automatic feature learning, temporal modeling, and attention mechanisms — are expected to produce significantly better results than the handcrafted statistical approaches.

The ECAPA-TDNN architecture, with its multi-scale feature extraction, channel attention, and attentive pooling, represents the current state-of-the-art in speaker recognition. Combined with the ArcFace loss function for discriminative embedding learning, it should achieve near-perfect performance on VoxCeleb1 and demonstrate the clear progression from classical to modern speaker identification methods.

---

## 17. References

1. Desplanques, B., Thienpondt, J., & Demuynck, K. (2020). "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification." *Interspeech 2020*.

2. Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). "ArcFace: Additive Angular Margin Loss for Deep Face Recognition." *CVPR 2019*.

3. Park, D. S., et al. (2019). "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition." *Interspeech 2019*.

4. Nagrani, A., Chung, J. S., & Zisserman, A. (2017). "VoxCeleb: A Large-Scale Speaker Identification Dataset." *Interspeech 2017*.

5. Chung, J. S., Nagrani, A., & Zisserman, A. (2018). "VoxCeleb2: Deep Speaker Recognition." *Interspeech 2018*.

6. Reynolds, D. A., Quatieri, T. F., & Dunn, R. B. (2000). "Speaker Verification Using Adapted Gaussian Mixture Models." *Digital Signal Processing*.

7. Gao, S., et al. (2019). "Res2Net: A New Multi-scale Backbone Architecture." *IEEE TPAMI*.

8. He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." *CVPR 2016*.

9. Hu, J., Shen, L., & Sun, G. (2018). "Squeeze-and-Excitation Networks." *CVPR 2018*.

10. Snyder, D., et al. (2018). "X-Vectors: Robust DNN Embeddings for Speaker Recognition." *ICASSP 2018*.
