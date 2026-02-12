# Image Captioning with Seq2Seq Architecture

## Project Overview

This project implements an **Image Captioning System** using a **Sequence-to-Sequence (Seq2Seq)** architecture with **LSTM** (Long Short-Term Memory) networks. The system generates natural language descriptions for images by combining computer vision (CNN-based feature extraction) and natural language processing (RNN-based caption generation).

## Architecture Components

The architecture consists of four main components:

1. **Feature Extraction Pipeline** (ResNet50)
2. **Vocabulary & Text Preprocessing**
3. **Seq2Seq Model** (Encoder-Decoder)
4. **Training & Inference**

---

## 1. Feature Extraction Pipeline

### Purpose
Extract visual features from images using a pre-trained CNN to avoid expensive joint CNN-RNN training.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input Image (224x224x3)                                    │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────┐                                       │
│  │   ResNet50       │  Pre-trained on ImageNet             │
│  │   (Backbone)     │  Weights: DEFAULT                     │
│  └────────┬─────────┘                                       │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                       │
│  │  Remove Final    │  Remove classification layer          │
│  │  FC Layer        │  Output: (batch, 2048, 1, 1)         │
│  └────────┬─────────┘                                       │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                       │
│  │   Flatten        │  Reshape to (batch, 2048)            │
│  └────────┬─────────┘                                       │
│           │                                                  │
│           ▼                                                  │
│  Feature Vector (2048-dim)                                  │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────┐                                       │
│  │  Save to Disk    │  Cached as .pkl file                 │
│  │  (Pickle)        │  {image_name: feature_vector}        │
│  └──────────────────┘                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Details

- **Model**: ResNet50 (pre-trained on ImageNet)
- **Input**: RGB images resized to 224×224
- **Output**: 2048-dimensional feature vector per image
- **Preprocessing**:
  - Resize to (224, 224)
  - Convert to Tensor
  - Normalize: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
- **Caching**: Features saved to `flickr30k_features.pkl` for reuse

### Why Cache Features?

Training a CNN and RNN together is computationally expensive. By pre-extracting and caching image features, we:
- Reduce training time significantly
- Enable faster experimentation with different decoder architectures
- Avoid redundant forward passes through the CNN

---

## 2. Vocabulary & Text Preprocessing

### Purpose
Convert raw text captions into numerical sequences that can be processed by the neural network.

### Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│              TEXT PREPROCESSING PIPELINE                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Raw Caption: "A dog playing in the park."                  │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────┐                                       │
│  │  Text Cleaning   │  - Lowercase                          │
│  │                  │  - Remove punctuation                 │
│  │                  │  - Strip whitespace                   │
│  └────────┬─────────┘                                       │
│           │                                                  │
│           ▼                                                  │
│  Cleaned: "a dog playing in the park"                       │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────┐                                       │
│  │  Add Special     │  <start> a dog playing in the        │
│  │  Tokens          │  park <end>                           │
│  └────────┬─────────┘                                       │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                       │
│  │  Build Vocab     │  - Count word frequencies             │
│  │                  │  - Filter by MIN_FREQ=5               │
│  │                  │  - Add special tokens                 │
│  └────────┬─────────┘                                       │
│           │                                                  │
│           ▼                                                  │
│  Vocabulary: {<pad>:0, <start>:1, <end>:2, <unk>:3, ...}   │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────┐                                       │
│  │  Tokenization    │  Convert words to indices             │
│  └────────┬─────────┘                                       │
│           │                                                  │
│           ▼                                                  │
│  Sequence: [1, 45, 123, 67, 12, 8, 234, 2]                 │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────┐                                       │
│  │  Save to Disk    │  - flickr30k_vocab.pkl               │
│  │                  │  - flickr30k_captions.pkl            │
│  └──────────────────┘                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Special Tokens

| Token | Index | Purpose |
|-------|-------|---------|
| `<pad>` | 0 | Padding shorter sequences |
| `<start>` | 1 | Mark beginning of caption |
| `<end>` | 2 | Mark end of caption |
| `<unk>` | 3 | Unknown/rare words |

### Vocabulary Statistics

- **Total vocabulary size**: 7,727 words
- **Minimum frequency**: 5 occurrences
- **Maximum caption length**: 80 tokens
- **Total images**: 31,783

---

## 3. Seq2Seq Architecture

### Overview

The Seq2Seq model consists of two main components:
1. **Encoder**: Transforms image features into initial hidden state
2. **Decoder**: Generates caption word-by-word using LSTM

### Complete Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SEQ2SEQ ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                          ENCODER                                    │    │
│  ├────────────────────────────────────────────────────────────────────┤    │
│  │                                                                     │    │
│  │  Image Feature (2048-dim)                                          │    │
│  │         │                                                           │    │
│  │         ▼                                                           │    │
│  │  ┌──────────────┐                                                  │    │
│  │  │ Linear Layer │  Input: 2048 → Output: 512                       │    │
│  │  │  (fc)        │  Trainable weights                               │    │
│  │  └──────┬───────┘                                                  │    │
│  │         │                                                           │    │
│  │         ▼                                                           │    │
│  │  ┌──────────────┐                                                  │    │
│  │  │    ReLU      │  Non-linear activation                           │    │
│  │  └──────┬───────┘                                                  │    │
│  │         │                                                           │    │
│  │         ▼                                                           │    │
│  │  ┌──────────────┐                                                  │    │
│  │  │   Dropout    │  p=0.5 (regularization)                          │    │
│  │  │   (0.5)      │                                                  │    │
│  │  └──────┬───────┘                                                  │    │
│  │         │                                                           │    │
│  │         ▼                                                           │    │
│  │  Hidden State (512-dim)                                            │    │
│  │                                                                     │    │
│  └─────────────────────────┬───────────────────────────────────────────┘    │
│                            │                                                │
│                            │ Initialize LSTM                                │
│                            ▼                                                │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                          DECODER                                    │    │
│  ├────────────────────────────────────────────────────────────────────┤    │
│  │                                                                     │    │
│  │  Input: Caption Tokens [<start>, w1, w2, ..., wn]                  │    │
│  │         │                                                           │    │
│  │         ▼                                                           │    │
│  │  ┌──────────────┐                                                  │    │
│  │  │  Embedding   │  vocab_size → embed_size (256)                   │    │
│  │  │   Layer      │  Learnable word embeddings                       │    │
│  │  └──────┬───────┘                                                  │    │
│  │         │                                                           │    │
│  │         ▼                                                           │    │
│  │  ┌──────────────┐                                                  │    │
│  │  │   Dropout    │  p=0.5                                           │    │
│  │  └──────┬───────┘                                                  │    │
│  │         │                                                           │    │
│  │         ▼                                                           │    │
│  │  ┌──────────────────────────────────────────┐                      │    │
│  │  │           LSTM Cell                      │                      │    │
│  │  │  ┌────────────────────────────────────┐  │                      │    │
│  │  │  │  Input: embeddings (batch, seq, 256)│ │                      │    │
│  │  │  │  Hidden: h_t (num_layers, batch, 512)│ │                      │    │
│  │  │  │  Cell: c_t (num_layers, batch, 512)  │ │                      │    │
│  │  │  │                                      │ │                      │    │
│  │  │  │  ┌─────────────────────────────┐    │ │                      │    │
│  │  │  │  │  Forget Gate (f_t)          │    │ │                      │    │
│  │  │  │  │  f_t = σ(W_f·[h_{t-1}, x_t])│    │ │                      │    │
│  │  │  │  └─────────────────────────────┘    │ │                      │    │
│  │  │  │  ┌─────────────────────────────┐    │ │                      │    │
│  │  │  │  │  Input Gate (i_t)           │    │ │                      │    │
│  │  │  │  │  i_t = σ(W_i·[h_{t-1}, x_t])│    │ │                      │    │
│  │  │  │  └─────────────────────────────┘    │ │                      │    │
│  │  │  │  ┌─────────────────────────────┐    │ │                      │    │
│  │  │  │  │  Cell Candidate (c̃_t)       │    │ │                      │    │
│  │  │  │  │  c̃_t = tanh(W_c·[h_{t-1}, x_t])│ │                      │    │
│  │  │  │  └─────────────────────────────┘    │ │                      │    │
│  │  │  │  ┌─────────────────────────────┐    │ │                      │    │
│  │  │  │  │  Cell State Update          │    │ │                      │    │
│  │  │  │  │  c_t = f_t * c_{t-1} + i_t * c̃_t│ │                      │    │
│  │  │  │  └─────────────────────────────┘    │ │                      │    │
│  │  │  │  ┌─────────────────────────────┐    │ │                      │    │
│  │  │  │  │  Output Gate (o_t)          │    │ │                      │    │
│  │  │  │  │  o_t = σ(W_o·[h_{t-1}, x_t])│    │ │                      │    │
│  │  │  │  └─────────────────────────────┘    │ │                      │    │
│  │  │  │  ┌─────────────────────────────┐    │ │                      │    │
│  │  │  │  │  Hidden State Update        │    │ │                      │    │
│  │  │  │  │  h_t = o_t * tanh(c_t)      │    │ │                      │    │
│  │  │  │  └─────────────────────────────┘    │ │                      │    │
│  │  │  └────────────────────────────────────┘  │                      │    │
│  │  │                                          │                      │    │
│  │  │  Output: (batch, seq, 512)              │                      │    │
│  │  └──────────────────────────────────────────┘                      │    │
│  │         │                                                           │    │
│  │         ▼                                                           │    │
│  │  ┌──────────────┐                                                  │    │
│  │  │ Linear Layer │  512 → vocab_size (7727)                         │    │
│  │  │     (fc)     │  Project to vocabulary                           │    │
│  │  └──────┬───────┘                                                  │    │
│  │         │                                                           │    │
│  │         ▼                                                           │    │
│  │  Logits (batch, seq, vocab_size)                                   │    │
│  │         │                                                           │    │
│  │         ▼                                                           │    │
│  │  ┌──────────────┐                                                  │    │
│  │  │   Softmax    │  Convert to probabilities                        │    │
│  │  └──────┬───────┘                                                  │    │
│  │         │                                                           │    │
│  │         ▼                                                           │    │
│  │  Predicted Word Probabilities                                      │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `EMBED_SIZE` | 256 | Word embedding dimension |
| `HIDDEN_SIZE` | 512 | LSTM hidden state size |
| `NUM_LAYERS` | 1 | Number of LSTM layers |
| `VOCAB_SIZE` | 7,727 | Total vocabulary size |
| `DROPOUT` | 0.5 | Dropout probability |

### Encoder Details

**Input**: Pre-extracted image feature vector (2048-dim)

**Architecture**:
```python
Linear(2048 → 512) → ReLU → Dropout(0.5)
```

**Output**: Hidden state vector (512-dim) used to initialize LSTM

**Purpose**: Transform high-dimensional image features into a compact representation suitable for initializing the decoder's hidden state.

### Decoder Details

**Input**: 
- Word embeddings of caption tokens
- Initial hidden state from encoder

**Architecture**:
```python
Embedding(vocab_size, 256) → Dropout(0.5) → LSTM(256, 512) → Linear(512, vocab_size)
```

**LSTM Mechanism**:

The LSTM maintains two states:
1. **Hidden state (h_t)**: Short-term memory
2. **Cell state (c_t)**: Long-term memory

At each timestep t:
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  # Input gate
c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)  # Cell candidate
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t  # Cell state update
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  # Output gate
h_t = o_t ⊙ tanh(c_t)  # Hidden state update
```

Where:
- σ = sigmoid function
- ⊙ = element-wise multiplication
- W, b = learnable parameters

---

## 4. Training & Inference

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐                                       │
│  │  Load Cached     │  - Image features (.pkl)              │
│  │  Data            │  - Caption sequences (.pkl)           │
│  │                  │  - Vocabulary (.pkl)                  │
│  └────────┬─────────┘                                       │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                       │
│  │  Train/Val Split │  90% train / 10% validation           │
│  └────────┬─────────┘                                       │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                       │
│  │  DataLoader      │  Batch size: 64                       │
│  │                  │  Shuffle: True (train)                │
│  │                  │  Collate: Pad sequences               │
│  └────────┬─────────┘                                       │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────────────────────────────────────┐        │
│  │           TRAINING LOOP (25 epochs)             │        │
│  │  ┌───────────────────────────────────────────┐  │        │
│  │  │  For each batch:                          │  │        │
│  │  │  1. features, captions → GPU             │  │        │
│  │  │  2. encoder_out = encoder(features)      │  │        │
│  │  │  3. hidden, cell = init_hidden(encoder_out)│ │        │
│  │  │  4. inputs = captions[:, :-1]            │  │        │
│  │  │  5. targets = captions[:, 1:]            │  │        │
│  │  │  6. logits = decoder(inputs, hidden, cell)│ │        │
│  │  │  7. loss = CrossEntropy(logits, targets) │  │        │
│  │  │  8. optimizer.zero_grad()                │  │        │
│  │  │  9. loss.backward()                      │  │        │
│  │  │  10. optimizer.step()                    │  │        │
│  │  └───────────────────────────────────────────┘  │        │
│  └─────────────────────────────────────────────────┘        │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                       │
│  │  Validation      │  Compute val loss (no gradient)       │
│  └────────┬─────────┘                                       │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                       │
│  │  Log Metrics     │  - Train loss                         │
│  │                  │  - Validation loss                    │
│  └──────────────────┘                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam |
| **Learning Rate** | 3e-4 |
| **Batch Size** | 64 |
| **Epochs** | 25 |
| **Loss Function** | CrossEntropyLoss |
| **Ignore Index** | 0 (padding token) |
| **Device** | CUDA (if available) |

### Teacher Forcing

During training, the model uses **teacher forcing**:
- **Input**: Ground truth tokens `[<start>, w1, w2, ..., wn]`
- **Target**: Shifted ground truth `[w1, w2, ..., wn, <end>]`

This helps the model learn faster by providing correct context at each step.

### Loss Function

```python
CrossEntropyLoss(ignore_index=PAD_IDX)
```

- Computes cross-entropy between predicted and target tokens
- Ignores padding tokens (index 0) in loss calculation
- Applied to flattened sequences: `(batch * seq_len, vocab_size)`

---

## 5. Inference Strategies

### Greedy Search

**Algorithm**:
```
1. Start with <start> token
2. For each timestep:
   a. Get model predictions
   b. Select word with highest probability
   c. Use as input for next step
3. Stop when <end> token is generated or max_len reached
```

**Diagram**:
```
┌─────────────────────────────────────────────────────────────┐
│                      GREEDY SEARCH                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Image Feature                                               │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────┐                                                │
│  │ Encoder │ → Hidden State                                 │
│  └─────────┘                                                │
│       │                                                      │
│       ▼                                                      │
│  [<start>] ──────┐                                          │
│       │          │                                          │
│       ▼          ▼                                          │
│  ┌─────────────────┐                                        │
│  │  LSTM Decoder   │                                        │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  [P(w1), P(w2), ..., P(wn)]                                 │
│           │                                                  │
│           ▼                                                  │
│  argmax → "a"  ──────┐                                      │
│           │          │                                      │
│           ▼          ▼                                      │
│  ┌─────────────────┐                                        │
│  │  LSTM Decoder   │                                        │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  argmax → "dog"  ──────┐                                    │
│           │            │                                    │
│           ▼            ▼                                    │
│       ... (continue until <end>)                            │
│                                                              │
│  Final Caption: "a dog playing in the park"                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Pros**:
- Fast (single forward pass per token)
- Deterministic output

**Cons**:
- May miss better overall sequences
- No backtracking if early mistake

### Beam Search

**Algorithm**:
```
1. Maintain top-k (beam_width) hypotheses
2. For each timestep:
   a. Expand each hypothesis with all possible next words
   b. Score each candidate: log P(sequence)
   c. Keep top-k candidates
3. Return highest-scoring complete sequence
```

**Diagram**:
```
┌─────────────────────────────────────────────────────────────┐
│                      BEAM SEARCH (k=3)                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Image Feature                                               │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────┐                                                │
│  │ Encoder │ → Hidden State                                 │
│  └─────────┘                                                │
│       │                                                      │
│       ▼                                                      │
│  [<start>]                                                  │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────┐                                        │
│  │  LSTM Decoder   │                                        │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  Top-3 words: ["a", "the", "an"]                            │
│       │         │         │                                 │
│       ▼         ▼         ▼                                 │
│  ┌─────┐   ┌─────┐   ┌─────┐                               │
│  │  a  │   │ the │   │ an  │  (3 beams)                    │
│  └──┬──┘   └──┬──┘   └──┬──┘                               │
│     │         │         │                                   │
│     ▼         ▼         ▼                                   │
│  ┌─────────────────┐                                        │
│  │  LSTM Decoder   │  (3 parallel forward passes)          │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  9 candidates (3 beams × top-3 words each)                  │
│  Score each: log P(w1) + log P(w2|w1)                       │
│           │                                                  │
│           ▼                                                  │
│  Keep top-3: ["a dog", "a cat", "the dog"]                  │
│       │         │         │                                 │
│       ▼         ▼         ▼                                 │
│  ... (continue until all beams end with <end>)              │
│                                                              │
│  Final: Select beam with highest normalized score           │
│  Score = log P(sequence) / length                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Scoring**:
```python
score = sum(log P(w_i | w_1, ..., w_{i-1})) / length
```

Length normalization prevents bias toward shorter sequences.

**Pros**:
- Better quality captions
- Explores multiple hypotheses

**Cons**:
- Slower (k forward passes per token)
- More memory intensive

---

## 6. Data Flow

### Complete End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE DATA FLOW                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TRAINING PHASE                                                              │
│  ──────────────                                                              │
│                                                                              │
│  1. Image (JPG)                                                              │
│       │                                                                      │
│       ▼                                                                      │
│  2. Resize & Normalize → (224, 224, 3)                                      │
│       │                                                                      │
│       ▼                                                                      │
│  3. ResNet50 → Feature Vector (2048)                                        │
│       │                                                                      │
│       ▼                                                                      │
│  4. Cache to Disk → flickr30k_features.pkl                                  │
│                                                                              │
│  5. Raw Caption: "A dog playing in the park."                               │
│       │                                                                      │
│       ▼                                                                      │
│  6. Clean & Tokenize → [1, 45, 123, 67, 12, 8, 234, 2]                     │
│       │                                                                      │
│       ▼                                                                      │
│  7. Cache to Disk → flickr30k_captions.pkl                                  │
│                                                                              │
│  8. Load Cached Data                                                         │
│       │                                                                      │
│       ▼                                                                      │
│  9. Create DataLoader (batch_size=64)                                       │
│       │                                                                      │
│       ▼                                                                      │
│  10. Training Loop:                                                          │
│      ┌────────────────────────────────────────────────────┐                 │
│      │  Batch: (features, captions)                       │                 │
│      │    │                                               │                 │
│      │    ▼                                               │                 │
│      │  Encoder(features) → hidden_state (512)           │                 │
│      │    │                                               │                 │
│      │    ▼                                               │                 │
│      │  Decoder(captions, hidden) → logits (vocab_size)  │                 │
│      │    │                                               │                 │
│      │    ▼                                               │                 │
│      │  Loss = CrossEntropy(logits, targets)             │                 │
│      │    │                                               │                 │
│      │    ▼                                               │                 │
│      │  Backpropagation & Update Weights                 │                 │
│      └────────────────────────────────────────────────────┘                 │
│                                                                              │
│  INFERENCE PHASE                                                             │
│  ───────────────                                                             │
│                                                                              │
│  1. New Image (JPG)                                                          │
│       │                                                                      │
│       ▼                                                                      │
│  2. Resize & Normalize → (224, 224, 3)                                      │
│       │                                                                      │
│       ▼                                                                      │
│  3. ResNet50 → Feature Vector (2048)                                        │
│       │                                                                      │
│       ▼                                                                      │
│  4. Encoder(feature) → hidden_state (512)                                   │
│       │                                                                      │
│       ▼                                                                      │
│  5. Initialize: token = <start>                                             │
│       │                                                                      │
│       ▼                                                                      │
│  6. Loop (Greedy/Beam Search):                                              │
│      ┌────────────────────────────────────────────────────┐                 │
│      │  Decoder(token, hidden) → logits                  │                 │
│      │    │                                               │                 │
│      │    ▼                                               │                 │
│      │  next_token = argmax(logits)                      │                 │
│      │    │                                               │                 │
│      │    ▼                                               │                 │
│      │  caption.append(next_token)                       │                 │
│      │    │                                               │                 │
│      │    ▼                                               │                 │
│      │  if next_token == <end>: break                    │                 │
│      └────────────────────────────────────────────────────┘                 │
│       │                                                                      │
│       ▼                                                                      │
│  7. Convert indices to words → "a dog playing in the park"                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Model Performance

### Training Results

After 25 epochs:
- **Final Training Loss**: 2.9497
- **Final Validation Loss**: 3.0456

### Loss Curve Analysis

The training shows:
1. **Rapid initial decrease** (Epochs 1-5): Loss drops from ~5.0 to ~3.7
2. **Steady improvement** (Epochs 5-20): Gradual decrease to ~3.0
3. **Convergence** (Epochs 20-25): Minimal improvement, model stabilizes

### Evaluation Metrics

| Metric | Score | Description |
|--------|-------|-------------|
| **BLEU-1** | 0.5891 | Unigram precision |
| **BLEU-2** | 0.3983 | Bigram precision |
| **BLEU-3** | 0.2616 | Trigram precision |
| **BLEU-4** | 0.1743 | 4-gram precision |
| **1-gram Precision** | 0.4691 | Token-level precision |
| **1-gram Recall** | 0.4675 | Token-level recall |
| **1-gram F1** | 0.4553 | Harmonic mean |
| **ROUGE-L** | 0.4303 | Longest common subsequence |

**Key Observations**:
- Strong BLEU-1 score (0.59) indicates good word-level accuracy
- Decreasing BLEU scores (1→4) show difficulty with longer n-grams
- Balanced precision/recall suggests no significant bias
- Average caption length: ~12.5 words (close to ground truth: 12.3)

---

## 8. Key Design Decisions

### 1. **Why Cache Image Features?**
- **Computational Efficiency**: Avoids redundant CNN forward passes
- **Faster Experimentation**: Can quickly test different decoder architectures
- **Memory Savings**: Store 2048-dim vectors instead of full images

### 2. **Why LSTM over GRU?**
- **Better Long-term Dependencies**: Separate cell state (c_t) and hidden state (h_t)
- **More Control**: Three gates (forget, input, output) vs two in GRU
- **Proven Performance**: Standard choice for sequence generation tasks

### 3. **Why Teacher Forcing?**
- **Faster Convergence**: Model learns from correct context
- **Stable Training**: Reduces error accumulation
- **Trade-off**: May cause exposure bias (model never sees own mistakes during training)

### 4. **Why Beam Search over Greedy?**
- **Better Quality**: Explores multiple hypotheses
- **Diversity**: Can recover from early mistakes
- **Trade-off**: Slower inference (k times more computation)

### 5. **Why Dropout?**
- **Regularization**: Prevents overfitting
- **Ensemble Effect**: Simulates training multiple models
- **Standard Practice**: p=0.5 is common for NLP tasks

---

## 9. Limitations & Future Work

### Current Limitations

1. **Exposure Bias**: Model trained with teacher forcing may struggle with own predictions
2. **Fixed Vocabulary**: Cannot handle out-of-vocabulary words
3. **Single Image Feature**: No attention mechanism to focus on different image regions
4. **Limited Context**: Single LSTM layer may miss complex dependencies

### Potential Improvements

1. **Attention Mechanism**:
   ```
   ┌─────────────────────────────────────────┐
   │  CNN → Spatial Features (7×7×2048)     │
   │    │                                    │
   │    ▼                                    │
   │  Attention(query=h_t, keys=features)   │
   │    │                                    │
   │    ▼                                    │
   │  Context Vector → Decoder              │
   └─────────────────────────────────────────┘
   ```
   - Allow model to focus on relevant image regions
   - Improve caption quality and interpretability

2. **Transformer Architecture**:
   - Replace LSTM with self-attention
   - Better parallelization
   - Capture long-range dependencies

3. **Scheduled Sampling**:
   - Gradually mix teacher forcing with model predictions
   - Reduce exposure bias

4. **Byte-Pair Encoding (BPE)**:
   - Handle rare/unknown words
   - Reduce vocabulary size

5. **Multi-layer LSTM**:
   - Increase model capacity
   - Learn hierarchical representations

---

## 10. Technical Specifications

### Hardware Requirements

- **GPU**: NVIDIA Tesla T4 (or equivalent)
- **Memory**: 16GB+ RAM
- **Storage**: ~5GB for cached features

### Software Dependencies

```python
torch==1.x
torchvision==0.x
PIL
tqdm
numpy
matplotlib
nltk
rouge-score
```

### File Structure

```
project/
├── flickr30k_features.pkl      # Cached image features (2048-dim)
├── flickr30k_vocab.pkl         # Vocabulary mappings
├── flickr30k_captions.pkl      # Tokenized captions
├── genai-ass01-22f-3327.ipynb  # Main notebook
└── /kaggle/input/flickr30k/
    ├── Images/                 # Raw images
    └── captions.txt            # Raw captions
```

### Model Size

- **Encoder**: ~1M parameters
- **Decoder**: ~20M parameters (depends on vocab size)
- **Total**: ~21M trainable parameters

---

## 11. Working Explanation

### Step-by-Step Execution

#### Phase 1: Feature Extraction
1. Load Flickr30k images from disk
2. Preprocess each image (resize, normalize)
3. Pass through ResNet50 (remove final layer)
4. Extract 2048-dim feature vector
5. Save to `flickr30k_features.pkl` as dictionary: `{image_name: feature_vector}`

#### Phase 2: Text Preprocessing
1. Load `captions.txt` (CSV format)
2. For each caption:
   - Convert to lowercase
   - Remove punctuation
   - Add `<start>` and `<end>` tokens
3. Build vocabulary:
   - Count word frequencies
   - Filter words with frequency < 5
   - Add special tokens: `<pad>`, `<start>`, `<end>`, `<unk>`
4. Convert captions to sequences of indices
5. Save vocabulary and sequences to disk

#### Phase 3: Model Training
1. Load cached features and captions
2. Split data: 90% train, 10% validation
3. Create DataLoader with padding collation
4. For each epoch:
   - **Training**:
     - Sample batch of (features, captions)
     - Encode features → hidden state
     - Decode captions with teacher forcing
     - Compute loss (ignore padding)
     - Backpropagate and update weights
   - **Validation**:
     - Compute loss without gradients
     - Log metrics
5. Save trained model

#### Phase 4: Inference
1. Load new image
2. Extract features using ResNet50
3. Encode features → initial hidden state
4. Generate caption:
   - **Greedy**: Select highest probability word at each step
   - **Beam**: Maintain top-k hypotheses, select best final sequence
5. Convert token indices to words
6. Return generated caption

### Example Walkthrough

**Input Image**: Dog playing in park

**Step 1**: Feature Extraction
```
Image (224×224×3) → ResNet50 → [0.23, -0.45, 0.67, ..., 0.12] (2048-dim)
```

**Step 2**: Encoding
```
Feature [2048] → Linear(2048→512) → ReLU → Dropout → Hidden [512]
```

**Step 3**: Decoding (Greedy)
```
t=0: Input: <start> → LSTM → Logits → argmax → "a"
t=1: Input: "a" → LSTM → Logits → argmax → "dog"
t=2: Input: "dog" → LSTM → Logits → argmax → "playing"
t=3: Input: "playing" → LSTM → Logits → argmax → "in"
t=4: Input: "in" → LSTM → Logits → argmax → "the"
t=5: Input: "the" → LSTM → Logits → argmax → "park"
t=6: Input: "park" → LSTM → Logits → argmax → <end>
```

**Output Caption**: "a dog playing in the park"

---

## 12. Mathematical Formulation

### Loss Function

During training, we minimize the negative log-likelihood:

```
L = -∑_{t=1}^{T} log P(y_t | y_1, ..., y_{t-1}, I)
```

Where:
- `I` = image features
- `y_t` = target word at timestep t
- `T` = caption length

### LSTM Equations

At each timestep t:

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)      # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)      # Input gate
c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)   # Cell candidate
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t          # Cell state
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)      # Output gate
h_t = o_t ⊙ tanh(c_t)                    # Hidden state
```

### Beam Search Scoring

For a sequence `S = [w_1, w_2, ..., w_n]`:

```
score(S) = (1/n) · ∑_{i=1}^{n} log P(w_i | w_1, ..., w_{i-1}, I)
```

Length normalization prevents bias toward shorter sequences.

---

## Conclusion

This architecture demonstrates a classic **Seq2Seq approach** to image captioning:
1. **CNN** extracts visual features
2. **Encoder** transforms features into initial state
3. **LSTM Decoder** generates caption word-by-word
4. **Beam Search** improves caption quality

The modular design allows easy experimentation with different components (e.g., replacing LSTM with Transformer, adding attention mechanism).

**Key Strengths**:
- Efficient feature caching
- Strong baseline performance (BLEU-1: 0.59)
- Flexible inference strategies

**Future Directions**:
- Add attention mechanism
- Experiment with Transformer architecture
- Implement scheduled sampling to reduce exposure bias
