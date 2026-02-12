# Image Captioning with Seq2Seq Architecture

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

This project implements an **Image Captioning System** using a **Sequence-to-Sequence (Seq2Seq)** architecture with **LSTM** networks. The model generates natural language descriptions for images by combining:
- **Computer Vision**: ResNet50 for feature extraction
- **Natural Language Processing**: 2-layer stacked LSTM for caption generation

### Key Features

✨ **Enhanced Architecture**
- 2-layer stacked LSTM decoder for hierarchical learning
- 512-dimensional word embeddings
- 1024-dimensional hidden states
- 10,029 word vocabulary

🎯 **Strong Performance**
- BLEU-1: 0.6007
- BLEU-4: 0.1760
- ROUGE-L: 0.4400
- Training on Flickr30k dataset (31,783 images)

🚀 **Advanced Features**
- Beam search inference (width=5) for better caption quality
- Cached image features for efficient training
- Comprehensive evaluation metrics (BLEU, METEOR, ROUGE)

## 📊 Model Architecture

The system uses a two-stage approach:

1. **Feature Extraction**: Pre-trained ResNet50 extracts 2048-dim image features
2. **Caption Generation**: 2-layer LSTM decoder generates captions word-by-word

For detailed architecture information, see [architecture.md](architecture.md).

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/CodeRafay/image-captioning-seq2seq.git
cd image-captioning-seq2seq
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data** (required for evaluation)
```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Usage

#### Training the Model

Open and run the Jupyter notebook:
```bash
jupyter notebook ImageCaptioning.ipynb
```

The notebook contains:
- Part 1: Feature extraction with ResNet50
- Part 2: Vocabulary and text preprocessing
- Part 3: Seq2Seq model definition
- Part 4: Training and inference
- Quantitative evaluation and caption examples

#### Generating Captions

```python
from PIL import Image
import torch

# Load the model
encoder = Encoder().to(device)
decoder = Decoder().to(device)
encoder.load_state_dict(torch.load('encoder.pth'))
decoder.load_state_dict(torch.load('decoder.pth'))

# Load and preprocess image
image = Image.open('your_image.jpg')
# ... (see USAGE.md for complete example)

# Generate caption
caption = beam_search(image_feature, beam_width=5)
print(' '.join(caption))
```

For detailed usage instructions, see [USAGE.md](USAGE.md).

## 📈 Performance Metrics

### Evaluation Results (35 epochs)

| Metric | Score | Description |
|--------|-------|-------------|
| **BLEU-1** | 0.6007 | Unigram precision |
| **BLEU-2** | 0.4065 | Bigram precision |
| **BLEU-3** | 0.2666 | Trigram precision |
| **BLEU-4** | 0.1760 | 4-gram precision |
| **1-gram F1** | 0.4649 | Token-level F1 score |
| **ROUGE-L** | 0.4400 | Longest common subsequence |

### Training Results

- **Final Training Loss**: 2.4667
- **Final Validation Loss**: 2.9608
- **Training Time**: ~30 minutes per epoch on Tesla T4 GPU

## 🏗️ Project Structure

```
image-captioning-seq2seq/
├── ImageCaptioning.ipynb      # Main notebook with complete pipeline
├── architecture.md             # Detailed architecture documentation
├── improvements.md             # Model improvement suggestions
├── Problem.md                  # Original problem statement
├── README.md                   # This file
├── USAGE.md                    # Detailed usage guide
├── CONTRIBUTING.md             # Contribution guidelines
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore rules
├── model_config.json           # Model configuration
├── encoder.pth                 # Pre-trained encoder weights (not in repo)
├── decoder.pth                 # Pre-trained decoder weights (not in repo)
├── flickr30k_features.pkl      # Cached image features (not in repo)
├── flickr30k_vocab.pkl         # Vocabulary mappings (not in repo)
└── flickr30k_captions.pkl      # Tokenized captions (not in repo)
```

**Note**: Model weights (.pth) and data files (.pkl) are not included in the repository due to their large size. You'll need to train the model or download pre-trained weights separately.

## 🔧 Model Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `EMBED_SIZE` | 512 | Word embedding dimension |
| `HIDDEN_SIZE` | 1024 | LSTM hidden state size |
| `NUM_LAYERS` | 2 | Number of stacked LSTM layers |
| `VOCAB_SIZE` | 10,029 | Total vocabulary size |
| `DROPOUT` | 0.5 | Dropout probability |
| `BATCH_SIZE` | 64 | Training batch size |
| `LEARNING_RATE` | 3e-4 | Adam optimizer learning rate |
| `NUM_EPOCHS` | 35 | Training epochs |

## 📚 Documentation

- **[architecture.md](architecture.md)**: Comprehensive architecture documentation with diagrams
- **[improvements.md](improvements.md)**: Suggested improvements and performance optimization tips
- **[USAGE.md](USAGE.md)**: Detailed usage instructions and API reference
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Guidelines for contributing to the project

## 🎯 Model Improvements

This implementation includes several enhancements over the baseline:

- **Increased model capacity**: 2-layer LSTM (was 1-layer)
- **Larger embeddings**: 512-dim (was 256-dim)
- **Larger hidden states**: 1024-dim (was 512-dim)
- **Better vocabulary coverage**: MIN_FREQ=3 (was 5)
- **Extended training**: 35 epochs (was 25)

See [improvements.md](improvements.md) for detailed improvement analysis and future work suggestions.

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory Error**
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size in the notebook:
```python
BATCH_SIZE = 32  # or 16
```

**2. NLTK Data Not Found**
```
LookupError: Resource wordnet not found
```
**Solution**: Download required NLTK data:
```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

**3. Model Files Not Found**
```
FileNotFoundError: encoder.pth not found
```
**Solution**: Train the model first by running all cells in the notebook, or download pre-trained weights.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Flickr30k dataset
- **Pre-trained Model**: ResNet50 from torchvision
- **Framework**: PyTorch
- **Evaluation**: NLTK, rouge-score

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the repository owner.

## 🔗 Related Resources

- [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)
- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

**Note**: This is an academic project for educational purposes. The model is trained on the Flickr30k dataset and may not generalize well to all types of images.
