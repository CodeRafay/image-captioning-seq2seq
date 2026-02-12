# Image Captioning with Enhanced Model

## Overview
This project implements an advanced image captioning model using PyTorch. The model leverages a ResNet50-based encoder and an LSTM-based decoder to generate descriptive captions for images. The project includes a Gradio-based web application for easy interaction and evaluation metrics such as BLEU, METEOR, and ROUGE to assess the model's performance.

## Features
- **Encoder-Decoder Architecture**: Combines ResNet50 for feature extraction and LSTM for sequence generation.
- **Beam Search**: Enhanced inference with a beam width of 8 for better caption quality.
- **Evaluation Metrics**: BLEU, METEOR, and ROUGE scores for comprehensive performance analysis.
- **Gradio App**: User-friendly interface for uploading images and viewing generated captions.
- **Training Enhancements**: Gradient clipping and BatchNorm for stable training.

## Installation
Clone the repository and install dependencies:
`bash
git clone https://github.com/CodeRafay/image-captioning-seq2seq.git
cd image-captioning-seq2seq
pip install -r requirements.txt
`

## Usage
Run the Jupyter notebook to train or evaluate the model:
`bash
jupyter notebook genai-ass01-22f-3327.ipynb
`

## Evaluation
Performance is evaluated using BLEU, METEOR, and ROUGE metrics.

## License
MIT License - see LICENSE file for details.
