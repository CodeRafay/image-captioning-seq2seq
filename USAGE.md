# Usage Guide

This guide provides detailed instructions for using the image captioning model.

## Table of Contents

- [Loading Pre-trained Models](#loading-pre-trained-models)
- [Generating Captions](#generating-captions)
- [Training from Scratch](#training-from-scratch)
- [Fine-tuning](#fine-tuning)
- [API Reference](#api-reference)

## Loading Pre-trained Models

### Prerequisites

Ensure you have the required files:
- `encoder.pth`: Pre-trained encoder weights
- `decoder.pth`: Pre-trained decoder weights
- `flickr30k_vocab.pkl`: Vocabulary mappings
- `model_config.json`: Model configuration

### Loading the Model

```python
import torch
import pickle
import json
from PIL import Image
from torchvision import transforms

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load configuration
with open('model_config.json', 'r') as f:
    config = json.load(f)

# Load vocabulary
with open('flickr30k_vocab.pkl', 'rb') as f:
    vocab_data = pickle.load(f)
    word2idx = vocab_data['word2idx']
    idx2word = vocab_data['idx2word']

# Initialize models
encoder = Encoder(
    feature_size=2048,
    hidden_size=config['hidden_size']
).to(device)

decoder = Decoder(
    embed_size=config['embed_size'],
    hidden_size=config['hidden_size'],
    vocab_size=config['vocab_size'],
    num_layers=config['num_layers']
).to(device)

# Load weights
encoder.load_state_dict(torch.load('encoder.pth', map_location=device))
decoder.load_state_dict(torch.load('decoder.pth', map_location=device))

# Set to evaluation mode
encoder.eval()
decoder.eval()

print("Models loaded successfully!")
```

## Generating Captions

### For a Single Image

```python
from torchvision import models

# Load ResNet50 for feature extraction
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet = resnet.to(device)
resnet.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def generate_caption(image_path, method='beam', beam_width=5):
    """
    Generate caption for an image.
    
    Args:
        image_path (str): Path to the image file
        method (str): 'greedy' or 'beam'
        beam_width (int): Beam width for beam search
        
    Returns:
        str: Generated caption
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Extract features
    with torch.no_grad():
        features = resnet(image_tensor).view(1, -1)
    
    # Generate caption
    if method == 'greedy':
        caption_words = greedy_search(features)
    else:
        caption_words = beam_search(features, beam_width=beam_width)
    
    # Convert to string
    caption = ' '.join(caption_words)
    return caption

# Example usage
caption = generate_caption('path/to/your/image.jpg', method='beam', beam_width=5)
print(f"Generated caption: {caption}")
```

### Greedy Search Implementation

```python
def greedy_search(image_feature, max_len=50):
    """
    Generate caption using greedy search.
    
    Args:
        image_feature (torch.Tensor): Image feature vector (1, 2048)
        max_len (int): Maximum caption length
        
    Returns:
        list: Generated caption as list of words
    """
    with torch.no_grad():
        # Encode image
        encoder_out = encoder(image_feature)
        hidden, cell = decoder.init_hidden_state(encoder_out)
        
        # Start with <start> token
        caption = [word2idx['<start>']]
        
        for _ in range(max_len):
            # Get current word embedding
            input_token = torch.tensor([caption[-1]], dtype=torch.long).unsqueeze(0).to(device)
            
            # Decoder step
            logits, hidden, cell = decoder(input_token, hidden, cell)
            
            # Get most probable word
            next_word_idx = logits.argmax(2).item()
            
            # Stop if <end> token
            if next_word_idx == word2idx['<end>']:
                break
                
            caption.append(next_word_idx)
        
        # Convert indices to words (skip <start>)
        caption_words = [idx2word[idx] for idx in caption[1:]]
        
    return caption_words
```

### Beam Search Implementation

```python
def beam_search(image_feature, beam_width=5, max_len=50):
    """
    Generate caption using beam search.
    
    Args:
        image_feature (torch.Tensor): Image feature vector (1, 2048)
        beam_width (int): Number of beams to maintain
        max_len (int): Maximum caption length
        
    Returns:
        list: Generated caption as list of words
    """
    with torch.no_grad():
        # Encode image
        encoder_out = encoder(image_feature)
        hidden, cell = decoder.init_hidden_state(encoder_out)
        
        # Initialize beams: (sequence, score, hidden, cell)
        beams = [([word2idx['<start>']], 0.0, hidden, cell)]
        completed = []
        
        for _ in range(max_len):
            candidates = []
            
            for seq, score, h, c in beams:
                # Skip if sequence already ended
                if seq[-1] == word2idx['<end>']:
                    completed.append((seq, score, h, c))
                    continue
                
                # Get predictions for current sequence
                input_token = torch.tensor([seq[-1]], dtype=torch.long).unsqueeze(0).to(device)
                logits, h_new, c_new = decoder(input_token, h, c)
                
                # Get top-k predictions
                log_probs = torch.log_softmax(logits.squeeze(1), dim=-1)
                topk_probs, topk_idx = log_probs.topk(beam_width, dim=-1)
                
                # Create new candidates
                for k in range(beam_width):
                    next_word = topk_idx[0, k].item()
                    next_score = score + topk_probs[0, k].item()
                    candidates.append((seq + [next_word], next_score, h_new, c_new))
            
            if not candidates:
                break
            
            # Keep top beam_width candidates (with length normalization)
            candidates.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
            beams = candidates[:beam_width]
            
            # Early stopping if all beams ended
            if all(beam[0][-1] == word2idx['<end>'] for beam in beams):
                completed.extend(beams)
                break
        
        # Add remaining beams to completed
        completed.extend(beams)
        
        # Select best sequence
        completed.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
        best_seq = completed[0][0]
        
        # Convert to words (skip <start> and <end>)
        caption_words = [idx2word[idx] for idx in best_seq 
                        if idx not in (word2idx['<start>'], word2idx['<end>'])]
        
    return caption_words
```

### Batch Processing

```python
import os
from tqdm import tqdm

def process_directory(image_dir, output_file='captions.txt'):
    """
    Generate captions for all images in a directory.
    
    Args:
        image_dir (str): Directory containing images
        output_file (str): Output file for captions
    """
    image_files = [f for f in os.listdir(image_dir) 
                   if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    with open(output_file, 'w') as f:
        for img_file in tqdm(image_files, desc="Processing images"):
            img_path = os.path.join(image_dir, img_file)
            caption = generate_caption(img_path, method='beam', beam_width=5)
            f.write(f"{img_file}: {caption}\n")
    
    print(f"Captions saved to {output_file}")

# Example usage
process_directory('path/to/images/', 'output_captions.txt')
```

## Training from Scratch

### Prepare Dataset

```python
# 1. Extract image features
from feature_extraction import extract_features

features_dict = extract_features(
    image_dir='path/to/flickr30k/images',
    output_file='flickr30k_features.pkl'
)

# 2. Build vocabulary
from preprocessing import build_vocabulary

vocab_data = build_vocabulary(
    captions_file='path/to/captions.txt',
    min_freq=3,
    output_file='flickr30k_vocab.pkl'
)
```

### Training Loop

```python
# See ImageCaptioning.ipynb for complete training code

# Key hyperparameters
EMBED_SIZE = 512
HIDDEN_SIZE = 1024
NUM_LAYERS = 2
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
NUM_EPOCHS = 35

# Training
for epoch in range(NUM_EPOCHS):
    # Training phase
    encoder.train()
    decoder.train()
    
    for features, captions, lengths in train_loader:
        # Forward pass
        encoder_out = encoder(features)
        hidden, cell = decoder.init_hidden_state(encoder_out)
        logits, _, _ = decoder(captions[:, :-1], hidden, cell)
        
        # Compute loss
        loss = criterion(logits.reshape(-1, vocab_size), 
                        captions[:, 1:].reshape(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Fine-tuning

### On New Dataset

```python
# 1. Load pre-trained models
encoder.load_state_dict(torch.load('encoder.pth'))
decoder.load_state_dict(torch.load('decoder.pth'))

# 2. Freeze encoder (optional)
for param in encoder.parameters():
    param.requires_grad = False

# 3. Fine-tune decoder with lower learning rate
optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)

# 4. Train on new dataset
# ... (same training loop as above)
```

## API Reference

### Encoder Class

```python
class Encoder(nn.Module):
    """
    Encoder that transforms image features into initial hidden state.
    
    Args:
        feature_size (int): Input feature dimension (default: 2048)
        hidden_size (int): Output hidden state dimension (default: 1024)
    """
```

### Decoder Class

```python
class Decoder(nn.Module):
    """
    LSTM-based decoder for caption generation.
    
    Args:
        embed_size (int): Word embedding dimension (default: 512)
        hidden_size (int): LSTM hidden state dimension (default: 1024)
        vocab_size (int): Vocabulary size
        num_layers (int): Number of LSTM layers (default: 2)
        dropout (float): Dropout probability (default: 0.5)
    """
```

## Troubleshooting

### Common Issues

**1. Model outputs repetitive captions**
- Try increasing beam width: `beam_width=10`
- Use greedy search instead
- Check if model is properly trained

**2. Captions are too short**
- Increase `max_len` parameter
- Check beam search length normalization

**3. Out of memory during inference**
- Reduce beam width
- Process images one at a time instead of batches

## Examples

See the `ImageCaptioning.ipynb` notebook for complete working examples of:
- Feature extraction
- Vocabulary building
- Model training
- Caption generation
- Evaluation metrics

## Further Reading

- [architecture.md](architecture.md) - Detailed model architecture
- [improvements.md](improvements.md) - Performance optimization tips
- [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute
