# Contributing to Image Captioning Seq2Seq

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all. Please be respectful and constructive in your interactions.

### Our Standards

**Positive behavior includes:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community

**Unacceptable behavior includes:**
- Harassment, trolling, or derogatory comments
- Publishing others' private information
- Other conduct which could reasonably be considered inappropriate

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

**When reporting bugs, include:**
- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior vs. actual behavior
- Your environment (OS, Python version, PyTorch version)
- Error messages or screenshots
- Sample code if applicable

**Example:**
```
Title: Training crashes with CUDA out of memory error

Environment:
- OS: Windows 11
- Python: 3.8.10
- PyTorch: 1.10.0
- GPU: NVIDIA GTX 1060 6GB

Steps to reproduce:
1. Run ImageCaptioning.ipynb
2. Execute training cell with BATCH_SIZE=64
3. Error occurs after first epoch

Error message:
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB...
```

### Suggesting Enhancements

Enhancement suggestions are welcome! Please provide:
- A clear and descriptive title
- Detailed explanation of the proposed feature
- Why this enhancement would be useful
- Possible implementation approach (if you have one)

### Pull Request Process

1. **Fork the repository** and create your branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards below

3. **Test your changes** thoroughly:
   - Ensure the notebook runs without errors
   - Verify model training/inference still works
   - Check that evaluation metrics are computed correctly

4. **Update documentation**:
   - Update README.md if you changed functionality
   - Add docstrings to new functions
   - Update architecture.md if you modified the model

5. **Commit your changes** with clear messages:
   ```bash
   git commit -m "feat: add attention mechanism to decoder"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request** with:
   - Clear title describing the change
   - Detailed description of what changed and why
   - Reference to related issues (if any)
   - Screenshots/examples if applicable

## Coding Standards

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use meaningful variable names
- Maximum line length: 100 characters
- Use 4 spaces for indentation (no tabs)

### Documentation

- Add docstrings to all functions and classes:
  ```python
  def beam_search(image_feature, beam_width=5, max_len=50):
      """
      Generate caption using beam search.
      
      Args:
          image_feature (torch.Tensor): Image feature vector (2048-dim)
          beam_width (int): Number of beams to maintain
          max_len (int): Maximum caption length
          
      Returns:
          list: Generated caption as list of words
      """
      # Implementation
  ```

- Comment complex logic
- Keep comments up-to-date with code changes

### Jupyter Notebooks

- Clear all outputs before committing
- Use markdown cells to explain each section
- Keep cells focused on one task
- Include error handling in code cells

### Git Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

**Examples:**
```
feat: add attention mechanism to decoder
fix: resolve CUDA out of memory error with gradient accumulation
docs: update README with new hyperparameters
refactor: extract data loading into separate module
```

## Development Setup

1. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/image-captioning-seq2seq.git
   cd image-captioning-seq2seq
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a branch for your work**:
   ```bash
   git checkout -b feature/my-new-feature
   ```

## Areas for Contribution

We welcome contributions in these areas:

### High Priority
- [ ] Implement attention mechanism
- [ ] Add scheduled sampling to reduce exposure bias
- [ ] Create Gradio/Streamlit web app for inference
- [ ] Add more evaluation metrics (CIDEr, SPICE)

### Medium Priority
- [ ] Implement transformer-based decoder
- [ ] Add data augmentation techniques
- [ ] Create visualization tools for attention weights
- [ ] Improve beam search with length normalization

### Low Priority
- [ ] Add support for other datasets (MSCOCO, etc.)
- [ ] Implement ensemble methods
- [ ] Add model compression techniques
- [ ] Create Docker container for easy deployment

## Questions?

Feel free to open an issue with the `question` label if you have any questions about contributing.

## Recognition

Contributors will be acknowledged in the README.md file.

Thank you for contributing! 🎉
