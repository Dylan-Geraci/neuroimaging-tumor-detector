[![CI](https://github.com/Dylan-Geraci/neuroimaging-tumor-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/Dylan-Geraci/neuroimaging-tumor-detector/actions/workflows/ci.yml)

# Brain Tumor Classification System

Deep learning system for classifying brain MRI scans into four categories (Glioma, Meningioma, Pituitary Tumor, No Tumor) using a fine-tuned ResNet18 model with 97.10% test accuracy. Includes Grad-CAM visual explanations and a web interface for real-time analysis.

## Features

- 97.10% test accuracy on 1,311 held-out images
- Batch processing: analyze multiple MRI scans simultaneously
- Visual explanations via Grad-CAM attention heatmaps
- Aggregated diagnosis with agreement scoring across batches
- REST API for integration with clinical workflows

## Tech Stack

**Deep Learning**: PyTorch, ResNet18 (transfer learning), Grad-CAM
**Backend**: FastAPI, uvicorn
**Frontend**: JavaScript, HTML5, CSS3
**Evaluation**: scikit-learn, matplotlib, seaborn

## Getting Started

```bash
git clone <repository-url>
cd neuroimaging-tumor-detector
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 main.py
# Open http://localhost:8000/static/index.html
```

## CLI Tools

```bash
python src/predict.py --image path/to/mri.jpg   # Single prediction
python src/evaluate.py                           # Model evaluation
python src/visualize.py                          # Grad-CAM visualizations
```

## Model

- Architecture: ResNet18 pre-trained on ImageNet, fine-tuned on brain MRI
- Input: 224x224 grayscale MRI images
- Training: 30 epochs with Adam optimizer and data augmentation
- Dataset: 7,000+ images from the Brain Tumor MRI Dataset (Kaggle)

## Disclaimer

This system is intended for research and educational purposes only. Not for clinical diagnosis.
