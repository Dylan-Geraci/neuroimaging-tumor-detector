# Brain Tumor Classification System

AI-powered system for classifying brain MRI scans into four categories (Glioma, Meningioma, Pituitary Tumor, No Tumor) with **97.10% test accuracy**. Features Grad-CAM visual explanations showing which brain regions influenced each prediction.

## Key Features

- **97.10% Test Accuracy** on 1,311 held-out test images
- **Visual Explanations**: Grad-CAM heatmaps highlight model attention areas
- **Web Application**: Professional interface for real-time MRI analysis
- **REST API**: FastAPI backend for system integration

## Tech Stack

- **Deep Learning**: PyTorch, ResNet18 (Transfer Learning)
- **Explainability**: Grad-CAM
- **Backend**: FastAPI, uvicorn
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Evaluation**: scikit-learn, matplotlib, seaborn

## Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd neuroimaging-tumor-detector
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run web application
python main.py
# Visit http://localhost:8000/static/index.html
```

## CLI Tools

```bash
# Single image prediction
python src/predict.py --image path/to/mri.jpg

# Model evaluation
python src/evaluate.py

# Generate Grad-CAM visualizations
python src/visualize.py
```

## Model Details

- **Architecture**: ResNet18 (pre-trained on ImageNet, fine-tuned on MRI data)
- **Input**: Grayscale MRI (224×224, single channel)
- **Training**: 14 epochs, Adam optimizer, data augmentation
- **Dataset**: 7,000+ brain MRI images across 4 classes

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) - 7,000+ MRI images across 4 tumor classes (Glioma, Meningioma, Pituitary, No Tumor)

## Disclaimer

Not intended for clinical diagnosis. Educational and research purposes only.

---

**Credits**: Dataset by Masoud Nickparvar | ResNet18 (torchvision) | Grad-CAM (pytorch-grad-cam)