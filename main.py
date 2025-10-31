"""
FastAPI backend for brain tumor classification web application.

Provides REST API endpoints for:
- Health checking
- Image upload and prediction
- Grad-CAM visualization generation
"""

import io
import base64
from pathlib import Path
from typing import Dict

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2

from src.model import create_model
from src.data import CLASSES
from src.predict import get_inference_transforms
from src.gradcam import create_gradcam

# Initialize FastAPI app
app = FastAPI(
    title="Brain Tumor Classification API",
    description="AI-powered brain tumor classification with visual explanations",
    version="1.0.0"
)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and Grad-CAM objects (loaded on startup)
MODEL = None
GRADCAM = None
DEVICE = None
TRANSFORM = None


@app.on_event("startup")
async def load_model():
    """Load model and Grad-CAM on startup."""
    global MODEL, GRADCAM, DEVICE, TRANSFORM

    print("Loading model...")

    # Determine device
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'
    else:
        DEVICE = 'cpu'

    print(f"Using device: {DEVICE}")

    # Load model
    checkpoint_path = "models/best_model.pth"
    MODEL = create_model(num_classes=len(CLASSES), pretrained=False, device=DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    MODEL.load_state_dict(checkpoint['model_state_dict'])
    MODEL.eval()

    print(f"Model loaded from {checkpoint_path}")
    print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}")

    # Initialize Grad-CAM
    GRADCAM = create_gradcam(MODEL, DEVICE)
    print("Grad-CAM initialized")

    # Get transforms
    TRANSFORM = get_inference_transforms()
    print("Ready to accept requests!")


def image_to_base64(image: np.ndarray) -> str:
    """
    Convert numpy image to base64 string for JSON response.

    Args:
        image: Numpy array (H, W) or (H, W, 3)

    Returns:
        Base64 encoded string
    """
    # Convert to PIL Image
    if len(image.shape) == 2:  # Grayscale
        pil_image = Image.fromarray((image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8))
    else:  # RGB
        pil_image = Image.fromarray(image.astype(np.uint8))

    # Convert to bytes
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")

    # Encode to base64
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return f"data:image/png;base64,{img_str}"


@app.get("/")
async def root():
    """Root endpoint - redirect to docs."""
    return {
        "message": "Brain Tumor Classification API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "gradcam_loaded": GRADCAM is not None,
        "device": DEVICE,
        "classes": CLASSES
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """
    Predict brain tumor type from uploaded MRI image.

    Args:
        file: Uploaded image file

    Returns:
        JSON with prediction, confidence, probabilities, and visualizations
    """
    if MODEL is None or GRADCAM is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('L')  # Grayscale

        # Store original image (resized to 224x224 for consistency)
        original_resized = image.resize((224, 224), Image.Resampling.LANCZOS)
        original_np = np.array(original_resized)

        # Preprocess for model
        input_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

        # Get prediction
        with torch.no_grad():
            output = MODEL(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[predicted_class].item()

        # Generate Grad-CAM
        grayscale_cam = GRADCAM.generate_heatmap(input_tensor, predicted_class)

        # Create overlay
        original_normalized = original_np / 255.0
        original_rgb = np.stack([original_normalized] * 3, axis=-1)

        # Create heatmap overlay
        from pytorch_grad_cam.utils.image import show_cam_on_image
        overlay = show_cam_on_image(original_rgb, grayscale_cam, use_rgb=True)

        # Create colored heatmap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * grayscale_cam),
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Format response
        response = {
            "success": True,
            "prediction": {
                "class": CLASSES[predicted_class],
                "class_index": predicted_class,
                "confidence": float(confidence),
                "probabilities": {
                    CLASSES[i]: float(probabilities[i])
                    for i in range(len(CLASSES))
                }
            },
            "images": {
                "original": image_to_base64(original_np),
                "heatmap": image_to_base64(heatmap_colored),
                "overlay": image_to_base64(overlay)
            }
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Mount static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn

    print("Starting Brain Tumor Classification API...")
    print("Visit http://localhost:8000/static/index.html for the web interface")
    print("Visit http://localhost:8000/docs for API documentation")

    uvicorn.run(app, host="0.0.0.0", port=8000)
