"""
FastAPI backend for brain tumor classification web application.

Provides REST API endpoints for:
- Health checking
- Image upload and prediction
- Grad-CAM visualization generation
"""

import io
import base64
from contextlib import asynccontextmanager

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.model import create_model
from src.data import CLASSES
from src.predict import get_inference_transforms
from src.gradcam import create_gradcam
from src.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and Grad-CAM on startup, clean up on shutdown."""
    print("Loading model...")

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Load model
    model = create_model(num_classes=len(CLASSES), pretrained=False, device=device)
    checkpoint = torch.load(settings.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded from {settings.model_path}")
    print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}")

    # Initialize Grad-CAM
    gradcam = create_gradcam(model, device)
    print("Grad-CAM initialized")

    # Get transforms
    transform = get_inference_transforms()
    print("Ready to accept requests!")

    # Store on app.state
    app.state.model = model
    app.state.gradcam = gradcam
    app.state.device = device
    app.state.transform = transform

    yield

    # Cleanup
    print("Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Brain Tumor Classification API",
    description="AI-powered brain tumor classification with visual explanations",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _process_single_image(
    image_bytes: bytes,
    model: torch.nn.Module,
    gradcam,
    device: str,
    transform,
) -> dict:
    """
    Shared prediction pipeline for a single image.

    Args:
        image_bytes: Raw image bytes
        model: Loaded BrainTumorClassifier
        gradcam: BrainTumorGradCAM instance
        device: Device string
        transform: Preprocessing transforms

    Returns:
        Dict with class, confidence, probabilities, and base64 images
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    # Store original image (resized to 224x224 for consistency)
    original_resized = image.resize((224, 224), Image.Resampling.LANCZOS)
    original_np = np.array(original_resized)

    # Preprocess for model
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[predicted_class].item()

    # Generate Grad-CAM
    grayscale_cam = gradcam.generate_heatmap(input_tensor, predicted_class)

    # Create overlay
    original_normalized = original_np / 255.0
    original_rgb = np.stack([original_normalized] * 3, axis=-1)
    overlay = show_cam_on_image(original_rgb, grayscale_cam, use_rgb=True)

    # Create colored heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    return {
        "class": CLASSES[predicted_class],
        "class_index": predicted_class,
        "confidence": float(confidence),
        "probabilities": {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))},
        "images": {
            "original": image_to_base64(original_np),
            "heatmap": image_to_base64(heatmap_colored),
            "overlay": image_to_base64(overlay),
        },
    }


def aggregate_predictions(predictions: list) -> dict:
    """
    Aggregate predictions from multiple scans.

    Args:
        predictions: List of individual prediction dictionaries

    Returns:
        Dictionary with aggregated results including averaged probabilities,
        final prediction, and agreement score
    """
    if not predictions:
        return {}

    # Get all class names from first prediction
    class_names = list(predictions[0]["probabilities"].keys())

    # Calculate average probabilities across all scans
    avg_probabilities = {}
    for class_name in class_names:
        total = sum(pred["probabilities"][class_name] for pred in predictions)
        avg_probabilities[class_name] = total / len(predictions)

    # Find class with highest average probability
    aggregated_class = max(avg_probabilities, key=avg_probabilities.get)
    aggregated_confidence = avg_probabilities[aggregated_class]

    # Calculate agreement score (% of scans predicting the same class as aggregated)
    agreement_count = sum(1 for pred in predictions if pred["class"] == aggregated_class)
    agreement_score = agreement_count / len(predictions)

    return {
        "class": aggregated_class,
        "confidence": aggregated_confidence,
        "probabilities": avg_probabilities,
        "agreement_score": agreement_score,
    }


def image_to_base64(image: np.ndarray) -> str:
    """
    Convert numpy image to base64 string for JSON response.

    Args:
        image: Numpy array (H, W) or (H, W, 3)

    Returns:
        Base64 encoded string
    """
    if len(image.shape) == 2:  # Grayscale
        pil_image = Image.fromarray((image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8))
    else:  # RGB
        pil_image = Image.fromarray(image.astype(np.uint8))

    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return f"data:image/png;base64,{img_str}"


@app.get("/")
async def root():
    """Root endpoint - redirect to docs."""
    return {"message": "Brain Tumor Classification API", "docs": "/docs", "health": "/health"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model = getattr(app.state, "model", None)
    gradcam = getattr(app.state, "gradcam", None)
    device = getattr(app.state, "device", None)
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gradcam_loaded": gradcam is not None,
        "device": device,
        "classes": CLASSES,
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
    model = getattr(app.state, "model", None)
    gradcam = getattr(app.state, "gradcam", None)
    device = getattr(app.state, "device", None)
    transform = getattr(app.state, "transform", None)

    if model is None or gradcam is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        result = _process_single_image(contents, model, gradcam, device, transform)

        response = {
            "success": True,
            "prediction": {
                "class": result["class"],
                "class_index": result["class_index"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
            },
            "images": result["images"],
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)) -> JSONResponse:
    """
    Predict brain tumor type from multiple uploaded MRI images and aggregate results.

    Args:
        files: List of uploaded image files

    Returns:
        JSON with individual predictions and aggregated results
    """
    model = getattr(app.state, "model", None)
    gradcam = getattr(app.state, "gradcam", None)
    device = getattr(app.state, "device", None)
    transform = getattr(app.state, "transform", None)

    if model is None or gradcam is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    print(f"Batch prediction: received {len(files)} files")
    for f in files:
        print(f"  - {f.filename} (type: {f.content_type})")

    individual_predictions = []

    for file in files:
        # Validate file type - be more permissive
        content_type = file.content_type or ""
        filename = file.filename or ""
        is_image = content_type.startswith("image/") or filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")
        )
        if not is_image:
            print(f"Skipping non-image file: {filename} (content_type: {content_type})")
            continue

        try:
            print(f"Processing file: {filename}")
            contents = await file.read()
            result = _process_single_image(contents, model, gradcam, device, transform)
            result["filename"] = file.filename
            individual_predictions.append(result)

        except Exception as e:
            print(f"Error processing {file.filename}: {str(e)}")
            import traceback

            traceback.print_exc()
            individual_predictions.append({"filename": file.filename, "error": str(e)})

    # Filter out failed predictions for aggregation
    successful_predictions = [p for p in individual_predictions if "error" not in p]

    if not successful_predictions:
        raise HTTPException(status_code=400, detail="No valid images could be processed")

    # Aggregate predictions
    aggregated = aggregate_predictions(successful_predictions)

    response = {
        "success": True,
        "batch_size": len(files),
        "processed_count": len(successful_predictions),
        "individual_predictions": individual_predictions,
        "aggregated_prediction": aggregated,
    }

    return JSONResponse(content=response)


# Mount static files (frontend) — html=True enables SPA fallback
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    print("Starting Brain Tumor Classification API...")
    print(f"Visit http://localhost:{settings.port}/ for the web interface")
    print(f"Visit http://localhost:{settings.port}/docs for API documentation")

    uvicorn.run(app, host=settings.host, port=settings.port)
