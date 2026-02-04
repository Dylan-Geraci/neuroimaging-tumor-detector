"""
FastAPI backend for brain tumor classification web application.

Provides REST API endpoints for:
- Health checking
- Image upload and prediction
- Grad-CAM visualization generation
"""

import io
import uuid
import base64
import time
from contextlib import asynccontextmanager

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query, Request
from typing import List, Optional
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image
from sqlalchemy.orm import Session

from src.model import create_model
from src.data import CLASSES
from src.predict import get_inference_transforms
from src.gradcam import create_gradcam
from src.config import settings
from src.database import engine, get_db, Base
from src.models_db import Prediction
<<<<<<< HEAD
from src.logger import setup_logger, log_request, log_error
from src.rate_limit import RateLimiter
from src.validation import validate_upload, validate_batch_upload

# Initialize logger
logger = setup_logger(level=settings.log_level)

# Initialize rate limiter
rate_limiter = RateLimiter(requests_per_minute=settings.rate_limit_per_minute)
=======
from src.auth import verify_api_key
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from src.rate_limit import limiter, rate_limit_handler
from src.logger import logger
from src.validation import validate_file_upload, validate_batch_upload
from src.model_loader import load_model_checkpoint
>>>>>>> 40c1d5a3f6c3f560c834e5adff95c2c15a0df926


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and Grad-CAM on startup, clean up on shutdown."""
<<<<<<< HEAD
    logger.info("Starting Brain Tumor Classification API...")
=======
    logger.info("Loading model...")
>>>>>>> 40c1d5a3f6c3f560c834e5adff95c2c15a0df926

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info(f"Using device: {device}")

    # Load model
    model = create_model(num_classes=len(CLASSES), pretrained=False, device=device)
    checkpoint = load_model_checkpoint(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(f"Model loaded from {settings.model_path}")
    logger.info(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}")

    # Initialize Grad-CAM
    gradcam = create_gradcam(model, device)
    logger.info("Grad-CAM initialized")

    # Get transforms
    transform = get_inference_transforms()
    logger.info("Ready to accept requests!")

    # Create database tables
    Base.metadata.create_all(bind=engine)

    # Store on app.state
    app.state.model = model
    app.state.gradcam = gradcam
    app.state.device = device
    app.state.transform = transform

    yield

    # Cleanup
    logger.info("Shutting down...")


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
    allow_credentials=True if settings.cors_origins != ["*"] else False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key"],
)

logger.info(f"CORS enabled for: {settings.cors_origins}")
if settings.is_production and "*" in settings.cors_origins:
    logger.warning("SECURITY WARNING: CORS wildcard in production!")

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with timing information."""
    start_time = time.time()

    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000

        log_request(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        return response

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        log_error(e, context=f"{request.method} {request.url.path}")

        log_request(
            method=request.method,
            path=request.url.path,
            status_code=500,
            duration_ms=duration_ms,
        )

        raise


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



@app.get("/health")
@limiter.limit("60/minute")
async def health_check(request: Request):
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
<<<<<<< HEAD
=======
@limiter.limit("10/minute")
>>>>>>> 40c1d5a3f6c3f560c834e5adff95c2c15a0df926
async def predict(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
<<<<<<< HEAD
=======
    _: str = Depends(verify_api_key),
>>>>>>> 40c1d5a3f6c3f560c834e5adff95c2c15a0df926
) -> JSONResponse:
    """
    Predict brain tumor type from uploaded MRI image.

    Args:
        request: FastAPI request object (for rate limiting)
        file: Uploaded image file
        db: Database session

    Returns:
        JSON with prediction, confidence, probabilities, and visualizations
    """
    # Rate limiting
    await rate_limiter.check_rate_limit(request)

    model = getattr(app.state, "model", None)
    gradcam = getattr(app.state, "gradcam", None)
    device = getattr(app.state, "device", None)
    transform = getattr(app.state, "transform", None)

    if model is None or gradcam is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
<<<<<<< HEAD
        # Validate and read file
        contents = await validate_upload(file)

=======
        contents = await validate_file_upload(file)
>>>>>>> 40c1d5a3f6c3f560c834e5adff95c2c15a0df926
        result = _process_single_image(contents, model, gradcam, device, transform)

        # Save to database
        db.add(Prediction(
            filename=file.filename or "unknown",
            predicted_class=result["class"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
        ))
        db.commit()

        logger.info(f"Prediction: {result['class']} ({result['confidence']:.2%}) for {file.filename}")

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

    except HTTPException:
        raise
    except Exception as e:
<<<<<<< HEAD
        log_error(e, context="Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
=======
        logger.error(f"Prediction failed for {file.filename}", exc_info=True)
        if settings.is_production:
            raise HTTPException(status_code=500, detail="Internal server error")
        else:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
@limiter.limit("5/minute")
>>>>>>> 40c1d5a3f6c3f560c834e5adff95c2c15a0df926
async def predict_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
<<<<<<< HEAD
=======
    _: str = Depends(verify_api_key),
>>>>>>> 40c1d5a3f6c3f560c834e5adff95c2c15a0df926
) -> JSONResponse:
    """
    Predict brain tumor type from multiple uploaded MRI images and aggregate results.

    Args:
        request: FastAPI request object (for rate limiting)
        files: List of uploaded image files
        db: Database session

    Returns:
        JSON with individual predictions and aggregated results
    """
    # Rate limiting
    await rate_limiter.check_rate_limit(request)

    model = getattr(app.state, "model", None)
    gradcam = getattr(app.state, "gradcam", None)
    device = getattr(app.state, "device", None)
    transform = getattr(app.state, "transform", None)

    if model is None or gradcam is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

<<<<<<< HEAD
    try:
        # Validate batch
        validated_files = await validate_batch_upload(files)

        logger.info(f"Batch prediction: processing {len(validated_files)} files")
=======
    await validate_batch_upload(files)

    logger.info(f"Batch prediction: received {len(files)} files")
    for f in files:
        logger.debug(f"  - {f.filename} (type: {f.content_type})")
>>>>>>> 40c1d5a3f6c3f560c834e5adff95c2c15a0df926

        individual_predictions = []

<<<<<<< HEAD
        for file, contents in validated_files:
            try:
                logger.debug(f"Processing file: {file.filename}")
                result = _process_single_image(contents, model, gradcam, device, transform)
                result["filename"] = file.filename
                individual_predictions.append(result)

            except Exception as e:
                log_error(e, context=f"Error processing {file.filename}")
                individual_predictions.append({"filename": file.filename, "error": str(e)})

        # Filter out failed predictions for aggregation
        successful_predictions = [p for p in individual_predictions if "error" not in p]

        if not successful_predictions:
            raise HTTPException(status_code=400, detail="No valid images could be processed")

        # Save successful predictions to database
        batch_id = str(uuid.uuid4())
        for pred in successful_predictions:
            db.add(Prediction(
                filename=pred.get("filename", "unknown"),
                predicted_class=pred["class"],
                confidence=pred["confidence"],
                probabilities=pred["probabilities"],
                batch_id=batch_id,
            ))
        db.commit()

        # Aggregate predictions
        aggregated = aggregate_predictions(successful_predictions)

        logger.info(
            f"Batch {batch_id}: {len(successful_predictions)}/{len(files)} successful. "
            f"Aggregated: {aggregated['class']} ({aggregated['confidence']:.2%})"
        )

        response = {
            "success": True,
            "batch_size": len(files),
            "processed_count": len(successful_predictions),
            "individual_predictions": individual_predictions,
            "aggregated_prediction": aggregated,
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        log_error(e, context="Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
=======
    for file in files:
        try:
            logger.debug(f"Processing file: {file.filename}")
            contents = await validate_file_upload(file)
            result = _process_single_image(contents, model, gradcam, device, transform)
            result["filename"] = file.filename
            individual_predictions.append(result)

        except Exception as e:
            logger.error(f"Error processing {file.filename}", exc_info=True)
            error_detail = "Processing failed" if settings.is_production else str(e)
            individual_predictions.append({"filename": file.filename, "error": error_detail})

    # Filter out failed predictions for aggregation
    successful_predictions = [p for p in individual_predictions if "error" not in p]

    if not successful_predictions:
        raise HTTPException(status_code=400, detail="No valid images could be processed")

    # Save successful predictions to database
    batch_id = str(uuid.uuid4())
    for pred in successful_predictions:
        db.add(Prediction(
            filename=pred.get("filename", "unknown"),
            predicted_class=pred["class"],
            confidence=pred["confidence"],
            probabilities=pred["probabilities"],
            batch_id=batch_id,
        ))
    db.commit()

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
>>>>>>> 40c1d5a3f6c3f560c834e5adff95c2c15a0df926


@app.get("/predictions")
def list_predictions(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """List prediction history (paginated, newest first)."""
    rows = (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    total = db.query(Prediction).count()
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "items": [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "filename": r.filename,
                "predicted_class": r.predicted_class,
                "confidence": r.confidence,
                "probabilities": r.probabilities,
                "batch_id": r.batch_id,
            }
            for r in rows
        ],
    }


@app.get("/predictions/{prediction_id}")
def get_prediction(prediction_id: str, db: Session = Depends(get_db)):
    """Get a single prediction by ID."""
    row = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return {
        "id": row.id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "filename": row.filename,
        "predicted_class": row.predicted_class,
        "confidence": row.confidence,
        "probabilities": row.probabilities,
        "batch_id": row.batch_id,
    }


@app.delete("/predictions/{prediction_id}")
def delete_prediction(
    prediction_id: str,
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """Delete a prediction by ID."""
    row = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Prediction not found")
    db.delete(row)
    db.commit()
    return {"deleted": True}


# Mount static files (frontend) — html=True enables SPA fallback
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Brain Tumor Classification API...")
    logger.info(f"Visit http://localhost:{settings.port}/ for the web interface")
    logger.info(f"Visit http://localhost:{settings.port}/docs for API documentation")

    uvicorn.run(app, host=settings.host, port=settings.port)
