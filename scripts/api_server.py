"""
FastAPI Server for AG News Classification

This module provides:
- RESTful API endpoints for inference
- Batch prediction support
- Request validation
- Health checks
- Comprehensive logging
"""

import logging
from typing import List, Optional, Dict
import yaml
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from inference import NewsClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
def load_config(config_path: str = "config/config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# Initialize FastAPI app
app = FastAPI(
    title="AG News Classification API",
    description="Production-grade API for news classification using fine-tuned DistilBERT",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize classifier
classifier = None


# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    """Single prediction request."""
    text: str = Field(..., min_length=1, max_length=5000, description="News text to classify")
    return_probabilities: bool = Field(True, description="Return full probability distribution")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence threshold")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of news texts")
    return_probabilities: bool = Field(True, description="Return full probability distribution")
    batch_size: int = Field(32, ge=1, le=128, description="Processing batch size")


class PredictionResponse(BaseModel):
    """Single prediction response."""
    text: str
    category: str
    category_id: int
    confidence: float
    probabilities: Optional[Dict[str, float]] = None


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    processing_time: float
    total_samples: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str


@app.on_event("startup")
async def startup_event():
    """Initialize classifier on startup."""
    global classifier
    logger.info("Starting up API server...")
    
    try:
        model_path = config['checkpoint']['save_dir']
        if Path(model_path).exists():
            classifier = NewsClassifier(model_path, "config/config.yaml")
            logger.info("Classifier initialized successfully")
        else:
            logger.warning(f"Model not found at {model_path}")
    except Exception as e:
        logger.error(f"Error initializing classifier: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API server...")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status and model information
    """
    return HealthResponse(
        status="healthy" if classifier is not None else "degraded",
        model_loaded=classifier is not None,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Single prediction endpoint.
    
    Args:
        request: Prediction request with text
        
    Returns:
        Prediction with category and confidence
        
    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        # Get prediction
        result = classifier.predict(request.text)
        
        # Apply threshold if specified
        if request.threshold is not None:
            if result['confidence'] < request.threshold:
                logger.warning(
                    f"Prediction confidence {result['confidence']:.4f} "
                    f"below threshold {request.threshold}"
                )
        
        # Format response
        return PredictionResponse(
            text=result['text'],
            category=result['category'],
            category_id=result['category_id'],
            confidence=result['confidence'],
            probabilities=result['probabilities'] if request.return_probabilities else None,
        )
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction endpoint.
    
    Args:
        request: Batch prediction request with multiple texts
        
    Returns:
        Batch predictions with processing time
        
    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        import time
        start_time = time.time()
        
        # Get batch predictions
        results = classifier.predict_batch(request.texts, batch_size=request.batch_size)
        
        # Format responses
        predictions = [
            PredictionResponse(
                text=result['text'],
                category=result['category'],
                category_id=result['category_id'],
                confidence=result['confidence'],
                probabilities=result['probabilities'] if request.return_probabilities else None,
            )
            for result in results
        ]
        
        processing_time = time.time() - start_time
        
        return BatchPredictionResponse(
            predictions=predictions,
            processing_time=processing_time,
            total_samples=len(request.texts),
        )
    
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/categories")
async def get_categories():
    """
    Get available news categories.
    
    Returns:
        Dictionary mapping category IDs to names
    """
    return {"categories": config['class_labels']}


@app.get("/model-info")
async def get_model_info():
    """
    Get model information.
    
    Returns:
        Model configuration and details
    """
    return {
        "model_name": config['model']['name'],
        "num_classes": config['model']['num_classes'],
        "max_length": config['model']['max_length'],
        "categories": config['class_labels'],
    }


@app.post("/predict-with-threshold")
async def predict_with_threshold(
    text: str,
    threshold: float = 0.5
):
    """
    Prediction with confidence threshold.
    
    Args:
        text: News text
        threshold: Confidence threshold
        
    Returns:
        Prediction if confidence >= threshold, else None
    """
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        result = classifier.predict_with_threshold(text, threshold)
        
        if result is None:
            return {
                "prediction": None,
                "message": f"Confidence below threshold {threshold}"
            }
        
        return {
            "prediction": PredictionResponse(
                text=result['text'],
                category=result['category'],
                category_id=result['category_id'],
                confidence=result['confidence'],
                probabilities=result['probabilities'],
            ),
            "message": "Prediction successful"
        }
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/top-k-predictions")
async def top_k_predictions(
    text: str,
    k: int = 3
):
    """
    Get top-k predictions.
    
    Args:
        text: News text
        k: Number of top predictions
        
    Returns:
        Top-k predictions sorted by confidence
    """
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    if k < 1 or k > 4:
        raise HTTPException(
            status_code=400,
            detail="k must be between 1 and 4"
        )
    
    try:
        top_k = classifier.get_top_k_predictions(text, k)
        return {"top_k_predictions": top_k}
    
    except Exception as e:
        logger.error(f"Error during top-k prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Top-k prediction failed: {str(e)}"
        )


def main():
    """Main function to run the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AG News API Server")
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    parser.add_argument('--log-level', default='info', help='Log level')
    
    args = parser.parse_args()
    
    logger.info(f"Starting API server on {args.host}:{args.port}")
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
