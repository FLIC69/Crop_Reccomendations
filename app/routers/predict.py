from fastapi import APIRouter, HTTPException, status, Security
from app.models import model, predictions
from app.utils import api_key
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import joblib

import torch
import numpy as np

router = APIRouter()

@router.post("/")
def get_prediction(data: predictions.PredictInput, api_key: str = Security(api_key.get_api_key)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    chosen_model = None

    if len(data.features) != 7:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Incorrect number of features. Got {len(data.features)} but expected 7"
        )

    try:
        if data.model ==  "StandardModel":
            chosen_model = model.StandardModel()

        if data.model ==  "SimpleModel":
            chosen_model = model.SimpleNN()

        if data.model ==  "DeepModel":
            chosen_model = model.DeeperNN()

        if data.model ==  "SuperDeepModel":
            chosen_model = model.SuperDeepNN()


        if chosen_model is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Incorrect model or no model specified, options are: (StandardModel, SimpleNN, DeeperNN, SuperDeepNN)"
            )
                
        chosen_model.eval()
        chosen_model.to(device)
        
        features = torch.FloatTensor([data.features])
        scaler = joblib.load("app/ai_models/scaler.pkl")
        encoded_features = torch.FloatTensor(scaler.transform(features))

        with torch.no_grad():
            # Forward pass to get model outputs
            outputs = chosen_model(encoded_features)
            
            # Calculate probabilities using softmax
            probs = F.softmax(outputs, dim=1).numpy()[0]  # Get first batch item if batch size > 1
            
            # Get predicted class index and confidence
            pred = int(np.argmax(probs))
            confidence = float(np.max(probs))
            
            # Get class name from mapping
            predicted_class_name = model.LABEL_MAPPING.get(pred, "unknown")

        return predictions.PredictOutput(
            predicted_class=predicted_class_name,
            model_used=data.model,
            features=data.features,
            confidence=confidence,
            )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )