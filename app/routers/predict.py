from fastapi import APIRouter, HTTPException, status
from app.models import model, predictions

import torch
import numpy as np


router = APIRouter()

@router.post("/")
def get_prediction(data: predictions.PredictInput):
    chosen_model = None

    if len(data.features) != 7:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Incorrect number of features. Got {len(data.features)} but expected 7"
        )

    try:
        match data.model:
            case "StandardModel":
                chosen_model = model.StandardModel()

            case "SimpleModel":
                chosen_model = model.SimpleNN()

            case "DeepModel":
                chosen_model = model.DeeperNN()

            case "SuperDeepModel":
                chosen_model = model.SuperDeepNN()

        if chosen_model is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Incorrect model or no model specified, options are: (StandardModel, SimpleNN, DeeperNN, SuperDeepNN)"
            )
        
        features = torch.FloatTensor([data.features])

        with torch.no_grad():
            outputs = chosen_model(features)
            probs = outputs.numpy()[0]
            pred = int(np.argmax(probs))
            confidence = float(np.max(probs))

        predicted_class_name = model.LABEL_MAPPING.get(pred, "unknown")

        return predictions.PredictOutput(
            predicted_class=predicted_class_name,
            model_used=data.model,
            features=data.features,
            confidence=confidence
            )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )



