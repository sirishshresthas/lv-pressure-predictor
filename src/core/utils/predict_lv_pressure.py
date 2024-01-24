from pathlib import Path

import numpy as np

from .get_model import load_xgboost_model


def predict_heart_elevation(model_dir: Path, model_name: str, input_data: np.array):

    # load the model
    model = load_xgboost_model(
        model_dir=model_dir,
        model_name=model_name
    )

    # Dummy prediction for illustration
    # input_data = np.array([param1, param2, param3, param4]).reshape(1, -1)
    predicted_class = model.predict(input_data)

    # Convert raw predictions to probabilities using the sigmoid function
    prediction_proba = model.predict_proba(input_data)
    predicted_class_probability = prediction_proba[0, predicted_class[0]]

    return predicted_class, predicted_class_probability, model
