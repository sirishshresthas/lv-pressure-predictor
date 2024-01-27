from pathlib import Path
from typing import Tuple
import numpy as np
import xgboost as xgb

from .get_model import load_xgboost_model


def predict_heart_elevation(model_dir: Path,
                            model_name: str,
                            input_data: np.array
                            ) -> Tuple[int, float, xgb.Booster]:
    """
    Predict heart elevation using an XGBoost model.

    Args:
        model_dir (Path): The directory containing the XGBoost model file.
        model_name (str): The filename of the XGBoost model to be loaded.
        input_data (np.array): Input data for making predictions.

    Returns:
        Tuple[int, float, xgb.Booster]: A tuple containing:
            - predicted_class (int): The predicted class or label.
            - predicted_class_probability (float): The probability of the predicted class.
            - model (xgb.Booster): The loaded XGBoost model.
    """

    # Validate input_data
    if not isinstance(input_data, np.ndarray):
        raise ValueError("Input data must be a numpy array.")
    if input_data.ndim != 2:
        raise ValueError("Input data must be a 2D array.")

    # load the model
    model = load_xgboost_model(
        model_dir=model_dir,
        model_name=model_name
    )

    try:

        # input_data = np.array([param1, param2, param3, param4]).reshape(1, -1)
        predicted_class = model.predict(input_data)[0]

        # Convert raw predictions to probabilities
        prediction_proba = model.predict_proba(input_data)
        predicted_class_probability = prediction_proba[0, predicted_class]

        return predicted_class, predicted_class_probability, model

    except Exception as e:
        raise RuntimeError(f"An error occurred during prediction: {str(e)}")
