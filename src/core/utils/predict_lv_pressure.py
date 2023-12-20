from pathlib import Path

import numpy as np
import xgboost as xgb

from .get_model import load_xgboost_model


def predict_heart_elevation(model_dir: Path, model_name: str, input_data: np.array):

    ## load the model
    model = load_xgboost_model(model_dir=model_dir, model_name=model_name)

    # Dummy prediction for illustration
    # input_data = np.array([param1, param2, param3, param4]).reshape(1, -1)
    prediction = model.predict(xgb.DMatrix(input_data))[0]

    # Convert raw predictions to probabilities using the sigmoid function
    probabilities = 1 / (1 + np.exp(-prediction))      

    return prediction, probabilities