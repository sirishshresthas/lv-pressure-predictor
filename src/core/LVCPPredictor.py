from pathlib import Path
from typing import List

import numpy as np

from src.core.utils import explain_model, predict_heart_elevation

ROOT_DIR: Path = Path.cwd()
MODEL_DIR: Path = Path.joinpath(ROOT_DIR, "model")

class LVCPredictor: 

    def __init__(self, model_name):
        self.model_name = model_name

    def predict(self, param1: int, param2: int, param3: int, param4: int):

        input_data = np.array([param1, param2, param3, param4], dtype=object).reshape(1, -1)

        figure = explain_model(model_dir=MODEL_DIR, model_name=self.model_name, input_data=input_data)

        prediction, proba = predict_heart_elevation(model_dir=MODEL_DIR, model_name=self.model_name, input_data=input_data)

        answer = 'Elevated' if prediction == 1 else 'Not elevated'

        answer = answer + f" ({round(proba, 2) * 100}%)"

        return answer, figure

