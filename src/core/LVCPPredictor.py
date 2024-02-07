from pathlib import Path
from typing import Tuple

import numpy as np

from src.core.utils import explain_model, predict_heart_elevation

ROOT_DIR: Path = Path.cwd()
MODEL_DIR: Path = Path.joinpath(ROOT_DIR, "model")

FEATURE_NAMES: list = ["E/A ratio",
                       "septal e' (cm/s)",
                       "septal E/e' ratio",
                       "TRPG (mmHg)",
                       "max IVC diameter (mm)",
                       "LV end-diastolic dimension (mm)",
                       "LV ejection fraction (%)",
                       "LA dimension (mm)",
                       "LA volume index (ml/m2)"]


class LVCPredictor:

    def __init__(self, model_name):
        self.model_name = model_name

    def predict(self, E_A_ratio: float, septal_e_cm_s: float, septal_E_e_ratio: float, TRPG_mmHg: float, max_IVC_diameter_mm: float, LV_end_diastolic_dimension_mm: float, LV_ejection_fraction: float, LA_dimension_mm: float, LA_volume_index_ml_m2: float) -> Tuple[np.ndarray, np.ndarray]:

        # Create a numpy array from the input data
        input_data = np.array([[
            E_A_ratio,
            septal_e_cm_s,
            septal_E_e_ratio,
            TRPG_mmHg,
            LA_volume_index_ml_m2,
            max_IVC_diameter_mm,
            LV_end_diastolic_dimension_mm,
            LV_ejection_fraction, 
            LA_dimension_mm
        ]])

        prediction_class, class_proba, model = predict_heart_elevation(
            model_dir=MODEL_DIR,
            model_name=self.model_name,
            input_data=input_data
        )

        # explainer
        force_plot = explain_model(model, input_data, FEATURE_NAMES)

        answer = 'Elevated' if prediction_class == 0 else 'Not elevated'

        answer = answer + f" ({round(class_proba * 100, 1)}%)"

        return answer, force_plot
