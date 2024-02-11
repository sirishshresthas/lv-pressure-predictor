from typing import Tuple

import matplotlib as plt
import numpy as np

from src.core.utils import (FEATURE_NAMES, MODEL_DIR, explain_model,
                            predict_heart_elevation)


class LVCPredictor:

    def __init__(self, model_name):
        self.model_name = model_name

    def predict(self,
                TTE_EbyA: float,
                TTE_Epr_sep: float,
                TTE_EbyEpr_sep: float,
                TTE_TRPG: float,
                TTE_LAVI: float,
                TTE_IVCmax: float,
                TTE_Dd: float,
                TTE_LVEF: float,
                TTE_LAd: float
                ) -> Tuple[str, plt.pyplot]:
        """
        Predict heart elevation using an XGBoost model and explain the prediction with a force plot.

        Args:
            TTE_EbyA (float): E/A ratio.
            TTE_Epr_sep (float): Septal e' (cm/s).
            TTE_EbyEpr_sep (float): Septal E/e' ratio.
            TTE_TRPG (float): TRPG (mmHg).
            TTE_LAVI (float): LA volume index (ml/m2). 
            TTE_IVCmax (float): Max IVC diameter (mm).
            TTE_Dd (float): LV end-diastolic dimension (mm).
            TTE_LVEF (float): LV ejection fraction (%).
            TTE_LAd (float): LA dimension (mm).

        Returns:
            Tuple[str, 'matplotlib.pyplot']: A tuple containing:
                - answer (str): Predicted label with probability percentage.
                - force_plot: Matplotlib plot object for SHAP force plot.
        """

        # Create a numpy array from the input data
        input_data = np.array([[
            TTE_EbyA,
            TTE_Epr_sep,
            TTE_EbyEpr_sep,
            TTE_TRPG,
            TTE_LAVI,
            TTE_IVCmax,
            TTE_Dd,
            TTE_LVEF,
            TTE_LAd
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
