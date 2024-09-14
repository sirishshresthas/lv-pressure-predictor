from typing import Tuple
import matplotlib as plt
from matplotlib.figure import Figure

import numpy as np

from src.core.utils import explain_model, predict_heart_elevation, FEATURE_NAMES, MODEL_DIR


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
                ) -> Tuple[str, plt.Figure]:
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

        # Determine if the prediction is elevated or not
        answer = 'Not elevated' if prediction_class == 1 else 'Elevated'

        # Calculate the percentage and round to one decimal place
        percentage = round((1 - class_proba) * 100, 1)

        # Format the answer string with the rounded percentage
        answer = f"{answer} ({percentage}%)"

        return answer, force_plot
