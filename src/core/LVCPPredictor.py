from typing import Tuple
import matplotlib as plt

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
                ) -> Tuple[str, plt.pyplot]:
        """
        Predict heart elevation using an XGBoost model and explain the prediction with a force plot.

        Args:
            E_A_ratio (float): E/A ratio.
            septal_e_cm_s (float): Septal e' (cm/s).
            septal_E_e_ratio (float): Septal E/e' ratio.
            TRPG_mmHg (float): TRPG (mmHg).
            max_IVC_diameter_mm (float): Max IVC diameter (mm).
            LV_end_diastolic_dimension_mm (float): LV end-diastolic dimension (mm).
            LV_ejection_fraction (float): LV ejection fraction (%).
            LA_dimension_mm (float): LA dimension (mm).
            LA_volume_index_ml_m2 (float): LA volume index (ml/m2).

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

        answer = 'Elevated' if prediction_class == 1 else 'Not elevated'

        answer = answer + f" ({round(class_proba * 100, 1)}%)"

        return answer, force_plot
