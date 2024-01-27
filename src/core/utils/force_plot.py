from pathlib import Path
from typing import List
import xgboost as xgb

import matplotlib.pyplot as plt
import numpy as np
import shap

ROOT_DIR: Path = Path.cwd()
MODEL_DIR: Path = Path.joinpath(ROOT_DIR, "model")


def explain_model(model: xgb.XGBModel, input_data: np.array, features: List[str]) -> plt:
    """
    Explain an XGBoost machine learning model using SHAP values and generate a force plot.

    Args:
        model (xgb.XGBModel): The XGBoost machine learning model to explain.
        input_data (np.array): Input data for which explanations are calculated.
        features (List[str]): List of feature names corresponding to the input data.

    Returns:
        plt: Matplotlib plot object containing the SHAP force plot.
    """

    # Error checking
    if not isinstance(model, xgb.XGBModel):
        raise ValueError("Provided model is not a valid XGBoost model.")

    if not isinstance(input_data, np.ndarray) or input_data.size == 0:
        raise ValueError("Input data must be a non-empty numpy array.")

    if not features or not all(isinstance(feature, str) for feature in features):
        raise ValueError("Features must be a non-empty list of strings.")

    if input_data.shape[1] != len(features):
        raise ValueError(
            "The number of features must match the number of columns in the input data.")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)

        # Generate SHAP force plot with Matplotlib
        shap.force_plot(explainer.expected_value,
                        shap_values[0], 
                        features, 
                        plot_cmap='DrDb',
                        matplotlib=True, 
                        show=False)

        plt.rcParams.update({'font.size': 20})
        plt.rcParams.update({'font.weight': 'normal'})

        plt.tight_layout()

        return plt

    except Exception as e:
        raise RuntimeError(
            f"An error occurred while explaining the model: {str(e)}")
