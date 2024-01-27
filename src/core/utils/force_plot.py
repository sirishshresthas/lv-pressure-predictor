import os
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

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    # Generate SHAP force plot with Matplotlib
    shap.force_plot(explainer.expected_value,
                    shap_values[0], features, matplotlib=True, show=False)

    plt.rcParams.update({'font.size': 20})
    plt.rcParams.update({'font.weight': 'normal'})

    plt.tight_layout()

    return plt
