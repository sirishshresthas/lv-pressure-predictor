import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import shap

ROOT_DIR: Path = Path.cwd()
MODEL_DIR: Path = Path.joinpath(ROOT_DIR, "model")


def explain_model(model, input_data:np.array, features): 

    # Create a shapely waterfall plot
    # input_data = input_data.astype(np.float64)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)


    # # Plot the waterfall chart
    # shap_fig = shap.force_plot(
    #     explainer.expected_value, 
    #     shap_values[0], features, matplotlib=True, show=False)

    # shap_fig.savefig('shap.png')

    # Generate SHAP force plot with Matplotlib
    shap.force_plot(explainer.expected_value, shap_values[0], features, matplotlib=True, show=False)

    plt.rcParams.update({'font.size': 20})
    plt.rcParams.update({'font.weight': 'normal'})

    plt.tight_layout()

    return plt



