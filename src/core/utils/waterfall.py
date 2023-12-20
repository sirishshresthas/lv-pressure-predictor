import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import shap
import xgboost as xgb

from .get_model import load_xgboost_model


def explain_model(model_dir:Path, model_name:str, input_data:np.array): 

    waterfall_png: str = str(Path.joinpath(model_dir, "shap_plot.png"))

    if not os.path.exists(waterfall_png):

        model = load_xgboost_model(model_dir=model_dir,model_name=model_name)
        print(model)

        # Create a shapely waterfall plot
        # input_data = input_data.astype(np.float64)
        explainer = shap.Explainer(model, input_data)
        shap_values = explainer(input_data)

        # Plot the waterfall chart
        shap.waterfall_plot(shap_values[0,0])
        plt.title("SHAP Values Waterfall Plot")
        plt.xlabel("SHAP Value")
        plt.tight_layout()
        plt.savefig(waterfall_png)
        
        return waterfall_png
    
    else: 
        return waterfall_png

