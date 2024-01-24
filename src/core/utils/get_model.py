from pathlib import Path

import xgboost as xgb


def load_xgboost_model(model_dir: Path, model_name:str):
    """
    Load an XGBoost model from the specified path.

    Returns:
        xgb.Booster: The loaded XGBoost model.

    Raises:
        FileNotFoundError: If the model file is not found at the specified path.
    """
    
    # Define the path to the XGBoost model file using pathlib
    model_path:Path = Path.joinpath(model_dir, model_name)

    # Check if the model file exists
    if not model_path.is_file():
        raise FileNotFoundError(f"XGBoost model file not found at: {model_path}")

    # Load the XGBoost model
    bst = xgb.XGBClassifier()
    bst.load_model(model_path)

    return bst

