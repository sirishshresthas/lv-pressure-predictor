from pathlib import Path

import xgboost as xgb


def load_xgboost_model(model_dir: Path, model_name: str) -> xgb.Booster:
    """
    Load an XGBoost model from the specified path.

    Args:
        model_dir (Path): The directory containing the XGBoost model file.
        model_name (str): The filename of the XGBoost model to be loaded.

    Returns:
        xgb.Booster: The loaded XGBoost model.

    Raises:
        FileNotFoundError: If the model file is not found at the specified path.
    """

    # path to the XGBoost model file
    model_path: Path = model_dir / model_name

    # Check if the model file exists
    if not model_path.is_file():
        raise FileNotFoundError(
            f"XGBoost model file not found at: {model_path}")

    # Load the XGBoost model
    bst = xgb.XGBClassifier()
    bst.load_model(str(model_path))

    return bst
