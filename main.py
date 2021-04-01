"""
Python program to 
1. Load sounds of factory machines
2. Use a machine learning model 
   to predict whether the machine is defective (output: normal O or abnormal 1)
   from the sound (input)
3. If the machine is predicted to be abnormal: as a first check, the user can listen
   - the pre-recorded sound of the machine when it is normal
   - the current sound of the machine

See README file for more information.
"""
# =====================================================================
# Import
# =====================================================================

# Import internal modules
import os.path
import joblib
import time
import random
from typing import List, Set, Dict, TypedDict, Tuple, Optional

# Import 3rd party modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import seaborn as sns


# Import local modules
from core.sound import get_all_sounds


# ============================================================
# Main functions
# ============================================================

def main() -> None:
    """
    Main function 
    """
    # Get all sounds
    df = pd.read_csv(os.path.join("assets", "data", "thread_csv_all.csv"))

    # Feature engineering
    # Replace abnormal by 1, normal by 0
    df.target = df.target.apply(lambda x: 1 if x == "abnormal" else 0)

    # Feature selection
    # Select numeric columns
    selected_cols = df.select_dtypes(include="number").columns.tolist()

    # Drop noise_db and model_id columns
    selected_cols.remove("noise_db")
    selected_cols.remove("model_id")
    selected_cols.remove("target")

    # Select features X and target variable y
    X = df[selected_cols]
    y = df.target

    # Split into the same training and test sets
    # as we did when training our models
    # to test the best machine learning model on unseen data (test set)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=y
                                                        )

    print(f"X_train: {X_train.shape}   - y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}    - y_train: {y_test.shape}")

    # Load best model
    current_script_file = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(
        current_script_file, "core", "assets", "data", "best_model_all_features.joblib")
    loaded_model = joblib.load(model_path)

    print(loaded_model)
    print(f"Best parameters: {loaded_model.best_params_}")
    print(classification_report(y_test, loaded_model.predict(X_test)))


# ============================================================
# Run
# ============================================================

# Executes the main() function if this file is directly run
if __name__ == "__main__":
    main()
