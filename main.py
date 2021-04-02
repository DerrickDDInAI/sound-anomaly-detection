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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import IPython.display as ipd


# ============================================================
# Main functions
# ============================================================

def load_model(filename: str) -> GridSearchCV:
    """
    Function to load a machine learning model 
    from core/assets/data directory
    * param: filename of model
    """
    current_script_file = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(
        current_script_file, "core", "assets", "data", filename)

    return joblib.load(model_path)


def main() -> None:
    """
    Main function 
    """
    # Get all sounds
    current_script_file = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(current_script_file, "core",
                     "assets", "data", "thread_csv_all.csv"))

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
    loaded_model: GridSearchCV = load_model("best_model_all_features.joblib")
    print(f"Best parameters: {loaded_model.best_params_}")

    y_pred: np.ndarray = loaded_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Create a DataFrame with true and predicted labels, keep the original index (to retrieve the sound)
    df_true_pred_test = pd.DataFrame(np.column_stack([y_test, y_pred]), index=y_test.index, columns=["y_test", "y_pred"])
    
    # Get all sounds predicted as abnormal (class 1)
    y_pred_abnormal: np.ndarray = df_true_pred_test[df_true_pred_test.y_pred == 1]

    # Choose a random sound from predicted abnormals
    random_idx: int = random.choice(y_pred_abnormal.index)
    random_sound: pd.DataFrame = df.loc[[random_idx]]
    print(f"machine type: {random_sound.machine_type.values[0]}") # .values[0] to print the value and not a list
    print(f"model id: {random_sound.model_id.values[0]}")
    print(f"noise db: {random_sound.noise_db.values[0]}")
    print(f"sound: {random_sound.sound.values[0]}")
    print(f"true label: {random_sound.target.values[0]}")

   # Ask the user if he wants to listen the predicted abnormal and normal sound of the machine
    listen: str = input("""
    Would you like to listen 
    - the current sound of the machine
    AND 
    - the pre-recorded sound of the machine when it is normal
    (Please type "yes" or "y" to listen)
    """).lower()

    if listen in ("yes", "y"):
        
        # Listen to current sound of machine
        print("Current sound")
        current_sound_path = os.path.join("core",
                     *df.loc[[random_idx]].sound_path.values[0].split("\\"))
        ipd.display(ipd.Audio(current_sound_path)) # ipd.display() to display multiple Audio objects at once

        # Listen to pre-recorded normal sound
        print("Pre-recorded normal sound")
        normal_sound = df.loc[
            (df.machine_type == random_sound.machine_type.values[0]) &  
            (df.noise_db == 0) & 
            (df.model_id == 0) & 
            (df.target == 0) & 
            (df.sound == "00000000.wav")
        ].sound_path.values[0].split("\\")
        normal_sound_path = os.path.join("core",
                     *normal_sound)
        ipd.display(ipd.Audio(normal_sound_path))

# ============================================================
# Run
# ============================================================
# Executes the main() function if this file is directly run
if __name__ == "__main__":
    main()
