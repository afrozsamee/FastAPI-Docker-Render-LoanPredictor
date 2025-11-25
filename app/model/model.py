# app/model/model.py
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import pickle
from custom_preprocessor import CustomPreprocessor
import pandas as pd
import os


__version__ = "0.1.0"
BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/trained_loanpipeline-{__version__}.pkl", "rb") as f:
    model = pickle.load(f)  # now pickle can find CustomPreprocessor

def predict_pipeline(data: dict):
    df = pd.DataFrame([data])
    # Predict
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return pred, prob
