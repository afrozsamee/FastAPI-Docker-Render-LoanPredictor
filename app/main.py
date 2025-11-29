from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline, __version__ as model_version
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI(title="Loan Approval API")



# ----------------------------
# Input schema (RAW features)
# ----------------------------
class LoanIn(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str   # <--- categorical
    person_emp_length: float
    loan_intent: str             # <--- categorical
    loan_grade: str              # <--- categorical (A, B, C, ...)
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str  # "Y" / "N"
    cb_person_cred_hist_length: int


# ----------------------------
# Output schema
# ----------------------------
class LoanOut(BaseModel):
    loan_status: int
    probability: float

# Correct absolute path to the static folder
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Mount static folder
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/frontend")
def serve_frontend():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict", response_model=LoanOut)
def predict(payload: LoanIn):

    # Convert Pydantic object → dict
    data = payload.dict()

    # Run model
    pred, prob = predict_pipeline(data)

    return {
        "loan_status": int(pred),
        "probability": float(prob)
    }
