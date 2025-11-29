from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline, __version__ as model_version
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

app = FastAPI(title="Loan Approval API")

# Input schema
class LoanIn(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int

# Output schema
class LoanOut(BaseModel):
    loan_status: int
    probability: float

# Fix static directory path
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Serve frontend at root
@app.get("/")
def serve_frontend():
    return FileResponse(str(STATIC_DIR / "index.html"))

# Health check endpoint
@app.get("/health")
def health():
    return {"health_check": "OK", "model_version": model_version}

# Prediction endpoint
@app.post("/predict", response_model=LoanOut)
def predict(payload: LoanIn):
    data = payload.dict()
    pred, prob = predict_pipeline(data)
    return {
        "loan_status": int(pred),
        "probability": float(prob)
    }