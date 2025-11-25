from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline, __version__ as model_version

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
