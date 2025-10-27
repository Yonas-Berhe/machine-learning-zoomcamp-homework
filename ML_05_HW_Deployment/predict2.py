import pickle
from typing import Literal
from pydantic import BaseModel, Field


from fastapi import FastAPI
import uvicorn



class Customer(BaseModel):
    lead_source: Literal['organic', 'paid_ads', 'referral', 'social_media']
    number_of_courses_viewed: int = Field(ge=0, le=20)
    annual_income: float = Field(ge=0)




class PredictResponse(BaseModel):
    convert_probability: float
    convert: bool


app = FastAPI(title="customer-conversion-prediction")

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)


@app.post("/predict2")
def predict(customer: Customer) -> PredictResponse:
    prob = predict_single(customer.model_dump())

    return PredictResponse(
        convert_probability=prob,
        convert=prob >= 0.5
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9090)