import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError

# Initialize FastAPI app
app = FastAPI(
    title="BigQuery ML Credit Card Default Prediction Service",
    description="API to predict credit card default using a trained BQML model.",
    version="1.0.0"
)

# Initialize BigQuery client (implicitly picks up credentials from the environment)
try:
    bq_client = bigquery.Client()
except Exception as e:
    raise RuntimeError(f"Failed to initialize BigQuery Client. Ensure GOOGLE_APPLICATION_CREDENTIALS is set. Error: {e}")

# Target configuration: Change these variables to point to your actual trained model
PROJECT_ID = bq_client.project
DATASET_ID = "abisaill_ml"  # e.g., "credit_card_analysis"
MODEL_NAME = "default_risk_v1"  # Your trained BQML model name


# Define the input schema based on bigquery-public-data.ml_datasets.credit_card_default columns
class PredictionInput(BaseModel):
    limit_balance: float = Field(..., description="Amount of the given credit (NT dollar)", example=50000.0)
    sex: str = Field(..., description="Gender (1 = male; 2 = female)", example="1")
    education_level: str = Field(..., description="Education Level (1 = graduate school; 2 = university; 3 = high school; 4 = others)", example="2")
    marital_status: str = Field(..., description="Marital status (1 = married; 2 = single; 3 = others)", example="1")
    age: int = Field(..., description="Age in years", example=35)
    pay_0: float = Field(..., description="Repayment status in September", example=0.0)
    pay_2: float = Field(..., description="Repayment status in August", example=0.0)
    pay_3: float = Field(..., description="Repayment status in July", example=0.0)
    pay_4: float = Field(..., description="Repayment status in June", example=0.0)
    pay_5: float = Field(..., description="Repayment status in May", example=0.0)
    pay_6: float = Field(..., description="Repayment status in April", example=0.0)
    bill_amt_1: float = Field(..., description="Amount of bill statement in September", example=10000.0)
    bill_amt_2: float = Field(..., description="Amount of bill statement in August", example=9500.0)
    bill_amt_3: float = Field(..., description="Amount of bill statement in July", example=9200.0)
    bill_amt_4: float = Field(..., description="Amount of bill statement in June", example=8000.0)
    bill_amt_5: float = Field(..., description="Amount of bill statement in May", example=7500.0)
    bill_amt_6: float = Field(..., description="Amount of bill statement in April", example=7000.0)
    pay_amt_1: float = Field(..., description="Amount paid in September", example=2000.0)
    pay_amt_2: float = Field(..., description="Amount paid in August", example=2000.0)
    pay_amt_3: float = Field(..., description="Amount paid in July", example=2000.0)
    pay_amt_4: float = Field(..., description="Amount paid in June", example=1500.0)
    pay_amt_5: float = Field(..., description="Amount paid in May", example=1500.0)
    pay_amt_6: float = Field(..., description="Amount paid in April", example=1500.0)


# Define output response schema
class PredictionProbability(BaseModel):
    label: int
    prob: float

class PredictionResponse(BaseModel):
    predicted_default_payment_next_month: int
    probabilities: List[PredictionProbability]


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Simple check to make sure the service is up."""
    return {"status": "healthy", "model_configured": f"{PROJECT_ID}.{DATASET_ID}.{MODEL_NAME}"}


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_default(payload: PredictionInput):
    """
    Accepts client data, queries BigQuery ML using ML.PREDICT, 
    and returns the classification outcome along with probability distributions.
    """
    
    # 1. Structure the parameterized query safely to mitigate injection risks.
    # Note: ML.PREDICT takes the model identifier and a inner SELECT query structure serving row inputs.
    query = f"""
        SELECT 
            predicted_default_payment_next_month,
            predicted_default_payment_next_month_probs
        FROM 
            ML.PREDICT(
                MODEL `{PROJECT_ID}.{DATASET_ID}.{MODEL_NAME}`,
                (
                    SELECT 
                        @limit_balance AS limit_balance,
                        @sex AS sex,
                        @education_level AS education_level,
                        @marital_status AS marital_status,
                        @age AS age,
                        @pay_0 AS pay_0,
                        @pay_2 AS pay_2,
                        @pay_3 AS pay_3,
                        @pay_4 AS pay_4,
                        @pay_5 AS pay_5,
                        @pay_6 AS pay_6,
                        @bill_amt_1 AS bill_amt_1,
                        @bill_amt_2 AS bill_amt_2,
                        @bill_amt_3 AS bill_amt_3,
                        @bill_amt_4 AS bill_amt_4,
                        @bill_amt_5 AS bill_amt_5,
                        @bill_amt_6 AS bill_amt_6,
                        @pay_amt_1 AS pay_amt_1,
                        @pay_amt_2 AS pay_amt_2,
                        @pay_amt_3 AS pay_amt_3,
                        @pay_amt_4 AS pay_amt_4,
                        @pay_amt_5 AS pay_amt_5,
                        @pay_amt_6 AS pay_amt_6
                )
            )
    """

    # 2. Map payload attributes to BigQuery query parameters
    query_params = [
        bigquery.ScalarQueryParameter(k, "FLOAT64" if isinstance(v, float) else ("INT64" if isinstance(v, int) else "STRING"), v)
        for k, v in payload.dict().items()
    ]

    job_config = bigquery.QueryJobConfig(query_parameters=query_params)

    try:
        # 3. Execute the sync call to BigQuery
        query_job = bq_client.query(query, job_config=job_config)
        results = query_job.result()  # Wait for query execution to complete
        
        # Convert row iterator to list
        rows = list(results)
        if not rows:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Model processing returned zero records."
            )
            
        first_row = rows[0]
        
        # 4. Format the array structure returned by BigQuery classification probabilities
        # BQML natively outputs probabilistic layouts as an array of structs: [{'label': 1, 'prob': 0.82}, ...]
        raw_probs = first_row.get("predicted_default_payment_next_month_probs", [])
        formatted_probs = [
            PredictionProbability(label=int(p["label"]), prob=float(p["prob"])) 
            for p in raw_probs
        ]
        
        return PredictionResponse(
            predicted_default_payment_next_month=int(first_row.get("predicted_default_payment_next_month")),
            probabilities=formatted_probs
        )

    except GoogleAPIError as bq_err:
        # Catch specific underlying Google Cloud exceptions
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"BigQuery Engine Error: {bq_err.message}"
        )
    except Exception as err:
        # Catch generic parsing or operational structural fallouts
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal processing error occurred: {str(err)}"
        )

if __name__ == "__main__":
    import uvicorn
    # Start app locally on port 8000
    uvicorn.run("main.py:app", host="0.0.0.0", port=8000, reload=True)
