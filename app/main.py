from fastapi import FastAPI, HTTPException
from app.serving import MultiModelAPI
from app.schemas import PredictionRequest  # import the schema from your app folder

app = FastAPI()

# Initialize the multi-model loader
model_api = MultiModelAPI(model_dir="models")


@app.get("/")
async def read_root() -> dict:
    return {"message": "Welcome to the Multi-Model ML API"}


@app.post("/predict")
async def predict(request: PredictionRequest) -> dict:
    try:
        prediction = model_api.predict(request.model_name, request.features)
        return {"model": request.model_name, "prediction": prediction}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
