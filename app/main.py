from fastapi import FastAPI, HTTPException
from app.schemas import EssayRequest, EssayResponse
from app.model_loader import model_service
from app.inference import preprocess_input

app = FastAPI()

@app.on_event("startup")
def load_model():
    print("🚀 Starting app...")
    model_service.load_model()

@app.get("/")
def root():
    return {"message": "API jalan 🚀"}

@app.post("/predict", response_model=EssayResponse)
def predict(request: EssayRequest):
    try:
        text = preprocess_input(request.soal, request.jawaban)
        score = model_service.predict(text)
        return {"skor": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))