import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.config import MODEL_PATH, DEVICE


class ModelService:
    def __init__(self):
        self.tokenizer = None
        self.model = None

    def load_model(self):
        print("🔄 Loading IndoBERT...")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH
        )

        self.model.to(DEVICE)
        self.model.eval()

        print("✅ Model ready!")

    def predict(self, text: str) -> float:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Ambil nilai skor
        score = logits.squeeze().item()

        score = torch.sigmoid(torch.tensor(score)).item()
        score = score * 100

        return float(score)


# global object (penting biar tidak load berulang)
model_service = ModelService()