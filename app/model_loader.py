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
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

        self.model.to(DEVICE)
        self.model.eval()

        print("✅ Model ready!")
        print("num_labels:", self.model.config.num_labels)
        print("problem_type:", self.model.config.problem_type)

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

        score = logits.squeeze().item()
        score = score * 10
        return float(score)


model_service = ModelService()