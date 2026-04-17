from pydantic import BaseModel

class EssayRequest(BaseModel):
    soal: str
    jawaban: str

class EssayResponse(BaseModel):
    skor: float