def preprocess_input(soal: str, jawaban: str) -> str:
    return f"Soal: {soal} [SEP] Jawaban: {jawaban}"