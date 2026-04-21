def preprocess_input(soal: str, jawaban: str) -> str:
    soal = soal.lower()
    jawaban = jawaban.lower()
    return f"{soal} {jawaban}"