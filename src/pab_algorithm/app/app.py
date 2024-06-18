from fastapi import FastAPI
from pab_algorithm.predictor.predictor import ScorePredictor
import uvicorn

class PabAlgorithmApi(FastAPI):
    def __init__(
        self,
        *args,
        predictor: ScorePredictor | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.predictor = predictor or ScorePredictor()

app = PabAlgorithmApi()

@app.post("/")
def predict(home: str, away: str) -> list:
    h, a = app.predictor.predict(home=home, away=away)
    return [float(h), float(a)]

uvicorn.run(app)