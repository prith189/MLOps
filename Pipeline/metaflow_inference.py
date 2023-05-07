import numpy as np
from fastapi import FastAPI
from metaflow import Flow
from pydantic import BaseModel

flow = Flow('TrainFlow').latest_successful_run

best_model = flow.data.best_model

class InferenceRequest(BaseModel):
    input_vector: list

app = FastAPI()

@app.post("/infer")
def infer(request: InferenceRequest):
    input_vector = np.array([request.input_vector])
    prediction = best_model.predict(input_vector)
    return {"prediction": int(prediction[0])}
