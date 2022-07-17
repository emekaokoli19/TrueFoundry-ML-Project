from infer import Inference
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


class Data(BaseModel):
    text: str


@app.post("/")
def predict_text(data: Data):
    model = Inference()
    to_read = [data.text]
    acc, data_frame, classifier = model.train(0.2)
    sentiment = model.predict(data_frame, classifier, to_read)
    return {data.text: sentiment, 'accuracy': acc}
