import uvicorn
from test2 import Inference
from fastapi import FastAPI, Path
from typing import Optional
from pydantic import BaseModel


app = FastAPI()


# @app.get('/{text}')
# def home(text: str = Path(None, description='Insert text')):
#     return {"name": "First test"}


@app.get('/')
def get_text(text: str):
    i = Inference()
    data = [text]
    acc, data_frame, classifier = i.train(0.2)
    sentiment = i.predict(data_frame, classifier, data)
    return {text: sentiment, 'accuracy': acc}
