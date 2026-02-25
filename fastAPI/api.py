
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
import tflite_runtime.interpreter as tflite
import joblib
import uvicorn

import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

import json

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

interpreter = tflite.Interpreter(model_path="Model/sentiment_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("Model/tokenizer_index.json", "r", encoding="utf-8") as f:
    word_index = json.load(f)

def texts_to_sequences_manual(text, word_index):
    words = text.split()
    sequence = [word_index.get(w, 0) for w in words]
    return [sequence]

stop_words = set(stopwords.words('english'))

def clean_text(t):
    text = t.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'-', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)

    text = ' '.join(word for word in word_tokenize(text) if word not in stop_words)

    return text

class inputData(BaseModel):
    text: str = Field(...)

    @field_validator("text")
    def text_not_empty(cls, t):
        if not t.strip():
            raise ValueError("Text cannot be empty")
        return t

@app.get("/",summary='API test',description='Simple check that the API works as expected')
def root():
    return {"message": "Keras model API running"}

@app.post("/predict"
          ,summary='Text sentiment prediction',
          description = 'Takes a text and returns the probability of a positive sentiment and the class (‘positive’ or ‘negative’).',
          response_description= 'Sentiment prediction result')
def predict(data: inputData):
    clear_text = clean_text(data.text)

    seq = texts_to_sequences_manual(clear_text, word_index)

    max_len = 200
    padded = np.zeros((32, max_len), dtype=np.float32)
    for i, val in enumerate(seq[0][:max_len]):
        padded[0, i] = val

    interpreter.set_tensor(input_details[0]['index'], padded)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"

    return {
        "score": float(prediction[0][0]),
        "sentiment": sentiment,
    }

#if __name__ == "__main__":
#    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)