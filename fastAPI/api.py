
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
import tensorflow as tf
import joblib
import uvicorn

import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = tf.keras.models.load_model('./Model/sentiment_model.keras')
tokenizer = joblib.load("./Model/tokenizer.joblib")

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
    clear_text = [clean_text(data.text)]

    seq = tokenizer.texts_to_sequences(clear_text)
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        seq, maxlen=200, padding='post', truncating='post'
    )

    prediction = model.predict(padded)

    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"

    return {
        "score": float(prediction[0][0]),
        "sentiment": sentiment,
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)