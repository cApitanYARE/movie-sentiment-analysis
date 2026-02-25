#.keras
import tensorflow as tf

model_path = 'Model/sentiment_model.keras'
model = tf.keras.models.load_model(model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()
tflite_model_path = 'Model/sentiment_model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)


#.joblib
import json
import joblib

tokenizer = joblib.load("Model/tokenizer.joblib")

word_index = tokenizer.word_index
with open("Model/tokenizer_index.json", "w", encoding="utf-8") as f:
    json.dump(word_index, f, ensure_ascii=False)
