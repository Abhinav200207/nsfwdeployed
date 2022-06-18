from fastapi import FastAPI,UploadFile,File
import uvicorn
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

app = FastAPI()

def load_model():
    pretrained_model = tf.keras.models.load_model("NSFW_Abhinav_Final.h5", custom_objects=None, compile=True, options=None)
    return pretrained_model
pretrained_model = load_model()



@app.get("/index")
def hello_world(name):
    return f"Hello {name}"


def predict(image):
    predictions = pretrained_model.predict(image)
    return np.round(predictions)

@app.post("/api/predict")
async def predict_image():
    image = plt.imread("https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/Shaqi_jrvej.jpg/1200px-Shaqi_jrvej.png")
    image = tf.keras.preprocessing.image.smart_resize(image,size=(250,250),interpolation='nearest')
    image = image.reshape(1,250,250,3)
    predictions = predict(image)
    print(predictions)
    if(predictions[0][0] != 0):
        return 'SFW'
    else:
        return 'NSFW'


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="localhost")