from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import imagenet_utils


input_shape = (224,224,3)

def load_model():
    model = tf.keras.applications.MobileNetV2(input_shape=input_shape)
    return model

_model = load_model()

def read_image(image_encoded):
    pil_image = Image.open(BytesIO(image_encoded))
    return pil_image

def preprocessor(image: Image.Image):
    image = image.resize(input_shape)
    image = np.asfarray(image)
    image = image / 127.5 - 1.0
    image = np.expand_dims(image,0)
    return image

def predict(image:np.ndarray):
    predictions = _model.predict(image)
    