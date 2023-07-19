from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.models import load_model

model = None

def get_model():
    model = load_model('model/malaria_detector_model.h5')
    print("Model loaded")
    return model

def predict(image: Image.Image):
    global model
    if model is None:
        model = get_model()

    image = np.asarray(image.resize((130, 130)))[..., :3]
    image = np.expand_dims(image, axis=0)
    #image = image / 127.5 - 1.0
    prediction = model.predict(image)
    score = float(prediction[0])

    return f"This cell is {100 * (1 - score):.2f}% infected and {100 * score:.2f}% uninfected."


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image