import keras
from PIL import Image, ImageOps
#from tensorflow.keras.models import model_from_json
import numpy as np


def predict_cls(img, weights_file):
    # Load the model
    #json_file = open(weights_file)
    #loaded_model_json = json_file.read()
    #json_file.close()
    #model = model_from_json(loaded_model_json)
    model= keras.models.load_model(weights_file)
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    #return np.argmax(prediction) # return position of the highest probability
    return prediction

