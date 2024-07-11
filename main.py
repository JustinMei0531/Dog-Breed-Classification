import os
import cv2
import keras
import numpy as np
from keras.api.ops import expand_dims
import argparse
from config import Config


def predict(image_path):
    # Read an image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Resize the image
    image = cv2.resize(image, (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT))
    # Add one dimension to the image to fix the batch size
    image = expand_dims(image, axis=0)

    # Load the pre-trained model
    model = keras.models.load_model(Config.MODEL_SAVE_PATH)
    # Make a prediction
    prediction = model.predict(image)
    # Set dog classes
    classes = (
        "Beagle", "Boxer", "Bulldog", "Dachshund", "German_Shepherd", "Golden_Retriever", "Labrador_Retriever",
        "Poodle", "Rottweiler", "Yorkshire_Terrier"
    )
    index = np.argmax(prediction)
    print("Predicted dog breed: {}".format(classes[index]))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dog Breed Classification")
    parser.add_argument("image_path", type=str, help="Please input image file path")
    args = parser.parse_args()

    predict(args.image_path)
