import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from config import Config

__all__ = ["x_train", "y_train", "x_test", "y_test"]


labels = os.listdir(Config.DATASET_PATH)

images = []
classes = []
for label in labels:
    for file in os.listdir(Config.DATASET_PATH + "/" + label):
        image_path = Config.DATASET_PATH + "/" + label + "/" + file
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT))
        images.append(image)
        classes.append(label)


images = np.array(images)
classes = np.array(classes)

encoder = LabelEncoder()
classes = encoder.fit_transform(classes)

x_train, x_test, y_train, y_test = train_test_split(images, classes, test_size=.2, shuffle=True, random_state=42)