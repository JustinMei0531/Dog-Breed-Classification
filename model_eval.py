import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from datasets import x_test, y_test
from config import Config

# Load model
model = keras.models.load_model(Config.MODEL_SAVE_PATH)

pred = model.predict(x_test)
pred_classes = np.argmax(pred, axis=1)

# Get confusion matrix
conf_matrix = confusion_matrix(y_test, pred_classes)

display = ConfusionMatrixDisplay(conf_matrix)
display.plot(cmap="Blues")
plt.show()