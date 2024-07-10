import keras
from keras import layers
from keras.api.callbacks import ModelCheckpoint
from keras.api.optimizers import Adam
import matplotlib.pyplot as plt
from datasets import x_train, x_test, y_train, y_test
from model import resnet18
from config import Config


image_augmentation_layers = (
    layers.RandomRotation(factor=Config.ROTATION_FACTOR),
    layers.RandomTranslation(width_factor=Config.TRANSLATION_WIDTH_FACTOR, height_factor=Config.TRANSLATION_HEIGHT_FACTOR),
    layers.RandomFlip(),
    layers.RandomContrast(factor=Config.CONTRAST_FACTOR),
)

def image_preprocess(image):
    for layer in image_augmentation_layers:
        image = layer(image)
    return image


model = resnet18((Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT, 3), Config.NUM_CLASSES, None)
if Config.OUTPUT_SUMMARY:
    model.summary()

# Define model checkpoint to get the optimization model
model_checkpoint = ModelCheckpoint(
    filepath=Config.MODEL_SAVE_PATH,
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode="min",
    save_freq="epoch"
)


model.compile(
    optimizer=Adam(learning_rate=Config.LEARNING_RATE),
    loss= "sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

record = model.fit(
    x_train,
    y_train,    
    epochs=Config.EPOCHS,
    batch_size=Config.TRAINING_BATCH_SIZE,
    validation_data=(x_test, y_test),
    callbacks=[model_checkpoint if Config.USE_CHECKPOINTS else None]
)

# Plot trainging accuracy and losses
def plot_accuracy_and_loss(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    
    epochs = range(len(acc))
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training accuracy')
    plt.plot(epochs, val_acc, label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()

plot_accuracy_and_loss(record)