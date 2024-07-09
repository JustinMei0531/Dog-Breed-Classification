import keras
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from config import Config

__all__ = ["train_gen", "validation_gen"]


# Create data generator
data_gen = ImageDataGenerator(
    rescale = 1.0 / 255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    validation_split = 0.2
)

train_gen = data_gen.flow_from_directory(
    Config.DATASET_PATH,
    target_size=(Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=Config.TRAINING_BATCH_SIZE,
    shuffle=Config.NEED_SHUFFLE,
    seed=Config.RANDOW_SEED,
    subset="training"
)

validation_gen = data_gen.flow_from_directory(
    Config.DATASET_PATH,
    target_size=(Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=Config.VALIDATION_BATCH_SIZE,
    subset = "validation"
)