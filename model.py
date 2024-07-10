import keras
from keras import layers, KerasTensor


__all__ = [
    "resnet18",
    "resnet34",
]


resnet_config = {
    "18": (2, 2, 2, 2),
    "34": (3, 4, 6, 3),
}

def residual_block(x, filters, kernel_size=3, stride=1, need_shortcut=False):
    shortcut = x
    if need_shortcut == True:
        shortcut = layers.Conv2D(filters, kernel_size=(1, 1), strides=stride)(x)
        shortcut = layers.BatchNormalization()(shortcut)
    # First convolutional layer
    x = layers.Conv2D(filters, kernel_size=kernel_size, padding="same", strides=stride)(x)
    x  = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # Second convolutional layer
    x = layers.Conv2D(filters, kernel_size=kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)

    # Combine two branches
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


# Defining two shallow residual networks
def __basic_resnet(version, input_shape, num_classes, image_preprocess=None):
    version_control = ("18", "34")
    if version not in version_control:
        raise ValueError("There is no corresponding version of the structure.")
    if not isinstance(input_shape, (tuple, list, KerasTensor)):
        raise TypeError("input_shape must be a tuple or list or keras.KerasTensor")
    if not isinstance(num_classes, int):
        raise TypeError("num_classes must be an integer")
    # Get resnet configuration
    config = resnet_config.get(version)
    
    # Initial convolutional and max pooling layers
    inputs = keras.Input(shape=input_shape)
    x = inputs
    if image_preprocess != None:
        x = image_preprocess(x)
    x = layers.Conv2D(64, kernel_size=7, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
    

    # Stack residual blocks
    for index, numbers in enumerate(config):
        filters = 64 * (2 ** index)
        for j in range(numbers):
            if j == 0 and index > 0:
                x = residual_block(x, filters, stride=2, need_shortcut=True)
            else:
                x = residual_block(x, filters, stride=1, need_shortcut=False)
    
    # Global average pooling and output layers
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.models.Model(inputs, outputs)
    return model


def resnet18(input_shape, num_classes, image_preprocess):
    return __basic_resnet("18", input_shape, num_classes, image_preprocess)

def resnet34(input_shape, num_classes, image_preprocess):
    return __basic_resnet("34", input_shape, num_classes, image_preprocess)

