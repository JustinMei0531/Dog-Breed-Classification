class Config:
    # Define dataset path
    DATASET_PATH = "./dataset"
    VALIDATION_SET_PATH = "./static/val"

    # Image parameters
    IMAGE_WIDTH = 224 # Input image width
    IMAGE_HEIGHT = 224 # Input image height
    # ROTATION_FACTOR = 0.15 # Image rotation factor
    # TRANSLATION_WIDTH_FACTOR = 0.1
    # TRANSLATION_HEIGHT_FACTOR = 0.1
    # CONTRAST_FACTOR = 0.1
    # H_FLIP = True
    # V_FLIP = True
    # The number of image categories
    NUM_CLASSES = 10

    # Data generator parameters
    TRAINING_BATCH_SIZE = 16
    VALIDATION_BATCH_SIZE = 8
    NEED_SHUFFLE = True # Whether shuffle the dataset
    RANDOM_SEED = 1 # random seed for data generator

    # Model parameters
    MODEL_SAVE_PATH = "./dog_breed.keras" # Model saved path
    OUTPUT_SUMMARY = True # Whetehr output model summary
    EPOCHS = 20 # Training epochs
    USE_CHECKPOINTS = True # Whether use breakpoint on the model
    LEARNING_RATE = 1e-4 # Learning rate
    PLOT_AFTER_TRAINING = False
    