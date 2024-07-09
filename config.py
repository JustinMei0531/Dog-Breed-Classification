class Config:
    # Define dataset path
    DATASET_PATH = "./dataset"

    # Image parameters
    IMAGE_WIDTH = 299
    IMAGE_HEIGHT = 299
    # The number of image categories
    NUM_CLASSES = 10

    # Data generator parameters
    TRAINING_BATCH_SIZE = 10
    VALIDATION_BATCH_SIZE = 4
    NEED_SHUFFLE = True # Whether shuffle the dataset
    RANDOW_SEED = 1 # random seed for data generator

    # Model parameters
    EPOCHS = 20 # Training epochs
    USE_CHECKPOINTS = True # Whether use breakpoint on the model
    LEARNING_RATE = 1e-4 # Learning rate
    MODEL_NAME = "dog_breed.keras" # Saved model name
    