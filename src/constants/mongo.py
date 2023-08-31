import os 
from config import PROJECT_ROOT
# MongoDB access related constants

MONGO_URL = 'mongodb://localhost:27017/'
DB_NAME = "local"
TRAINING_COLLECTION_NAME = "wafer-fault-training-collection"
TESTING_COLLECTION_NAME = "wafer-fault-testing-collection"


# DataIngestion related constants

TRAINING_RAW_DATASET_DIR = os.path.join(PROJECT_ROOT,"valid_training")
#TRAINING_RAW_DATASET_DIR = os.path.join(PROJECT_ROOT,"dataset","training-dataset")
TESTING_RAW_DATASET_DIR = os.path.join(PROJECT_ROOT,"dataset","prediction-dataset")
