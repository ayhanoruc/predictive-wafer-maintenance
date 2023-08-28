import os 
from config import PROJECT_ROOT
# MongoDB access related constants

MONGO_URL = 'mongodb://localhost:27017/'
DB_NAME = "local"
TRAINING_COLLECTION_NAME = "wafer-fault-training-collection"
TESTING_COLLECTION_NAME = "wafer-fault-testing-collection"


# DataIngestion related constants

TRAINING_RAW_DATASET_DIR = os.path.join(PROJECT_ROOT,"dataset","training-dataset")
TESTING_RAW_DATASET_DIR = os.path.join(PROJECT_ROOT,"dataset","prediction-dataset")
INGESTED_TRAINING_RAW_DATASET_DIR = "ingested_data_collection2/training"
INGESTED_TESTING_RAW_DATASET_DIR = "ingested_data_collection2/testing"
