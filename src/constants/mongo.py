import os 
from from_root import from_root
# MongoDB access related constants

MONGO_URL = 'mongodb://localhost:27017/'
DB_NAME = "local"
COLLECTION_NAME = "wafer-fault-collection"


# DataIngestion related constants

TRAINING_RAW_DATASET_DIR = os.path.join(from_root(),"Desktop","predictive-wafer-maintenance","dataset","training-dataset")
TESTING_RAW_DATASET_DIR = os.path.join(from_root(),"Desktop","predictive-wafer-maintenance","dataset","testing-dataset")
INGESTED_TRAINING_RAW_DATASET_DIR = "ingested_data_collection2/training"
INGESTED_TESTING_RAW_DATASET_DIR = "ingested_data_collection2/testing"