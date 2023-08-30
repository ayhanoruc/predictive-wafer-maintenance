import os 
from config import PROJECT_ROOT


#common constants related to training pipeline

PIPELINE_NAME = "training_pipeline"
TARGET_COLUMN = "Good/Bad"
ARTIFACT_DIR_NAME = "artifact" # directory for outputs from each stage
TRAINING_SCHEMA_FILE_PATH = os.path.join(PROJECT_ROOT,"DSA","training_schema.json")
PREDICTION_SCHEMA_FILE_PATH = os.path.join(PROJECT_ROOT,"DSA","prediction_schema.json")
SCHEMA_DROP_COLS = "drop_cols"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"

# constants related to DataIngestion component

TRAINING_COLLECTION_NAME = "wafer-raw-training-collection"
UNSEEN_COLLECTION_NAME = "wafer-raw-unseen-collection"
DATA_INGESTION_DIR = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"
DATA_INGESTION_INGESTED_DIR = "ingested"  
DATA_INGESTION_SPLIT_RATIO = 0.25


# constants related to DataValidation component 
DATA_VALIDATION_DIR = "data_validation"
DATA_VALIDATION_TRAINING_DIR = "training"
DATA_VALIDATION_UNSEEN_DIR = "unseen"
DATA_VALIDATION_VALID_DIR = "valid"
DATA_VALIDATION_INVALID_DIR = "invalid"
MONGO_VALID_COLLECTION_NAME = "wafer-valid-training-collection"
MONGO_INVALID_COLLECTION_NAME = "wafer-invalid-training_collection"