import os 
from config import PROJECT_ROOT


#common constants related to training pipeline

PIPELINE_NAME = "training_pipeline"
TARGET_COLUMN = "Good/Bad"
ARTIFACT_DIR_NAME = "artifact" # directory for outputs from each stage
SCHEMA_FILE_PATH = os.path.join(PROJECT_ROOT,"DSA","schema.json")
SCHEMA_DROP_COLS = "drop_cols"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"

# constants related to DataIngestion component

TRAINING_COLLECTION_NAME = "wafer-fault-training-collection"
UNSEEN_COLLECTION_NAME = "wafer-fault-unseen-collection"
DATA_INGESTION_DIR_NAME = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR_NAME = "feature_store"
DATA_INGESTION_INGESTED_DIR_NAME = "ingested"  
DATA_INGESTION_SPLIT_RATIO = 0.25

