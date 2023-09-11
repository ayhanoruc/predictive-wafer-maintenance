import os 
from config import PROJECT_ROOT


#common constants related to training pipeline

PIPELINE_NAME = "training_pipeline"
TARGET_COLUMN = "Good/Bad"
ARTIFACT_DIR_NAME = "artifact" # directory for outputs from each stage
TRAINING_SCHEMA_FILE_PATH = os.path.join(PROJECT_ROOT,"DSA","training_schema.json")
PREDICTION_SCHEMA_FILE_PATH = os.path.join(PROJECT_ROOT,"DSA","prediction_schema.json") # no need to store here, not related to training pipeline
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

# constants related to DataTransformation component

DATA_TRANSFORMATION_DIR= "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR = "transformed"
DATA_TRANSFORMATION_OBJECT_DIR = "preprocessor_object"
PREPROCESSOR_OBJECT_FILE_NAME = "preprocessor.pkl"
TRAIN_TEST_SPLIT_RATIO = 0.2


# constants related to ModelTrainer Component

MODEL_TRAINER_DIR_NAME = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME = "model.pkl"
MODEL_TRAINER_EXPECTED_ROC_AUC_SCORE = 0.6
MODEL_TRAINER_EXPECTED_COST_SCORE = 140
MODEL_TRAINER_OVER_UNDER_THRESHOLD = 0.05 # %5



