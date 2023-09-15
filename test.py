#from src.constants.mongo import MONGO_URL, DB_NAME, TRAINING_COLLECTION_NAME,TESTING_COLLECTION_NAME,TRAINING_RAW_DATASET_DIR,TESTING_RAW_DATASET_DIR , INGESTED_TRAINING_RAW_DATASET_DIR, INGESTED_TESTING_RAW_DATASET_DIR
import json
import pandas as pd 
import numpy as np 
from src.data_access.mongo_db import MongoConnect
import os 
from src.components.data_ingestion import DataIngestionComponent
from src.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from src.constants.training_pipeline import TRAINING_SCHEMA_FILE_PATH, DATA_INGESTION_INGESTED_DIR,DATA_INGESTION_DIR,\
                                            DATA_INGESTION_FEATURE_STORE_DIR, MONGO_VALID_COLLECTION_NAME

from src.utility.generic import create_regex, read_json_file
from src.components.data_validation import DataValidationComponent
from src.components.data_transformation import DataTransformationComponent
from config import PROJECT_ROOT
from src.constants.mongo import TRAINING_RAW_DATASET_DIR,TESTING_RAW_DATASET_DIR
from src.constants.mongo import MONGO_URL,DB_NAME

from src.pipeline.training_pipeline import TrainingPipeline 
from src.pipeline.prediction_pipeline import PredictionPipeline

#---------------------------------------------
#---------------------------------------------

"""training_pipeline_config = TrainingPipelineConfig()

data_ingestion_config = DataIngestionConfig(training_pipeline_config)

#ingestion_component = DataIngestionComponent(data_ingestion_config)
#ingestion_component.run_data_ingestion()
#---------------------------------------------
#---------------------------------------------

mongo_connect = MongoConnect()
mongo_connect.mongo_connection(MONGO_URL,DB_NAME)

#dataset_dir = os.path.join(from_root(),"dataset","prediction-dataset")
#training_csv_path_list = [os.path.join(TRAINING_RAW_DATASET_DIR,csv) for csv in os.listdir(TRAINING_RAW_DATASET_DIR)]
#testing_csv_path_list = [os.path.join(TESTING_RAW_DATASET_DIR,csv) for csv in os.listdir(TESTING_RAW_DATASET_DIR)]

training_target_dir = TRAINING_RAW_DATASET_DIR
testing_target_dir = TESTING_RAW_DATASET_DIR
os.makedirs(training_target_dir,exist_ok=True)
print(training_target_dir)

#mongo_connect.insert_with_fs(training_csv_path_list,data_ingestion_config.training_collection_name)
#mongo_connect.insert_with_fs(testing_csv_path_list,data_ingestion_config.unseen_collection_name)

#mongo_connect.ingest_with_fs("C:/Users/ayhan\Desktop/predictive-wafer-maintenance","wafer-valid-training-collection")


mongo_connect.ingest_with_fs(training_target_dir,MONGO_VALID_COLLECTION_NAME)
#mongo_connect.ingest_with_fs(testing_target_dir,TESTING_COLLECTION_NAME)

mongo_connect.client.close()"""


#---------------------------------------------
#---------------------------------------------


"""validation_schema = {
    "SampleFileName": "wafer-raw-training-collection_Wafer_11012020_151432.csv",
	"LengthOfDateStampInFile": 8,
	"LengthOfTimeStampInFile": 6,
	"NumberofColumns" : 592, # 590 sensor feature + 1 Wafer id + 1 target column
    "MissingThreshold":0.5,
    "col_name": {}
}

sensor_names = [f"Sensor-{i}" for i in range (1,591)]

validation_schema["col_name"]["Wafer"] = "varchar"

for sensor_name in sensor_names:
    validation_schema["col_name"][sensor_name]=["int","int64","float","float64"]

validation_schema["col_name"]["Good/Bad"] = "int64"

with open(TRAINING_SCHEMA_FILE_PATH,"w") as schema_file:
    json.dump(validation_schema,schema_file,indent=4)"""


#---------------------------------------------

#---------------------------------------------


""" regex_object = create_regex()
sample_filename = "Wafer-Fault-training-Collection_Wafer_11012021_151432.csv"

match = regex_object.match(sample_filename)

if match:
    date_stamp = match.group("DateStamp")
    time_stamp = match.group("TimeStamp")
    print(f"Date Stamp: {date_stamp}")
    print(f"Time Stamp: {time_stamp}")
else:
    print("Filename does not match the pattern.") """

#---------------------------------------------
#---------------------------------------------

""" training_schema= read_json_file(TRAINING_SCHEMA_FILE_PATH)
col_names = list(training_schema["col_name"].keys())

print(col_names) """

#---------------------------------------------
#---------------------------------------------
"""training_pipeline_config = TrainingPipelineConfig()

data_ingestion_config = DataIngestionConfig(training_pipeline_config)
ingestion_component = DataIngestionComponent(data_ingestion_config)
ingestion_artifact= ingestion_component.run_data_ingestion() 
data_validation_config = DataValidationConfig(training_pipeline_config)
data_validation_component = DataValidationComponent(data_validation_config,ingestion_artifact)
data_validation_component.run_data_validation()"""

#---------------------------------------------
#---------------------------------------------

""" feature_store= os.path.join(PROJECT_ROOT,"artifact","28_08_2023_17_33","data_ingestion","feature_store")

for file_path in os.listdir(feature_store):
    df = pd.read_csv(os.path.join(feature_store,file_path),index_col=0)
    #df = df.rename(columns={df.columns[0]: "Wafer"})  # Rename the first column to "Wafer"
    #df = df.drop("Wafer.1", axis=1, errors="ignore")  # Drop the "Wafer.1" column if it exists
    #df = df.drop(columns=["Unnamed: 0"])
    #df= df.rename(columns={df.columns[0]:"Wafer"})
    #print(df.columns[0])
    df.to_csv(os.path.join(file_path),index=False) """


#---------------------------------------------
#---------------------------------------------
""" 
file_list = os.listdir(feature_store)

if len(file_list) > 0:
    # Take the first file from the list
    first_file_path = os.path.join(feature_store, file_list[0])
    
    # Read the first CSV file
    df = pd.read_csv(first_file_path, header=0)
    
    # Print the columns of the DataFrame
    print(df.columns)
else:
    print("No files found in the directory.") """

"""
#---------------------------------------------
# MAIN PIPELINE
#---------------------------------------------
training_pipeline_config = TrainingPipelineConfig()

data_ingestion_config = DataIngestionConfig(training_pipeline_config)
ingestion_component = DataIngestionComponent(data_ingestion_config)
ingestion_artifact= ingestion_component.run_data_ingestion() 
data_validation_config = DataValidationConfig(training_pipeline_config)
data_validation_component = DataValidationComponent(data_validation_config,ingestion_artifact)
data_validation_artifact = data_validation_component.run_data_validation()
data_transformation_config = DataTransformationConfig(training_pipeline_config)
data_transformation_component = DataTransformationComponent(data_transformation_config,data_validation_artifact)
data_transformation_artifact = data_transformation_component.run_data_transformation()"""
#---------------------------------------------
#---------------------------------------------

#trainin_pipeline = TrainingPipeline()
#trainin_pipeline.run_training_pipeline()

"""prediction_pipeline = PredictionPipeline()
y_pred , best_model_artifact = prediction_pipeline.start_prediction_pipeline()

print(y_pred)
print(best_model_artifact)"""


from pathlib import Path
import fnmatch

def is_excluded(path, dockerignore_patterns):
    for pattern in dockerignore_patterns:
        if fnmatch.fnmatch(path, pattern):
            return True
    return False

def list_files_and_directories(root_dir, dockerignore_path):
    root_path = Path(root_dir)
    all_files = []
    all_dirs = []

    with open(dockerignore_path) as f:
        dockerignore_patterns = [line.strip() for line in f]

    for item in root_path.glob('**/*'):
        item_path = str(item.relative_to(root_path))
        if item.is_file() and not is_excluded(item_path, dockerignore_patterns):
            all_files.append(item)
        elif item.is_dir() and not is_excluded(item_path, dockerignore_patterns):
            all_dirs.append(item)

    return all_files, all_dirs

# Specify the root directory of your project and the path to your .dockerignore file
root_directory = r"C:\Users\ayhan\Desktop\predictive-wafer-maintenance"
dockerignore_path = r"C:\Users\ayhan\Desktop\predictive-wafer-maintenance\.dockerignore"

# Get the list of all files and directories
files, dirs = list_files_and_directories(root_directory, dockerignore_path)

# Print the results
print("Files:")
for file in files:
    print(file)

print("\nDirectories:")
for dir in dirs:
    print(dir)

