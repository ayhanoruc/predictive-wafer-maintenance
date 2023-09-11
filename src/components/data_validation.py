import pandas as pd 

import os 
import shutil

from src.exception_handler import CustomException, handle_exceptions
from src.log_handler import AppLogger
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.utility.generic import read_json_file
from src.constants.training_pipeline import TRAINING_SCHEMA_FILE_PATH
from src.utility.generic import create_regex, check_regex_match
from src.data_access.mongo_db import MongoConnect
from src.constants.training_pipeline import MONGO_VALID_COLLECTION_NAME, MONGO_INVALID_COLLECTION_NAME
from src.constants.mongo import MONGO_URL,DB_NAME


class DataValidationComponent:

    def __init__(self,data_validation_config:DataValidationConfig, data_ingestion_artifact:DataIngestionArtifact):
        self.data_validation_config = data_validation_config
        self.data_ingestion_artifact = data_ingestion_artifact
        self.training_schema = read_json_file(TRAINING_SCHEMA_FILE_PATH) # data sharring agreement schema
        self.sample_file_name = self.training_schema["SampleFileName"]
        self.date_length = self.training_schema["LengthOfDateStampInFile"]
        self.timestamp_length= self.training_schema["LengthOfTimeStampInFile"]
        self.cols= self.training_schema["col_name"]
        self.missing_threshold= self.training_schema["MissingThreshold"]

        #self.nu_of_cols= self.training_schema["NumberofColumns"]
        #self.col_names = set(self.training_schema["col_name"].keys())
        #self.col_dtypes = set(self.training_schema["col_name"].values())



        self.log_writer = AppLogger("DataValidation")

        self.mongo_connect = MongoConnect()
        self.mongo_connect.mongo_connection(MONGO_URL,DB_NAME)

        

    
    @handle_exceptions
    def validate_file_name(self,file_path)->bool:
        
        file_name= os.path.basename(file_path)
        self.log_writer.handle_logging(f"Filename validation stage for {file_name}")
        re_obj= create_regex()
        return check_regex_match(re_obj,file_name) # returns boolean
    
    

    """@handle_exceptions
    def validate_columns(self,file_path):
        df = pd.read_csv(file_path)
        
  
        
        for col_name, expected_dtype in self.cols.items():
            if col_name not in df.columns:

                print("not found")
                return False 
            if df[col_name].dtype != expected_dtype:
                return False 
            print(f"col_name {col_name} found")
        return True"""


    @handle_exceptions
    def validate_columns_2(self,file_path):
        df = pd.read_csv(file_path)
        self.log_writer.handle_logging(f"Column number and dtype validation stage for {os.path.basename(file_path)}")
        df_columns= set(df.columns)
        expected_columns = set(self.cols.keys())
        missing_cols = expected_columns - df_columns
        if missing_cols:
            print("missing cols: ",missing_cols)
            return False 
        for col_name, expected_dtype in self.cols.items():
            if str(df[col_name].dtype) not in expected_dtype:
                print(col_name,df[col_name].dtype, expected_dtype,": are missing")
                return False
        
        return True


    @handle_exceptions
    def check_missing_value_percentage(self,file_path):
        df = pd.read_csv(file_path)
        self.log_writer.handle_logging(f"Missing Value Check stage for {os.path.basename(file_path)}")
        missing_percentage = (df.isna().sum().sum())/(df.shape[0]*df.shape[1]) # for each column
        self.log_writer.handle_logging(f"Missing value % {missing_percentage}")
        
        
        #print(any(above_threshold))
        """if any(above_threshold):
            result = {col: above_thresh for col, above_thresh in above_threshold.items()}
            #then log the results here """
            
        return missing_percentage > self.missing_threshold

    @handle_exceptions
    def check_zero_std(self,file_path):
        df= pd.read_csv(file_path)
        df_std = df.iloc[:,1:-1].std()
        zero_std_cols = df_std[df_std==0].index 
        """if any(zero_std_cols):
            pass 
            #log here
        #return any(zero_std_cols)"""
        return False
    


    @handle_exceptions
    def data_drift_check(self):
        return True 

    
    @handle_exceptions
    def check_all_conditions(self, file_path):
        self.log_writer.handle_logging("Validation Check Stages is started...")
        if self.validate_file_name(file_path):
            self.log_writer.handle_logging(f"Filename Validated!")
            if self.validate_columns_2(file_path):
                self.log_writer.handle_logging(f"Columns Validated!")
                if not self.check_missing_value_percentage(file_path):
                    self.log_writer.handle_logging(f"Missing Value Validation Passed!")

                    if not self.check_zero_std(file_path):
                        print("check_zero")
                        valid_data_path = self.data_validation_config.valid_training_data_dir
                        os.makedirs(valid_data_path,exist_ok=True)
                        print(os.path.basename(file_path))
                        shutil.move(file_path,os.path.join(valid_data_path,os.path.basename(file_path))) # öncesinde copy yapıp sonra move da edilebilir.
                    
                    else:
                        #print("invalid")
                        invalid_data_path = self.data_validation_config.invalid_training_data_dir
                        os.makedirs(invalid_data_path,exist_ok=True)
                        shutil.move(file_path,os.path.join(invalid_data_path,os.path.basename(file_path)))

                else:
                    self.log_writer.handle_logging(f"Missing Value Percentage is above the permitted threshold!")
                    invalid_data_path = self.data_validation_config.invalid_training_data_dir
                    os.makedirs(invalid_data_path,exist_ok=True)
                    shutil.move(file_path,os.path.join(invalid_data_path,os.path.basename(file_path))) 
                    self.log_writer.handle_logging(f"File {os.path.basename(file_path)} is moved to invalid data directory")       


            else:
                self.log_writer.handle_logging(f"Columns are not validated")
                invalid_data_path = self.data_validation_config.invalid_training_data_dir
                os.makedirs(invalid_data_path,exist_ok=True)
                shutil.move(file_path,os.path.join(invalid_data_path,os.path.basename(file_path)))
                self.log_writer.handle_logging(f"File {os.path.basename(file_path)} is moved to invalid data directory") 

        else:
            self.log_writer.handle_logging(f"Filename is not validated")
            invalid_data_path = self.data_validation_config.invalid_training_data_dir
            os.makedirs(invalid_data_path,exist_ok=True)
            shutil.move(file_path,os.path.join(invalid_data_path,os.path.basename(file_path)))
            self.log_writer.handle_logging(f"File {os.path.basename(file_path)} is moved to invalid data directory")



    @handle_exceptions
    def post_validation_insertion(self):
        valid_files_path_list = [os.path.join(self.data_validation_config.valid_training_data_dir,file) for file in os.listdir(self.data_validation_config.valid_training_data_dir)]
        invalid_files_path_list = [os.path.join(self.data_validation_config.invalid_training_data_dir,file) for file in os.listdir(self.data_validation_config.invalid_training_data_dir)]
        
        self.log_writer.handle_logging("INSERTING VALID FILES TO MONGODB")
        self.mongo_connect.insert_with_fs(valid_files_path_list,MONGO_VALID_COLLECTION_NAME)

        self.log_writer.handle_logging("INSERTING INVALID FILES TO MONGODB")
        self.mongo_connect.insert_with_fs(invalid_files_path_list,MONGO_INVALID_COLLECTION_NAME)
        self.log_writer.handle_logging("POST VALIDATION INSERTION COMPLETED SUCCESFULLY!")



    @handle_exceptions
    def post_validation_deletion(self):
        self.log_writer.handle_logging("DELETING VALID & INVALID FILES FROM LOCAL!")
        shutil.rmtree(self.data_validation_config.valid_training_data_dir)
        shutil.rmtree(self.data_validation_config.invalid_training_data_dir)
        self.log_writer.handle_logging("DELETION SUCCESFULL!")
        

        


    @handle_exceptions
    def run_data_validation(self):
        
        #feature_store_dir = "C:/Users/ayhan\Desktop/predictive-wafer-maintenance/artifact/30_08_2023_14_02/data_ingestion/feature_store"
        self.log_writer.handle_logging("-------------ENTERED DATA VALIDATION STAGE------------")

        feature_store_dir = self.data_ingestion_artifact.feature_store
        csv_path_list = os.listdir(feature_store_dir)

        for csv_path in csv_path_list:
            file_path = os.path.join(feature_store_dir,csv_path)
            self.check_all_conditions(file_path)
        
        self.post_validation_insertion()
        self.post_validation_deletion()
        
        data_validation_artifact = DataValidationArtifact(
            valid_data_dir= self.data_validation_config.valid_training_data_dir ,
            invalid_data_dir= self.data_validation_config.invalid_training_data_dir
        )   
        return data_validation_artifact
    