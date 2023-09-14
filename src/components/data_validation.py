import pandas as pd 

from src.exception_handler import CustomException, handle_exceptions
from src.log_handler import AppLogger
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.utility.generic import create_regex, check_regex_match , read_json_file
from src.data_access.mongo_db import MongoConnect
from src.constants.training_pipeline import (MONGO_VALID_COLLECTION_NAME, MONGO_INVALID_COLLECTION_NAME,
                                              TRAINING_SCHEMA_FILE_PATH, PREDICTION_SCHEMA_FILE_PATH)
from src.constants.mongo import MONGO_URL,DB_NAME

import os 
import shutil


class DataValidationComponent:

    def __init__(self,data_validation_config:DataValidationConfig, data_ingestion_artifact:DataIngestionArtifact):

        # Initialize the logger, configuration, and other attributes
        self.log_writer = AppLogger("DataValidation")

        self.data_validation_config = data_validation_config
        self.data_ingestion_artifact = data_ingestion_artifact

        # Load schemas and other parameters from JSON files
        self.training_schema = read_json_file(TRAINING_SCHEMA_FILE_PATH) # data sharing agreement schema for training
        self.log_writer.handle_logging("read TRAINING SCHEMA succesfully")
        self.sample_file_name = self.training_schema["SampleFileName"]
        #self.date_length = self.training_schema["LengthOfDateStampInFile"]
        self.timestamp_length= self.training_schema["LengthOfTimeStampInFile"]
        self.training_cols= self.training_schema["col_name"]
        self.missing_threshold= self.training_schema["MissingThreshold"]

        self.prediction_schema = read_json_file(PREDICTION_SCHEMA_FILE_PATH) # data sharing agreement schema for prediction
        self.log_writer.handle_logging("read PREDICTION SCHEMA succesfully")
        self.prediction_cols = self.prediction_schema["col_name"]

        self.re_obj = None

        self.df = None

        #connect to mongodb
        try:
            self.mongo_connect = MongoConnect()
            self.mongo_connect.mongo_connection(MONGO_URL,DB_NAME)

        except Exception as e:
            CustomException("Couldnt succesfully connected to mongodb",e) 


        

    
    @handle_exceptions
    def validate_file_name(self,file_path)->bool: 
        """
        Validates the filename using a regular expression.

        Args:
            file_path (str): The path of the file to validate.

        Returns:
            bool: True if the filename matches the expected pattern, False otherwise.
        """
        file_name= os.path.basename(file_path)
        self.log_writer.handle_logging(f"Filename validation stage for {file_name}")

        #re_obj= create_regex(is_prediction) # this can be turned into instance object for the datavalidation class
        #self.log_writer.handle_logging("Related regex pattern compiled succesfully .")

        return check_regex_match(self.re_obj,file_name) # returns boolean, [re_obj -> self.re_obj]
    
    

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
    def validate_columns_2(self,file_path, is_prediction=False):
        """
        Validates the columns and data types of a DataFrame loaded from a file.

        Args:
            file_path (str): The path of the CSV file to validate.
            is_prediction (bool, optional): True if validating prediction data, False for training data. Defaults to False.

        Returns:
            bool: True if the columns and data types match the expected schema, False otherwise.
        """
        #df = pd.read_csv(file_path) # --> should be instance attribute since shared among instance methods.
        self.log_writer.handle_logging(f"Column number and dtype validation stage for {os.path.basename(file_path)}")
        df_columns= set(self.df.columns)
        if is_prediction:
            expected = self.prediction_cols
        else:
            expected = self.training_cols

        missing_cols = set(expected.keys()) - df_columns

        if missing_cols:
            #print("missing cols: ",missing_cols)
            self.log_writer.handle_logging(f"missing cols: {missing_cols}")
            return False 
        
        for col_name, expected_dtype in expected.items():
            if str(self.df[col_name].dtype) not in expected_dtype:
                self.log_writer.handle_logging(f"Dtype un-match detected {col_name}, {self.df[col_name].dtype} != {expected_dtype}")
                return False
        
        return True


    @handle_exceptions
    def check_missing_value_percentage(self,file_path):
        """
        Checks the percentage of missing values in a DataFrame loaded from a file.

        Args:
            file_path (str): The path of the CSV file to check.

        Returns:
            bool: True if the missing value percentage is below the threshold, False otherwise.
        """
        
        #df = pd.read_csv(file_path)
        self.log_writer.handle_logging(f"Missing Value Check stage for {os.path.basename(file_path)}")

        missing_percentage = (self.df.isna().sum().sum())/(self.df.shape[0]*self.df.shape[1]) # for the entire dataset
        self.log_writer.handle_logging(f"Missing value % {missing_percentage}")
        
        #print(any(above_threshold))
        """if any(above_threshold):
            result = {col: above_thresh for col, above_thresh in above_threshold.items()}
            #then log the results here """ # -> this logic can be adjusted for individual column check
            
        return missing_percentage > self.missing_threshold

    @handle_exceptions
    def check_zero_std(self,file_path):
        """
        Checks if any column in the DataFrame has zero standard deviation.

        Args:
            file_path (str): The path of the CSV file to check.

        Returns:
            bool: Always returns False, as this check is for preprocessing, not validation.
        """
        """df= pd.read_csv(file_path)
        df_std = df.iloc[:,1:-1].std()
        zero_std_cols = df_std[df_std==0].index 
        if any(zero_std_cols):
            pass 
            #log here
        #return any(zero_std_cols)""" # i commented out this part, since thats a preprocessing job, not a validation.
        # therefore, return False always, or just remove this part.
        return False
    


    @handle_exceptions
    def data_drift_check(self):
        """
        Performs data drift checks.

        Returns:
            bool: Always returns True for now.
        """
        return True 

    
    @handle_exceptions
    def check_all_conditions(self, file_path,is_prediction=False):
        """
        Checks all validation conditions for a given file.

        Args:
            file_path (str): The path of the file to validate.
            is_prediction (bool, optional): True if validating prediction data, False for training data. Defaults to False.

        Description:
            This method performs a series of data validation checks for the specified file, including filename validation,
            column validation, missing value percentage, and zero standard deviation checks. Depending on the validation results,
            the file may be moved to either the valid or invalid data directory.
        """

        #!!! the following if-else conditionals can be move outside this loop element since we choose prediction or training only once for a pipeline.
        if is_prediction:
            valid_data_path = self.data_validation_config.valid_unseen_data_dir
            invalid_data_path = self.data_validation_config.invalid_unseen_data_dir
            
        else:
            valid_data_path = self.data_validation_config.valid_training_data_dir
            invalid_data_path = self.data_validation_config.invalid_training_data_dir

        self.re_obj = create_regex(is_prediction=is_prediction)
        self.log_writer.handle_logging("Related regex pattern compiled succesfully .")

        self.df = pd.read_csv(file_path)
        self.log_writer.handle_logging("dataframe has been read succesfully!")

        os.makedirs(valid_data_path,exist_ok=True)
        os.makedirs(invalid_data_path,exist_ok=True)
        self.log_writer.handle_logging(f"directories succesfully created 1-{valid_data_path} 2-{invalid_data_path}")

        self.log_writer.handle_logging("Validation Check Stages now starting...")

        if self.validate_file_name(file_path):
            self.log_writer.handle_logging(f"Filename Validated!")
            if self.validate_columns_2(file_path,is_prediction):
                self.log_writer.handle_logging(f"Columns Validated!")
                if not self.check_missing_value_percentage(file_path):
                    self.log_writer.handle_logging(f"Missing Value Validation Passed!")

                    if not self.check_zero_std(file_path):
                        #print("check_zero")                        
                        shutil.move(file_path,os.path.join(valid_data_path,os.path.basename(file_path))) # öncesinde copy yapıp sonra move da edilebilir.
                        self.log_writer.handle_logging(f"{os.path.basename(file_path)} has been validated and moved to related valid dataset directory")
                    
                    else:
                        shutil.move(file_path,os.path.join(invalid_data_path,os.path.basename(file_path)))
                        self.log_writer.handle_logging(f"File {os.path.basename(file_path)} is moved to invalid data directory")

                else:

                    self.log_writer.handle_logging(f"Missing Value Percentage is above the permitted threshold!")
                    shutil.move(file_path,os.path.join(invalid_data_path,os.path.basename(file_path))) 
                    self.log_writer.handle_logging(f"File {os.path.basename(file_path)} is moved to invalid data directory")       


            else:
                self.log_writer.handle_logging(f"Columns are not validated")
                shutil.move(file_path,os.path.join(invalid_data_path,os.path.basename(file_path)))
                self.log_writer.handle_logging(f"File {os.path.basename(file_path)} is moved to invalid data directory") 

        else:
            self.log_writer.handle_logging(f"Filename is not validated")
            shutil.move(file_path,os.path.join(invalid_data_path,os.path.basename(file_path)))
            self.log_writer.handle_logging(f"File {os.path.basename(file_path)} is moved to invalid data directory")



    @handle_exceptions
    def post_validation_insertion(self):
        """
        Inserts valid and invalid files into MongoDB.

        Description:
            This method inserts the files located in the valid and invalid data directories
            into MongoDB collections designated for valid and invalid data.
        """
        valid_files_path_list = [os.path.join(self.data_validation_config.valid_training_data_dir,file) for file in os.listdir(self.data_validation_config.valid_training_data_dir)]
        invalid_files_path_list = [os.path.join(self.data_validation_config.invalid_training_data_dir,file) for file in os.listdir(self.data_validation_config.invalid_training_data_dir)]
        
        self.log_writer.handle_logging("INSERTING VALID FILES TO MONGODB")
        self.mongo_connect.insert_with_fs(valid_files_path_list,MONGO_VALID_COLLECTION_NAME)

        self.log_writer.handle_logging("INSERTING INVALID FILES TO MONGODB")
        self.mongo_connect.insert_with_fs(invalid_files_path_list,MONGO_INVALID_COLLECTION_NAME)
        self.log_writer.handle_logging("POST VALIDATION INSERTION COMPLETED SUCCESFULLY!")



    @handle_exceptions
    def post_validation_deletion(self):
        """
        Deletes local valid and invalid data directories.

        Description:
            This method deletes the local directories containing valid and invalid data
            after they have been successfully processed and ingested into MongoDB.
        """
        self.log_writer.handle_logging("DELETING VALID & INVALID FILES FROM LOCAL!")
        shutil.rmtree(self.data_validation_config.valid_training_data_dir)
        shutil.rmtree(self.data_validation_config.invalid_training_data_dir)
        self.log_writer.handle_logging("DELETION SUCCESFULL!")
        

    @handle_exceptions
    def post_validation_ingestion(self):
        """
        Ingests valid data files from MongoDB into the local directory.

        Description:
            This method retrieves valid data files from MongoDB and stores them in the local
            directory specified for valid data.
        """
        self.log_writer.handle_logging("INGESTING VALID FILES FROM MONGODB")
        target_dir = self.data_validation_config.valid_training_data_dir
        self.mongo_connect.ingest_with_fs(target_dir,self.data_validation_config.mongo_valid_collection_name)
        self.log_writer.handle_logging("POST VALIDATION INGESTION SUCCESFULL!")


    @handle_exceptions
    def run_data_validation(self)-> DataValidationArtifact:
        """
        Run data validation checks for training data.

        Description:
            This method performs data validation checks on the training data files found in
            the configured feature store directory. It includes checks for file names, columns,
            missing values, and zero standard deviations. Validated files are inserted into MongoDB
            and removed from the local directory. An artifact containing directory paths for valid
            and invalid data is returned.

        Returns:
            DataValidationArtifact: An artifact containing directory paths for valid and invalid data.
        """
        
        #feature_store_dir = "C:/Users/ayhan\Desktop/predictive-wafer-maintenance/artifact/30_08_2023_14_02/data_ingestion/feature_store"
        self.log_writer.handle_logging("-------------ENTERED DATA VALIDATION STAGE------------")

        feature_store_dir = self.data_ingestion_artifact.feature_store
        csv_path_list = os.listdir(feature_store_dir)

        for csv_name in csv_path_list:
            print(csv_name)
            file_path = os.path.join(feature_store_dir,csv_name)
            self.check_all_conditions(file_path,is_prediction=False)

        self.log_writer.handle_logging("CONDITIONAL CHECKS COMPLETED!")
        
        

        self.post_validation_insertion()
        self.post_validation_deletion()
        self.post_validation_ingestion()
        
        data_validation_artifact = DataValidationArtifact(
            valid_data_dir= self.data_validation_config.valid_training_data_dir ,
            invalid_data_dir= self.data_validation_config.invalid_training_data_dir, 
            valid_unseen_data_dir= self.data_validation_config.valid_unseen_data_dir,
            invalid_unseen_data_dir= self.data_validation_config.invalid_unseen_data_dir
        )   
        self.log_writer.handle_logging("Data Validation Artifact Created Succesfully..")
        return data_validation_artifact
    

    @handle_exceptions
    def run_prediction_data_validation(self)-> DataValidationArtifact:
        """
        Run data validation checks for prediction data.

        Description:
            This method performs data validation checks on the prediction data files found in
            the configured prediction data directory. It includes checks for file names, columns,
            missing values, and zero standard deviations. Validated files are inserted into MongoDB.
            An artifact containing directory paths for valid and invalid data is returned.

        Returns:
            DataValidationArtifact: An artifact containing directory paths for valid and invalid data.
        """

        self.log_writer.handle_logging("-------------ENTERED DATA VALIDATION STAGE FOR PREDICTION DATASET------------")
        prediction_files_dir = self.data_validation_config.prediction_data_dir
        csv_path_list = os.listdir(prediction_files_dir)
        for csv_name in csv_path_list:
            print(csv_name)
            file_path = os.path.join(prediction_files_dir,csv_name)
            self.check_all_conditions(file_path,is_prediction=True)

        self.log_writer.handle_logging("CONDITIONAL CHECKS COMPLETED!")

        data_validation_artifact = DataValidationArtifact(
            valid_data_dir= self.data_validation_config.valid_training_data_dir ,
            invalid_data_dir= self.data_validation_config.invalid_training_data_dir, 
            valid_unseen_data_dir= self.data_validation_config.valid_unseen_data_dir,
            invalid_unseen_data_dir= self.data_validation_config.invalid_unseen_data_dir
        )  
        self.log_writer.handle_logging("Data Validation Artifact Created Succesfully..")

        return data_validation_artifact


    