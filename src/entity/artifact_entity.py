from dataclasses import dataclass 



@dataclass 
class DataIngestionArtifact:
    feature_store:str
    train_file_path:str 
    test_file_path: str 



@dataclass
class DataValidationArtifact:

    valid_data_dir:str 
    invalid_data_dir:str
    

@dataclass 
class DataTransformationArtifact:

    transformed_train_file_path : str 
    transformed_test_file_path : str 
    preprocessor_object_file_path : str


