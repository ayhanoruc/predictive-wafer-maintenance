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
    