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


@dataclass 
class ClassificationMetricsArtifact:
    f1_score : float
    roc_auc_score : float 
    cost_score : int 


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str 
    train_metric_artifact : None
    test_metric_artifact  : ClassificationMetrics

 


