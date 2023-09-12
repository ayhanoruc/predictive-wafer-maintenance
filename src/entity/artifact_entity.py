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
    test_metric_artifact  : ClassificationMetricsArtifact



 
@dataclass 
class ModelEvaluationArtifact:
    
    is_model_accepted:bool
    improved_cost_score:float
    improved_roc_auc_score:float
    improved_f1_score:float
    best_model_path:str 
    trained_model_file_path: str 
    trained_model_metrics_artifact: ClassificationMetricsArtifact
    best_model_metrics_artifact: ClassificationMetricsArtifact



@dataclass
class ModelPusherArtifact:
    saved_model_path : str
    model_file_path : str