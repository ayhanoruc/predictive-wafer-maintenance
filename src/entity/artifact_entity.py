from dataclasses import dataclass 



@dataclass 
class DataIngestionArtifact:
    feature_store:str
    train_file_path:str 
    test_file_path: str 



@dataclass
class DataValidationArtifact:

    valid_data_dir : str 
    invalid_data_dir : str
    valid_unseen_data_dir : str 
    invalid_unseen_data_dir : str

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

    def to_dict(self):
        return {
            "f1_score": self.f1_score,
            "roc_auc_score": self.roc_auc_score,
            "cost_score": int(self.cost_score)
        }


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str 
    train_metric_artifact : None
    test_metric_artifact  : ClassificationMetricsArtifact

    def to_dict(self):
        return {
            "trained_model_file_path": self.trained_model_file_path,
            "train_metric_artifact": self.train_metric_artifact,
            "test_metric_artifact": self.test_metric_artifact.to_dict()
        }



 
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

    def to_dict(self):
        return {
            "is_model_accepted": self.is_model_accepted,
            #"improved_cost_score": self.improved_cost_score,
            "improved_roc_auc_score": self.improved_roc_auc_score,
            "improved_f1_score" : self.improved_f1_score,
            "best_model_path" : self.best_model_path,
            "trained_model_file_path" : self.trained_model_file_path,
            "trained_model_metrics_artifact" : self.trained_model_metrics_artifact,
            "best_model_metrics_artifact" : self.best_model_metrics_artifact

        }



@dataclass
class ModelPusherArtifact:
    saved_model_path : str
    model_file_path : str