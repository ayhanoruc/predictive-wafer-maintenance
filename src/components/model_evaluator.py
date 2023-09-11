from src.utility.generic import save_object , load_object
from src.utility.metrics.classification_metrics import cost_function, get_classification_metrics

from src.exception_handler import CustomException , handle_exceptions
from src.log_handler import AppLogger

from src.entity.artifact_entity import DataValidationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelEvaluationConfig
from src.components.data_transformation import DataTransformationComponent, TrainingPreprocessor

from src.utility.model.model_operations import ReadyModel, ModelResolver


import pandas as pd 
import numpy as np 


class ModelEvaluatorComponent:

    def __init__(self,model_evaluation_config:ModelEvaluationConfig,
                 data_validation_artifact:DataValidationArtifact,
                 model_trainer_artifact:ModelTrainerArtifact):
        
        self.model_evaluation_config= model_evaluation_config
        self.data_validation_artifact = data_validation_artifact
        self.model_trainer_artifact = model_trainer_artifact
        
        self.log_writer = AppLogger("Model Evaluator Component")

    
    @handle_exceptions
    def run_model_evaluator(self)->ModelEvaluationArtifact:
        # i got self.valid_unseen_data_dir & self.invalid_unseen_data_dir which is under data_validation / unseen / valid_data | invali_data
        valid_train_data_dir = self.data_validation_artifact.valid_data_dir

        """
        listdir-> merge csv files -> 
        call Model Resolver, check is_model_exists , if not exists:  return model_eval_artifact
        continue:best_model_exists -> compare them:
        get_best_model_path -> load object or ReadyModel, transform & predict or access stored metrics
        get_current_model_path -> load object or ReadyModel, transform & predict
        calculate score differences for roc_auc & cost 
        check if model_shift_threshold is met: change is_accepted based on this (True by default)
        update model_evaluation artifact and return it 


        """

 
        

