from src.utility.generic import save_object , load_object, load_data, write_json_file
from src.utility.metrics.classification_metrics import get_classification_metrics

from src.exception_handler import CustomException , handle_exceptions
from src.log_handler import AppLogger

from src.entity.artifact_entity import DataValidationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact, DataTransformationArtifact, ClassificationMetricsArtifact
from src.entity.config_entity import ModelEvaluationConfig, ModelTrainerConfig
from src.components.data_transformation import DataTransformationComponent, TrainingPreprocessor
from src.components.model_trainer import ModelTrainerComponent

from src.utility.model.model_operations import ReadyModel, ModelResolver


import pandas as pd 
import numpy as np 

import os,sys
import shutil



class ModelEvaluatorComponent:

    def __init__(self,model_evaluation_config:ModelEvaluationConfig,
                 data_validation_artifact:DataValidationArtifact,
                 model_trainer_artifact:ModelTrainerArtifact):
        
        self.model_evaluation_config= model_evaluation_config
        self.model_trainer_artifact = model_trainer_artifact
        self.data_validation_artifact = data_validation_artifact
        

        
        self.log_writer = AppLogger("Model Evaluator Component")

    
    @handle_exceptions
    def run_model_evaluator(self,threshold)->ModelEvaluationArtifact:
        # threshold değeri model_trainer objectinde model_trainer().best_threshold attribute'u şeklinde çağırılır ve buraya pass edilir.


        # i got self.valid_unseen_data_dir & self.invalid_unseen_data_dir which is under data_validation / unseen / valid_data | invali_data
        # valid dataset dir ilk önce dolacak, validationdan sonra silinecek, sonra yeni training process
        # başlarsa tekrardan dolu olacak. ModelEvaluation'dan önce haliyle dolu olması gerekir. 
        # Main Training Pipeline runtime'da valid training data ingest edilip data_validation artifact elde edilir
        # bu run_model_eval içinde yeni model training için datatransformation component run edilir, data transformation artifact elde edilir
        # devamında ModelTrainer çağırılır ve metrikler elde edilip karşılaştırılır.

        # here iki model için X_train, X_test, y_train, y_test şart
        # 
        # best modeli ReadyModel cinsinde model_eval/best_models altında depoladıgımız için best_preprocessor_obj, best_model = load_object(best_model_path) diye assign ederiz
        # aynı şekilde current modelimiz de bi önceki step olan modeltrainer'dan preprocessor_obj ve model_obj olarak model_trainer_artifact'te depolandı
        # bunu modeltrainer artifactten load ederiz, böylece her iki model de predicte hazır durumda olur

        valid_train_data_dir = self.data_validation_artifact.valid_data_dir 
        test_df = load_data(valid_train_data_dir)
        X= test_df.iloc[:,:-1]
        y= test_df.iloc[:,-1]

        is_model_accepted = True 
        trained_model_file_path = self.model_trainer_artifact.trained_model_file_path


        model_resolver = ModelResolver()
        if not model_resolver.is_a_model_exists(): # if no best model exists, save as best model w/out comparison process

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted = is_model_accepted,
                improved_cost_score = 0.0,
                improved_roc_auc_score= 0.0,
                trained_model_file_path= trained_model_file_path,
                trained_model_metrics_artifact=self.model_trainer_artifact.test_metric_artifact,
                best_model_metrics_artifact= self.model_trainer_artifact.test_metric_artifact 
            ) 
            return model_evaluation_artifact
        

        best_model_path:str = model_resolver.get_best_model_path()
        best_ready_model:ReadyModel = load_object(best_model_path)
        current_ready_model:ReadyModel = load_object(trained_model_file_path) 
        #tek model oldugunda kendisiyle kıyaslayacak, logic daha farklı kurulabilir ama bu da pratik bir implementation

        best_model_y_pred = best_ready_model.predict(X,threshold)
        current_model_y_pred = current_ready_model.predict(X,threshold)

        best_model_f1_score,best_model_roc_auc_score,best_model_test_cost = get_classification_metrics(y,best_model_y_pred)
        current_f1_score,current_roc_auc_score,current_test_cost = get_classification_metrics(y,current_model_y_pred)

        #             >>>>  get_classification_metrics'i ClassificationMetricsArtifact döndürecek hale getirsem daha iyi olur <<<<<<<<<<<<<<<<

        current_model_metrics_artifact = ClassificationMetricsArtifact(
            f1_score= current_f1_score,
            roc_auc_score= current_roc_auc_score,
            cost_score= current_test_cost
        )
        best_model_metrics_artifact = ClassificationMetricsArtifact(
        f1_score= best_model_f1_score,
        roc_auc_score= best_model_roc_auc_score,
        cost_score= best_model_test_cost)

        # COMPARISON
        # calculate and normalize the change in related metric scores
        roc_auc_score_diff = (current_roc_auc_score - best_model_roc_auc_score)/best_model_roc_auc_score  
        f1_score_diff = (current_f1_score - best_model_f1_score)/best_model_f1_score
        cost_score_diff = (current_test_cost - best_model_test_cost)/ best_model_test_cost # we need to minimize this


        # in this project, be more concerned about total cost score
        if cost_score_diff < -0.01:
            is_model_accepted = True
        else:
            is_model_accepted = False 

        
        model_evaluation_artifact = ModelEvaluationArtifact(
            is_model_accepted=is_model_accepted,
            improved_f1_score = f1_score_diff,
            improved_roc_auc_score= roc_auc_score_diff,
            best_model_path = best_model_path ,# this can change over time, so it is a good practice to update here
            trained_model_file_path = trained_model_file_path,
            trained_model_metrics_artifact = current_model_metrics_artifact,
            best_model_metrics_artifact = best_model_metrics_artifact
        )
        self.log_writer.handle_logging(f"Model evaluation artifact : {model_evaluation_artifact}")

        model_eval_report = model_evaluation_artifact.__dict__
        write_json_file(self.model_evaluation_config.report_file_path,model_eval_report)

        return model_evaluation_artifact

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

 
        

