from src.entity.config_entity import TrainingPipelineConfig, DataValidationConfig, DataIngestionConfig, ModelPusherConfig

from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact


from src.components.data_ingestion import DataIngestionComponent
from src.components.data_validation import DataValidationComponent

from src.exception_handler import CustomException, handle_exceptions
from src.log_handler import AppLogger


from src.utility.model.model_operations import ModelResolver, ReadyModel
from src.utility.generic import load_object, load_data, read_json_file



class PredictionPipeline:
    is_pipeline_running = False
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        

        self.log_writer = AppLogger("PredictionPipeline")
    
    def start_data_ingestion(self):
        data_ingestion_config = DataIngestionConfig(self.training_pipeline_config)
        data_ingestion_artifact = None
        return data_ingestion_artifact


    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact):
        data_validation_config = DataValidationConfig(self.training_pipeline_config)
        #prediction_data_dir = self.data_validation_config.prediction_data_dir -> component bunu zaten config file'dan alıyor 

        data_validation_component = DataValidationComponent(data_validation_config,data_ingestion_artifact)
        data_validation_artifact = data_validation_component.run_prediction_data_validation()

        return data_validation_artifact
    
    def start_model_prediction(self, data_validation_artifact:DataValidationArtifact):

        self.model_pusher = ModelPusherConfig(self.training_pipeline_config)
        model_resolver = ModelResolver()
        best_model_path = model_resolver.get_best_model_path()
        best_model:ReadyModel = load_object(best_model_path)
        

        dataframe = load_data(data_validation_artifact.valid_unseen_data_dir,is_prediction=True)
        print(dataframe.shape)

        best_model_y_pred = best_model.predict(dataframe,is_testing=True)
        
        latest_report_yaml_path =  model_resolver.get_latest_best_model_artifact()
        latest_report_yaml = read_json_file(latest_report_yaml_path)
        best_model_metrics = latest_report_yaml["best_model_metrics_artifact"]

        return (best_model_y_pred,best_model_metrics) 
    


        
    def start_prediction_pipeline(self):

        PredictionPipeline.is_pipeline_running = True 
        data_ingestion_artifact = self.start_data_ingestion()
        data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
        
        return self.start_model_prediction(data_validation_artifact)


