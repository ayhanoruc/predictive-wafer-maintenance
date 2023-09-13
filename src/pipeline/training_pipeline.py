from src.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig,\
                                        DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig,\
                                        ModelPusherConfig                                          

from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact,DataTransformationArtifact,\
                                       ModelTrainerArtifact,ModelEvaluationArtifact,ModelPusherArtifact

from src.components.data_ingestion import DataIngestionComponent
from src.components.data_validation import DataValidationComponent
from src.components.data_transformation import DataTransformationComponent
from src.components.model_trainer import ModelTrainerComponent
from src.components.model_evaluator import ModelEvaluatorComponent
from src.components.model_pusher import ModelPusherComponent

from src.exception_handler import CustomException, handle_exceptions
from src.log_handler import AppLogger

from src.constants.training_pipeline import SAVED_MODEL_DIR


class TrainingPipeline:
    is_pipeline_running = False

    def __init__(self) -> None:
        self.training_pipeline_config = TrainingPipelineConfig()

        self.log_writer = AppLogger("Traning Pipeline")

    def start_data_ingestion(self) -> DataIngestionArtifact:
        self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
        #log
        data_ingestion_component = DataIngestionComponent(data_ingestion_config=self.data_ingestion_config)
        data_ingestion_artifact = data_ingestion_component.run_data_ingestion()
        # log
        return data_ingestion_artifact


    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:

        data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config) 
        data_validation_component = DataValidationComponent(data_validation_config,data_ingestion_artifact)
        data_validation_artifact = data_validation_component.run_data_validation()

        return data_validation_artifact
    

    def start_data_transformation(self, data_validation_artifact:DataValidationArtifact)->DataTransformationArtifact:
        data_transformation_config  = DataTransformationConfig(self.training_pipeline_config)
        data_transformation_component = DataTransformationComponent(data_transformation_config,data_validation_artifact)
        data_transformation_artifact = data_transformation_component.run_data_transformation()

        return data_transformation_artifact
    

    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact:
        model_trainer_config = ModelTrainerConfig(self.training_pipeline_config)
        model_trainer_component = ModelTrainerComponent(model_trainer_config,data_transformation_artifact)
        model_trainer_artifact = model_trainer_component.run_model_trainer()

        return model_trainer_artifact
    

    def start_model_evaluation(self, data_validation_artifact:DataValidationArtifact,
                                     model_trainer_artifact: ModelTrainerArtifact)->ModelEvaluationArtifact:
        model_evaluation_config = ModelEvaluationConfig(self.training_pipeline_config)
        model_evaluator_component = ModelEvaluatorComponent(model_evaluation_config,data_validation_artifact,model_trainer_artifact)
        model_evaluator_artifact = model_evaluator_component.run_model_evaluator(threshold=0.26)

        return model_evaluator_artifact
    

    def start_model_pusher(self,model_evaluator_artifact:ModelEvaluationArtifact)->ModelPusherArtifact:
        model_pusher_config = ModelPusherConfig(self.training_pipeline_config)
        model_pusher_component = ModelPusherComponent(model_pusher_config,model_evaluator_artifact)
        model_pusher_artifact = model_pusher_component.run_model_pusher()
        
        return model_pusher_artifact
    

    def run_training_pipeline(self):

        TrainingPipeline.is_pipeline_running = True 
        data_ingestion_artifact:DataIngestionArtifact = self.start_data_ingestion()
        data_validation_artifact:DataValidationArtifact = self.start_data_validation(data_ingestion_artifact)
        
        # bu arada tekrardan file ingest etmem gerekebilir. Bu pipeline'ı gözden geçir.
        data_transformation_artifact:DataTransformationArtifact = self.start_data_transformation(data_validation_artifact)
        model_trainer_artifact:ModelTrainerArtifact = self.start_model_trainer(data_transformation_artifact)
        model_evaluator_artifact:ModelEvaluationArtifact = self.start_model_evaluation(data_validation_artifact,
                                                                                        model_trainer_artifact)
        if not model_evaluator_artifact.is_model_accepted:
            # CUSTOMIZE ET
            raise Exception("Current Trained Model is not improved or couldn satisfied the min accuracy")

        model_pusher_artifact:ModelPusherArtifact = self.start_model_pusher(model_evaluator_artifact)
        
        TrainingPipeline.is_pipeline_running = False
        self.log_writer.handle_logging("TRAINING PIPELINE COMPLETED!")
