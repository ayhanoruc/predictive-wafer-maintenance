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

from src.exception_handler import handle_exceptions
from src.log_handler import AppLogger

from src.constants.training_pipeline import DATA_INGESTION_UPLOADED_FEATURE_STORE


class TrainingPipeline:
    is_pipeline_running = False

    def __init__(self) -> None:
        """
        Initialize the TrainingPipeline class.

        This class orchestrates the end-to-end training pipeline, including data
        ingestion, validation, transformation, model training, evaluation, and pushing.

        Attributes:
            training_pipeline_config (TrainingPipelineConfig): Configuration object
                for the training pipeline.
            log_writer (AppLogger): Logger instance for logging pipeline-related
                information.
        """
        self.training_pipeline_config = TrainingPipelineConfig()

        self.log_writer = AppLogger("Traning Pipeline")

    @handle_exceptions
    def start_component(self, config_class, component_class, run_method: str, *args,**kwargs) -> object:
        """
        Start a specific pipeline component.

        Args:
            config_class: The configuration class for the component.
            component_class: The component class to be started.
            run_method (str): The name of the method to run within the component.
            *args: Additional positional arguments to pass to the component.
            **kwargs: Additional keyword arguments to pass to the component.

        Returns:
            object: The artifact returned by the component.
        """
        config = config_class(training_pipeline_config=self.training_pipeline_config)   
        component = component_class(config, *args, )
        self.log_writer.handle_logging(f"{component.__class__} is succesfully initialized")
        artifact = getattr(component, run_method)(**kwargs)
        self.log_writer.handle_logging(f"{artifact.__class__} is succesfully created")
        
        return artifact
    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        return self.start_component(DataIngestionConfig, DataIngestionComponent, "run_data_ingestion")

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        return self.start_component(DataValidationConfig, DataValidationComponent, "run_data_validation", data_ingestion_artifact)

    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        return self.start_component(DataTransformationConfig, DataTransformationComponent, "run_data_transformation", data_validation_artifact)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        return self.start_component(ModelTrainerConfig, ModelTrainerComponent, "run_model_trainer", data_transformation_artifact)

    def start_model_evaluation(self, data_validation_artifact: DataValidationArtifact, model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        return self.start_component(ModelEvaluationConfig, ModelEvaluatorComponent, "run_model_evaluator", data_validation_artifact, model_trainer_artifact,threshold=0.13)
    

    def start_model_pusher(self, model_evaluator_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        return self.start_component(ModelPusherConfig, ModelPusherComponent, "run_model_pusher", model_evaluator_artifact)
    
    @handle_exceptions
    def run_training_pipeline(self, is_manual_ingestion = False):
        """
        Run the entire training pipeline, including data ingestion, validation,
        transformation, training, evaluation, and model pushing.

        Args:
            is_manual_ingestion (bool, optional): Whether to use manually uploaded
            feature store directory or default ingested directory. Defaults to False.

        Returns:
            dict: Model evaluation metrics as a dictionary.
        """
        self.log_writer.handle_logging("-------------ENTERED TRAINING PIPELINE STAGE------------")

        TrainingPipeline.is_pipeline_running = True 

        if  not is_manual_ingestion:
            """
            this uses default ingested feature store directory which is from mongo db
            """
            
            data_ingestion_artifact:DataIngestionArtifact = self.start_data_ingestion()

        else:
            """
            this uses manually uploaded feature store directory which is ../uploaded_feature_store
            """
            data_ingestion_artifact= DataIngestionArtifact(
                feature_store= DATA_INGESTION_UPLOADED_FEATURE_STORE,
                train_file_path = None,
                test_file_path = None
            )
            self.log_writer.handle_logging("Preferred data ingestion by manual upload")

        # rest is common among the two ingestion options.

        data_validation_artifact:DataValidationArtifact = self.start_data_validation(data_ingestion_artifact)
        data_transformation_artifact:DataTransformationArtifact = self.start_data_transformation(data_validation_artifact)
        model_trainer_artifact:ModelTrainerArtifact = self.start_model_trainer(data_transformation_artifact)
        model_evaluator_artifact:ModelEvaluationArtifact = self.start_model_evaluation(data_validation_artifact,
                                                                                        model_trainer_artifact)
        if not model_evaluator_artifact.is_model_accepted:
            self.log_writer.handle_logging("Current trained model did not meet minimum accuracy requirements.")
            print("Current Trained Model is not improved or couldn satisfied the min accuracy")
            return False 

        model_pusher_artifact:ModelPusherArtifact = self.start_model_pusher(model_evaluator_artifact)
        
        TrainingPipeline.is_pipeline_running = False
        self.log_writer.handle_logging("TRAINING PIPELINE COMPLETED WITH SUCCESS!!!")

        return model_evaluator_artifact.to_dict()








