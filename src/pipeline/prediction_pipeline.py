from src.entity.config_entity import (
    TrainingPipelineConfig,
    DataValidationConfig,
    DataIngestionConfig,
    ModelPusherConfig,
)

from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact

from src.components.data_validation import DataValidationComponent

from src.exception_handler import CustomException, handle_exceptions
from src.log_handler import AppLogger


from src.utility.model.model_operations import ModelResolver, ReadyModel
from src.utility.generic import load_object, load_data, read_json_file


class PredictionPipeline:
    """
    A class representing the prediction pipeline for making predictions using a trained model.

    This pipeline involves data ingestion, data validation, and model prediction.

    Attributes:
        training_pipeline_config (TrainingPipelineConfig): The configuration for the training pipeline.
        log_writer (AppLogger): An instance of the AppLogger class for logging.
    """

    is_pipeline_running = False

    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

        self.log_writer = AppLogger("PredictionPipeline")

    @handle_exceptions
    def start_data_ingestion(self):
        """
        Start the data ingestion process.

        Returns:
            DataIngestionArtifact
        """
        data_ingestion_config = DataIngestionConfig(self.training_pipeline_config)
        data_ingestion_artifact = None
        return data_ingestion_artifact

    @handle_exceptions
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
        data_validation_config = DataValidationConfig(self.training_pipeline_config)
        # prediction_data_dir = self.data_validation_config.prediction_data_dir -> component bunu zaten config file'dan alıyor

        data_validation_component = DataValidationComponent(
            data_validation_config, data_ingestion_artifact
        )
        data_validation_artifact = (
            data_validation_component.run_prediction_data_validation()
        )

        return data_validation_artifact

    @handle_exceptions
    def start_model_prediction(self, data_validation_artifact: DataValidationArtifact):
        """
        Perform model prediction on unseen data using the best-trained model.

        This method loads the best-trained model, predicts the outcomes on the provided
        unseen data, and retrieves the performance metrics of the best model from the
        latest model evaluation report.

        Args:
            data_validation_artifact (DataValidationArtifact): The artifact containing
                information about the validated and prepared unseen data.

        Returns:
            tuple: A tuple containing the predicted labels (best_model_y_pred) and the
            performance metrics (best_model_metrics) of the best-trained model.
        """

        self.model_pusher = ModelPusherConfig(self.training_pipeline_config)
        model_resolver = ModelResolver()
        best_model_path = model_resolver.get_best_model_path()
        best_model: ReadyModel = load_object(best_model_path)
        self.log_writer.handle_logging(f"Loaded best model succesfully!")

        dataframe = load_data(
            data_validation_artifact.valid_unseen_data_dir, is_prediction=True
        )
        self.log_writer.handle_logging(
            f"Loaded dataframe for prediction with shape: {dataframe.shape}"
        )
        # print(dataframe.shape)

        best_model_y_pred = best_model.predict(dataframe, is_testing=True)
        self.log_writer.handle_logging(f"Predicted using best model succesfully!")

        latest_report_yaml_path = model_resolver.get_latest_best_model_artifact()
        latest_report_yaml = read_json_file(latest_report_yaml_path)
        best_model_metrics = latest_report_yaml["best_model_metrics_artifact"]
        self.log_writer.handle_logging(
            f"Obtained and returned best_model_y_pred & best_model_metrics"
        )

        return (best_model_y_pred, best_model_metrics)

    def start_prediction_pipeline(self):
        """
        Start the prediction pipeline, which includes data ingestion, data validation, and model prediction.

        This method orchestrates the entire prediction pipeline, including data ingestion,
        data validation, and model prediction stages. It sets a flag to indicate that the
        pipeline is running, performs each stage, and logs the successful completion of
        the prediction process.

        Returns:
            tuple: A tuple containing the predicted labels (best_model_y_pred) and the
            performance metrics (best_model_metrics) of the best-trained model.
        """

        self.log_writer.handle_logging(
            "-------------ENTERED PREDICTION PIPELINE STAGE------------"
        )

        PredictionPipeline.is_pipeline_running = True
        data_ingestion_artifact = self.start_data_ingestion()
        data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
        self.log_writer.handle_logging("Prediction Completed with Success!!!")
        return self.start_model_prediction(data_validation_artifact)
