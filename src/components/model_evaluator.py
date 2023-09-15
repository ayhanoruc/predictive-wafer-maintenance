from src.utility.generic import load_object, load_data, write_json_file
from src.utility.metrics.classification_metrics import get_classification_metrics

from src.exception_handler import handle_exceptions
from src.log_handler import AppLogger

from src.entity.artifact_entity import (
    DataValidationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ClassificationMetricsArtifact,
)
from src.entity.config_entity import ModelEvaluationConfig

from src.utility.model.model_operations import ReadyModel, ModelResolver

import os


class ModelEvaluatorComponent:
    """
    This class is responsible for evaluating machine learning models using various metrics and comparing
    them to determine whether a newly trained model should be accepted as the best model or not.

    Args:
        model_evaluation_config (ModelEvaluationConfig): Configuration for model evaluation.
        data_validation_artifact (DataValidationArtifact): Artifact from data validation.
        model_trainer_artifact (ModelTrainerArtifact): Artifact from model training.

    Attributes:
        model_evaluation_config (ModelEvaluationConfig): Configuration for model evaluation.
        model_trainer_artifact (ModelTrainerArtifact): Artifact from model training.
        data_validation_artifact (DataValidationArtifact): Artifact from data validation.
        log_writer (AppLogger): Logger for recording component-specific logs.

    Methods:
        run_model_evaluator(threshold: float) -> ModelEvaluationArtifact:
            Run the model evaluation process and return the evaluation artifact.
    """

    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        data_validation_artifact: DataValidationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifact = model_trainer_artifact
        self.data_validation_artifact = data_validation_artifact

        self.log_writer = AppLogger("Model Evaluator Component")

    @handle_exceptions
    def run_model_evaluator(self, threshold) -> ModelEvaluationArtifact:
        self.log_writer.handle_logging(
            "-------------ENTERED MODEL EVALUATOR STAGE------------"
        )
        """
        Run the model evaluator component.

        This method evaluates the current trained machine learning model against the best model (if available) using the specified threshold for binary classification.
        It calculates various classification metrics, compares the current model's performance with the best model's performance, and returns a ModelEvaluationArtifact containing evaluation results.

        Args:
            threshold (float): The threshold for binary classification.

        Returns:
            ModelEvaluationArtifact: An artifact containing detailed evaluation results and comparison information.
        """
        # threshold değeri model_trainer objectinde model_trainer().best_threshold attribute'u şeklinde çağırılır ve buraya pass edilir.
        # here iki model için X_train, X_test, y_train, y_test şart
        # best modeli ReadyModel cinsinde model_eval/best_models altında depoladıgımız için best_preprocessor_obj, best_model = load_object(best_model_path) diye assign ederiz
        # aynı şekilde current modelimiz de bi önceki step olan modeltrainer'dan preprocessor_obj ve model_obj olarak model_trainer_artifact'te depolandı
        # valid_train_data_dir = r"C:\Users\ayhan\Desktop\predictive-wafer-maintenance\valid_feature_store\valid_training_data" # test purpose

        # Load validation data
        valid_train_data_dir = (
            self.data_validation_artifact.valid_data_dir
        )  # <- UPDATED
        test_df = load_data(valid_train_data_dir)
        X = test_df.iloc[:, :-1]
        y = test_df.iloc[:, -1]
        self.log_writer.handle_logging(
            f"Succesfully loaded validation data from directory: {valid_train_data_dir}"
        )

        # Initialize variables for comparison
        is_model_accepted = True
        trained_model_file_path = self.model_trainer_artifact.trained_model_file_path

        model_resolver = ModelResolver()
        # Check if there is a best model available
        if not model_resolver.is_a_model_exists():
            # if no best model exists, save as best model w/out comparison process
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_f1_score=0.0,
                improved_cost_score=0.0,
                improved_roc_auc_score=0.0,
                trained_model_file_path=trained_model_file_path,
                trained_model_metrics_artifact=self.model_trainer_artifact.test_metric_artifact,
                best_model_path=trained_model_file_path,
                best_model_metrics_artifact=self.model_trainer_artifact.test_metric_artifact,
            )
            self.log_writer.handle_logging(
                "No existing best model found. The current model is set as the best model."
            )
            return model_evaluation_artifact

        # Load the best model and the current trained model
        best_model_path: str = model_resolver.get_best_model_path()
        best_ready_model: ReadyModel = load_object(best_model_path)
        current_ready_model: ReadyModel = load_object(trained_model_file_path)
        self.log_writer.handle_logging(
            f"Succesfully loaded the best and current trained models"
        )

        # Predict using both models on the unseen dataset
        best_model_y_pred = best_ready_model.predict(X, threshold)
        self.log_writer.handle_logging("Best Model Prediction completed")
        current_model_y_pred = current_ready_model.predict(X, threshold)
        self.log_writer.handle_logging("Current Model prediction completed")

        # Calculate classification metrics for both models
        (
            best_model_f1_score,
            best_model_roc_auc_score,
            best_model_test_cost,
        ) = get_classification_metrics(y, best_model_y_pred)
        (
            current_f1_score,
            current_roc_auc_score,
            current_test_cost,
        ) = get_classification_metrics(y, current_model_y_pred)
        self.log_writer.handle_logging(
            "Calculated classification metrics for both models for comparison."
        )

        #!!! its better practice to turn  get_classification_metrics to a method returning ClassificationMetricsArtifact

        # Create ClassificationMetricsArtifact for both models
        current_model_metrics_artifact = ClassificationMetricsArtifact(
            f1_score=current_f1_score,
            roc_auc_score=current_roc_auc_score,
            cost_score=current_test_cost,
        )
        best_model_metrics_artifact = ClassificationMetricsArtifact(
            f1_score=best_model_f1_score,
            roc_auc_score=best_model_roc_auc_score,
            cost_score=best_model_test_cost,
        )

        # COMPARISON
        # Calculate and normalize the change in related metric scores between the current model and the best model
        roc_auc_score_diff = (
            current_roc_auc_score - best_model_roc_auc_score
        ) / best_model_roc_auc_score
        f1_score_diff = (current_f1_score - best_model_f1_score) / best_model_f1_score
        cost_score_diff = (
            current_test_cost - best_model_test_cost
        ) / best_model_test_cost

        # Determine if the current model is accepted based on predefined criteria (e.g., F1 score improvement)
        if f1_score_diff >= 0.01:  # needs to be stored in constants
            is_model_accepted = True
            self.log_writer.handle_logging(
                "The current model is accepted as the new best model due to a significant improvement in F1 score."
            )
        else:
            is_model_accepted = False
            self.log_writer.handle_logging(
                "The current model is not accepted as the new best model due to insufficient improvement in F1 score."
            )

        # Create the ModelEvaluationArtifact to store evaluation results and comparison information
        model_evaluation_artifact = ModelEvaluationArtifact(
            is_model_accepted=is_model_accepted,
            improved_f1_score=f1_score_diff,
            improved_roc_auc_score=roc_auc_score_diff,
            improved_cost_score=cost_score_diff,
            best_model_path=best_model_path,  # this can change over time, so it is a good practice to update here
            trained_model_file_path=trained_model_file_path,
            trained_model_metrics_artifact=current_model_metrics_artifact.to_dict(),
            best_model_metrics_artifact=best_model_metrics_artifact.to_dict(),
        )
        self.log_writer.handle_logging(
            f"Model evaluation artifact generated succesfully : {model_evaluation_artifact}"
        )

        os.makedirs(
            os.path.dirname(self.model_evaluation_config.report_file_path),
            exist_ok=True,
        )
        model_eval_report = model_evaluation_artifact.__dict__
        write_json_file(
            self.model_evaluation_config.report_file_path, model_eval_report
        )
        self.log_writer.handle_logging(
            f"Saved the model evaluation report as a JSON file: {self.model_evaluation_config.report_file_path}"
        )

        self.log_writer.handle_logging("Model evaluation process completed.")
        return model_evaluation_artifact
