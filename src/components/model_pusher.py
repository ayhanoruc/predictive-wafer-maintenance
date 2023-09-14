from src.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from src.entity.config_entity import ModelPusherConfig

from src.exception_handler import handle_exceptions
from src.log_handler import AppLogger

import os
import shutil


class ModelPusherComponent:
    """
    Execute the model pushing process.

    This method performs the following steps:
    1. Copy the last trained model to the specified model destination path.
    2. Copy the last trained model to a saved models directory.
    3. Create a Model Pusher Artifact containing paths to the saved model and the model file.

    Returns:
        ModelPusherArtifact: An artifact containing paths to the saved model and the model file.
    """


    def __init__(self,model_pusher_config:ModelPusherConfig,
                 model_evaluation_artifact:ModelEvaluationArtifact):
        
        self.model_pusher_config = model_pusher_config
        self.model_evaluation_artifact = model_evaluation_artifact

        self.log_writer = AppLogger("Model Pusher")

    @handle_exceptions
    def run_model_pusher(self)->ModelPusherArtifact:
        self.log_writer.handle_logging("-------------ENTERED MODEL PUSHER STAGE------------")

        trained_model_path = self.model_evaluation_artifact.trained_model_file_path

        model_dst_file_path = self.model_pusher_config.model_file_path
        os.makedirs(os.path.dirname(model_dst_file_path),exist_ok=True)
        shutil.copy(src=trained_model_path, dst=model_dst_file_path)
        self.log_writer.handle_logging("last trained model is copied & moved to model pusher artifact as .pkl object")

        saved_model_path = self.model_pusher_config.saved_model_path
        os.makedirs(os.path.dirname(saved_model_path),exist_ok=True)
        shutil.copy(src= trained_model_path, dst = saved_model_path)
        self.log_writer.handle_logging("last trained model is copied & moved to saved models as .pkl object")

        model_pusher_artifact = ModelPusherArtifact(
            saved_model_path=saved_model_path,
            model_file_path= model_dst_file_path
        )
        self.log_writer.handle_logging("Generated Model Pusher Artifact succesfully!")

        return model_pusher_artifact