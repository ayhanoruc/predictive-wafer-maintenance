from src.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from src.entity.config_entity import ModelPusherConfig

from src.exception_handler import CustomException, handle_exceptions
from src.log_handler import AppLogger

import os
import shutil


class ModelPusherComponent:


    def __init__(self,model_pusher_config:ModelPusherConfig,
                 model_evaluation_artifact:ModelEvaluationArtifact):
        
        self.model_pusher_config = model_pusher_config
        self.model_evaluation_artifact = model_evaluation_artifact

        self.log_writer = AppLogger("Model Pusher")

    
    def run_model_pusher(self)->ModelPusherArtifact:

        trained_model_path = self.model_evaluation_artifact.trained_model_file_path

        model_dst_file_path = self.model_pusher_config.model_file_path
        os.makedirs(os.path.dirname(model_dst_file_path),exist_ok=True)
        shutil.copy(src=trained_model_path, dst=model_dst_file_path)

        saved_model_path = self.model_pusher_config.saved_model_path
        os.makedirs(os.path.dirname(saved_model_path),exist_ok=True)
        shutil.copy(src= trained_model_path, dst = saved_model_path)

        model_pusher_artifact = ModelPusherArtifact(
            saved_model_path=saved_model_path,
            model_file_path= model_dst_file_path
        )

        return model_pusher_artifact