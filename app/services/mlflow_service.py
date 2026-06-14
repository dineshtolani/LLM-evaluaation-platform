import mlflow
import uuid
import logging
from app.config import settings

logger = logging.getLogger(__name__)


class MLflowService:
    def __init__(self):
        self.enabled = True
        self.active_run_id = None
        import socket
        host, port = settings.mlflow_tracking_uri.replace("http://", "").split(":")
        port = int(port.split("/")[0])
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        if result != 0:
            logger.warning(f"MLflow not available at {settings.mlflow_tracking_uri}. Running without tracking.")
            self.enabled = False
            return
        try:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment_name)
        except Exception as e:
            logger.warning(f"MLflow error: {e}. Running without tracking.")
            self.enabled = False

    def start_run(self, prompt_name: str, prompt_id: uuid.UUID, llm_model_id: uuid.UUID):
        if not self.enabled:
            return None
        try:
            run = mlflow.start_run(
                run_name=f"{prompt_name}_{prompt_id}",
                tags={
                    "prompt_id": str(prompt_id),
                    "model_id": str(llm_model_id),
                    "platform": "llm_eval_platform",
                },
            )
            self.active_run_id = run.info.run_id
            return run
        except Exception as e:
            logger.warning(f"MLflow start_run failed: {e}")
            self.enabled = False
            return None

    def log_evaluation_metrics(self, **metrics):
        if not self.enabled:
            return
        try:
            for name, value in metrics.items():
                if value is not None:
                    mlflow.log_metric(name, value)
        except Exception as e:
            logger.warning(f"MLflow log_metrics failed: {e}")

    def log_params(self, params: dict):
        if not self.enabled:
            return
        try:
            mlflow.log_params(params)
        except Exception:
            pass

    def log_artifact(self, local_path: str):
        if not self.enabled:
            return
        try:
            mlflow.log_artifact(local_path)
        except Exception:
            pass

    def end_run(self):
        if not self.enabled:
            return
        try:
            mlflow.end_run()
        except Exception:
            pass
        self.active_run_id = None
