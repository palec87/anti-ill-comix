import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


class ModelPipelineError(RuntimeError):
    """Exception raised for errors in the model generation pipeline."""

    def __init__(self, message: str):
        super().__init__(message)
        logger.error("ModelPipelineError: %s", message)
