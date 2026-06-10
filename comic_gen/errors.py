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


class TextGenerationError(RuntimeError):
    """Exception raised for errors in text generation."""

    def __init__(self, message: str):
        super().__init__(message)
        logger.error("TextGenerationError: %s", message)


class UnifiedGenerationError(RuntimeError):
    """Exception raised for errors in unified generation."""

    def __init__(self, message: str):
        super().__init__(message)
        logger.error("UnifiedGenerationError: %s", message)
