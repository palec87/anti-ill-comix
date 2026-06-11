import logging
import os
from inspect import currentframe

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _resolve_error_origin(
    file_name: str | None,
    method_name: str | None,
) -> tuple[str, str]:
    """Resolve the source location where the exception was raised."""
    if file_name and method_name:
        return file_name, method_name

    frame = currentframe()
    helper_caller = frame.f_back if frame else None
    caller_frame = helper_caller.f_back if helper_caller else None
    resolved_file = file_name or "unknown_file"
    resolved_method = method_name or "unknown_method"

    if caller_frame is not None:
        code = caller_frame.f_code
        resolved_file = file_name or os.path.basename(code.co_filename)
        resolved_method = method_name or code.co_name

    return resolved_file, resolved_method


def _build_message(message: str, file_name: str, method_name: str) -> str:
    """Create a standardized exception message with origin details."""
    return f"{message} [source: {file_name}:{method_name}]"


class ModelPipelineError(RuntimeError):
    """Exception raised for errors in the model generation pipeline."""

    def __init__(
        self,
        message: str,
        file_name: str | None = None,
        method_name: str | None = None,
    ):
        self.file_name, self.method_name = _resolve_error_origin(
            file_name,
            method_name,
        )
        full_message = _build_message(
            message,
            self.file_name,
            self.method_name,
        )
        super().__init__(full_message)
        logger.error("ModelPipelineError: %s", full_message)


class TextGenerationError(RuntimeError):
    """Exception raised for errors in text generation."""

    def __init__(
        self,
        message: str,
        file_name: str | None = None,
        method_name: str | None = None,
    ):
        self.file_name, self.method_name = _resolve_error_origin(
            file_name,
            method_name,
        )
        full_message = _build_message(
            message,
            self.file_name,
            self.method_name,
        )
        super().__init__(full_message)
        logger.error("TextGenerationError: %s", full_message)


class UnifiedGenerationError(RuntimeError):
    """Exception raised for errors in unified generation."""

    def __init__(
        self,
        message: str,
        file_name: str | None = None,
        method_name: str | None = None,
    ):
        self.file_name, self.method_name = _resolve_error_origin(
            file_name,
            method_name,
        )
        full_message = _build_message(
            message,
            self.file_name,
            self.method_name,
        )
        super().__init__(full_message)
        logger.error("UnifiedGenerationError: %s", full_message)


class ImageGenerationError(Exception):
    """Raised when live image generation fails."""

    def __init__(
        self,
        message: str,
        file_name: str | None = None,
        method_name: str | None = None,
    ):
        self.file_name, self.method_name = _resolve_error_origin(
            file_name,
            method_name,
        )
        full_message = _build_message(
            message,
            self.file_name,
            self.method_name,
        )
        super().__init__(full_message)
        logger.error("ImageGenerationError: %s", full_message)
