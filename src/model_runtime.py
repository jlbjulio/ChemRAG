import logging
import os
import sys
import warnings
from io import TextIOWrapper


os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


def configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="replace")


def configure_quiet_model_loading() -> None:
    logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message=".*unauthenticated requests.*",
    )

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
    except ImportError:
        pass
