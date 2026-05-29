import logging
import sys
import traceback

import fire

from rag.config.app_config import AppConfig
from rag.exceptions import RAGException
from rag.pipeline.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def run_app() -> None:
    fire.Fire(RAGPipeline)


def init_logger() -> None:
    log_level = AppConfig().log_level
    logging.basicConfig(
        filename="rag.log",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=log_level,
    )
    logging.getLogger("bm25s").setLevel(logging.ERROR)
    logger.info(f"Logger setup with level {log_level}")


def error_handler(e: Exception, error_type: str) -> None:
    logger.error(f"{error_type} ({type(e).__name__}): {str(e)}")
    logger.error(traceback.format_exc())
    print(f"{error_type}: {e}", file=sys.stderr)


def main() -> None:
    try:
        init_logger()
        logger.info("Starting App...")
        run_app()
    except KeyboardInterrupt:
        sys.exit(1)
    except RAGException as e:
        error_handler(e, "Processing error")
        sys.exit(1)
    except Exception as e:
        error_handler(e, "Unexpected error")
        sys.exit(1)
