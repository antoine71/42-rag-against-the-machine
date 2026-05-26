import logging
import sys
import traceback

import fire

from rag.exceptions import RAGException
from rag.pipeline.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def run_app() -> None:
    fire.Fire(RAGPipeline)


def init_logger() -> None:
    logging.basicConfig(
        filename="rag.log",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.ERROR,
    )
    logging.getLogger("bm25s").setLevel(logging.ERROR)
    # logging.getLogger("rag").setLevel(logging.DEBUG)


def error_handler(e: Exception, error_type: str) -> None:
    logger.error(f"{error_type} ({type(e).__name__}): {str(e)}")
    logger.error(traceback.format_exc())
    print(f"{error_type}: {e}", file=sys.stderr)


def main() -> None:
    try:
        init_logger()
        logger.debug("Starting App...")
        run_app()
    except KeyboardInterrupt:
        sys.exit(1)
    except RAGException as e:
        error_handler(e, "Processing error")
        sys.exit(1)
    except Exception as e:
        error_handler(e, "Unexpected error")
        sys.exit(1)

