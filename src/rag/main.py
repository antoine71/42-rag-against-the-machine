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
        # filename="rag.log",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.ERROR,
    )
    logging.getLogger("bm25s").setLevel(logging.INFO)
    logging.getLogger("rag").setLevel(logging.INFO)


def main() -> None:
    try:
        init_logger()
        run_app()
    except RAGException as e:
        logger.error(f"Processing Error: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as e:
        logger.error(f"unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
