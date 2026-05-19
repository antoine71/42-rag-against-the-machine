import logging
import sys
import traceback

import fire

from rag.pipeline.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def run_app() -> None:
    fire.Fire(RAGPipeline)


def init_logger() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def main() -> None:
    try:
        init_logger()
        run_app()
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.print_exc())
        sys.exit(1)
