import sys

import fire

from rag.pipeline.rag_pipeline import RAGPipeline


def run_app() -> None:
    fire.Fire(RAGPipeline)


def main() -> None:
    try:
        run_app()
    except Exception:
        print("Error!", file=sys.stderr)
        sys.exit(1)
