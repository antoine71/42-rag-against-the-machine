
*This project has been created as part of the 42 curriculum by arebilla.*


# RAG against the machine: Retrieval Augmented Generation

<p align="center">
  <a href="https://www.python.org">
    <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white" />
  </a>
  <a href="https://github.com/antoine71/42-rag-against-the-machine/actions/workflows/python-app.yml">
    <img src="https://github.com/antoine71/42-rag-against-the-machine/actions/workflows/python-app.yml/badge.svg?branch=main&event=push" />
  </a>
</p>

<p align="center">
  <b>Retrieval Augmented Generation pipeline</b><br/>
    ....... TO BE COMPLETED
</p>


## Description
**RAG** is a retrieval augmented generation application that implement the following functionnalities:
- Build an indexed knowledge base from the project attached files.
- Retrieve and rank the most relevant pieces of information.
- Pass them to the LLM within context limitations.
- Generate structured JSON output as described in the output section.
- Implement intelligent chunking strategies for the different file types.
- Provide a comprehensive CLI interface for all operations.
- Include evaluation metrics and performance analysis.

....ADD A PICTURE

## Instructions

### Prerequisites
- Python 3.10
- `uv` (recommended) or `pip`

### Installation
The project uses `uv` for dependency management. You can install everything using the provided `Makefile`:

```bash
make install
```

### Execution
To run the pipeline with a specific stage
```bash
make run ARGS="index"
```
or
```bash
uv run rag index
```

### Linting and Testing
To ensure code quality and type safety:
```bash
make lint        # Runs flake8 and mypy
make lint-strict # Runs flake8 and mypy --strict
make test        # Runs pytest suite
```

