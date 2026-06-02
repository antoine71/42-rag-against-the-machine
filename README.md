
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
  <b>A full Retrieval-Augmented Generation pipeline over a code repository</b><br/>
  Index → Retrieve → Answer — with BM25, vector embeddings, or hybrid search.
</p>


## Description

**RAG against the machine** is a command-line RAG (Retrieval-Augmented Generation) application built around the [vLLM](https://github.com/vllm-project/vllm) repository. Given a set of natural-language questions, the system:

1. **Indexes** a code/documentation repository by splitting files into semantically meaningful chunks.
2. **Retrieves** the most relevant chunks for each query using BM25, dense vector embeddings, or a hybrid combination of both.
3. **Answers** each question by passing the retrieved context to a large language model (LLM) and returning a structured JSON response.

The output conforms to the Pydantic models defined in the subject (`StudentSearchResults` / `StudentSearchResultsAndAnswer`), and performance is measured with a **Recall@k** metric.


## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        RAGPipeline (CLI)                         │
│  index | search | search_dataset | answer | answer_dataset       │
│                          | evaluate                              │
└───────────────┬──────────────────────────┬───────────────────────┘
                │                          │
    ┌───────────▼──────────┐   ┌───────────▼──────────────────────┐
    │   Indexing pipeline  │   │       Retrieving pipeline        │
    │                      │   │                                  │
    │ FilesRepositoryScanner│  │  RetrievingProcessorFactory      │
    │ LangChainChunking    │   │  ┌──────────┬────────────────┐   │
    │  Processor           │   │  │ BM25     │ Vector         │   │
    │ IndexingProcessor    │   │  │ Retriever│ Retriever      │   │
    │  Factory             │   │  └────┬─────┴────────┬───────┘   │
    │  ┌──────┬──────────┐ │   │       └──── RRF ─────┘           │
    │  │ BM25 │ Vector   │ │   │       (hybrid mode)              │
    │  └──────┴──────────┘ │   └───────────┬──────────────────────┘
    └──────────────────────┘               │
                                ┌──────────▼──────────┐
                                │    LLM Manager      │
                                │  (HuggingFace /     │
                                │   transformers)     │
                                └─────────────────────┘
```

The pipeline is fully modular: each component (chunker, indexer, retriever, LLM) can be used independently.


## Chunking Strategy

Files are split using **LangChain text splitters** tailored to each file type:

| File type | Splitter | Logic |
|-----------|----------|-------|
| `.py`     | `PythonCodeTextSplitter` | Respects Python AST boundaries (class, function, block) |
| `.md`     | `MarkdownTextSplitter`   | Respects Markdown heading and paragraph boundaries |

Both splitters are configured with:
- **`chunk_size`**: maximum character length per chunk (default: `2000`). Otimal chunk size after performing tests is **1400**.
- **`chunk_overlap`**: `max(10, chunk_size // 5)` to avoid cutting relevant context at boundaries, chunk overlap is set to 20%

Only `.py` and `.md` files are indexed.

## Indexing Strategy

Two distinct indexes are build, for *python* files and *markdown* files in order to optimise the retrieval. The type of files to be indexed can be set via the flag `--file_type` (defaults to `markdown`)

Three indexing strategies are available via the `--indexing_method` flag:

### BM25 (default)
- Uses [bm25s](https://github.com/xhluca/bm25s)
- Index is serialized to disk.

### Vector (dense)
- Encodes chunks a **sentence-transformers** model. After testing, the model offering the best performance / cost ratio is `sentence-transformers/msmarco-bert-base-dot-v5`
- Stores embeddings in a [ChromaDB](https://www.trychroma.com/) persistent collection.

### Hybrid (BM25 + Vector)
- Runs both indexing methods


## Retrieval Method

Three retrieval strategies are available via the `--retrieving_method` flag:

### BM25 (default)
- Index is loaded from disk
- Performance has been improved by adjusting k1 (term frequency saturation) and b (document length normalization).
- Returns the top-k chunks ranked by BM25 score.

### Vector (dense)
- Loads chunks embeddings from [ChromaDB](https://www.trychroma.com/).
- Retrieves top-k chunks by cosine similarity.

### Hybrid (BM25 + Vector)
- Runs both retrievers independently with `k × k_factor` results each.
- Fuses the ranked lists using **Reciprocal Rank Fusion (RRF)** with configurable per-method weights.
- Produces a final merged top-k ranking.


## Performance Analysis

Evaluation is performed with a **Recall@k** metric: a retrieved source is counted as a hit if it overlaps by at least 5 % with any ground-truth source.

| Method  | Recall@5 (code) | Recall@5 (code) |
|---------|------------------------|------------------------|
| BM25    | 55 %                 | 86 %                 |
| Vector  | N/A                 | 73 %                 |
| Hybrid  | N/A                 | 88 %                 |

The performance of dense vector embedding search is lower than BM25 for docs search, but it allows improvement for hybrid search over BM25.
The performance of dense vector embedding is not considered for code search, as it does not offer improvement over BM25.

## Design Decisions

- **Python Fire** is used for CLI generation: `RAGPipeline` public methods become subcommands automatically, with type-checked arguments via Pydantic `validate_with` decorators.
- **Factory pattern** for indexers and retrievers: adding a new retrieval backend only requires a new processor class and a single factory entry.
- **Use of langchain text splitters** Langchain library has efficient text splitters that are designed to respect markdown and python natural boundaries. Efficient chunking improves retrieving efficiency.
- **Use of sentence transformers for dense embedding** Sentence transformers is the state of the art librairy for using embedding models. A wide selection of models are available from Hugging Face Hub and ready to use.


## Challenges Faced

The following challenges were faced during the course of the project:
- *Technology stack selection:* The AI ecosystem is characterized by a large and rapidly evolving set of libraries and frameworks. Selecting the appropriate stack required a careful assessment of each tool’s purpose, strengths, and limitations in order to identify the most suitable components for the project requirements.
- *System architecture design:* The RAG system required a modular architecture to ensure scalability and facilitate iterative improvements. This was essential to accommodate additional functionalities aimed at enhancing pipeline performance, such as hybrid search and multi-indexing strategies.
- *Performance optimization:* System performance was progressively improved through an iterative trial-and-error process, involving the tuning of several critical RAG parameters (chunk size, chunk overlap, BM25 parameters, embedding model, and RRF weights). Due to the absence of GPU resources, only lightweight embedding models could be used, which made it challenging to achieve acceptable performance with dense vector retrieval methods.


## Example Usage

### 1. Index the repository

```bash
# Using the Makefile (BM25 by default)
make run ARGS="index"

# Full options
uv run --env-file .env.hf rag index \
  --repository data/raw/vllm-0.10.1 \
  --save_directory data/processed/ \
  --max_chunk_size 2000 \
  --indexing_method bm25 \
  --file_type markdown
```

Available `indexing_method` values: `bm25`, `vector`, `hybrid`.
Available `file_type` values: `python`, `markdown`

### 2. Search for a single query

```bash
uv run --env-file .env.hf rag search \
  --query "How does vLLM handle continuous batching?" \
  --retrieving_method hybrid \
  --k 5
```

### 3. Batch search over a dataset

```bash
uv run --env-file .env.hf rag search_dataset \
  --dataset_path datasets_public/public/UnansweredQuestions/dataset_docs_public.json \
  --retrieving_method bm25 \
  --k 10
```

### 4. Generate answers for a single query

```bash
uv run --env-file .env.hf rag answer \
  --query "What is the difference between eager and lazy mode in vLLM?" \
  --retrieving_method vector \
  --k 5
```

### 5. Generate answers for a full dataset

```bash
uv run --env-file .env.hf rag answer_dataset \
  --student_search_result_path data/output/search_results/dataset_docs_public.json \
  --save_directory data/output/search_result_and_answer/
```

### 6. Evaluate recall

```bash
uv run --env-file .env.hf rag evaluate \
  --student_answer_path data/output/search_results/dataset_code_public.json \
  --dataset_path datasets_public/public/AnsweredQuestions/dataset_code_public.json
```


## Instructions

### Prerequisites
- Python 3.10
- [`uv`](https://docs.astral.sh/uv/)

### Installation

```bash
make install
# or
uv sync
```

### Execution

```bash
# Run any pipeline command
make run ARGS="<command> [options]"

# Examples
make run ARGS="index"
make run ARGS="search --query 'What is PagedAttention?'"
```

Or directly with `uv`:

```bash
uv run --env-file=.env.hf rag <command> [options]
```

### Linting and Testing

```bash
make lint         # flake8 + mypy (standard)
make lint-strict  # flake8 + mypy --strict
make test         # pytest suite
make clean        # remove caches and build artifacts
```


## Resources

### RAG & Information Retrieval
- [RAG : augmenter un LLM avec vos données - Stéphane Robert](https://blog.stephane-robert.info/docs/developper/programmation/python/rag-introduction/)
- [Speech and Language Processing (3rd ed. draft) Dan Jurafsky and James H. Martin - Chapter 11: Information Retrieval and Retrieval-Augmented Generation](https://web.stanford.edu/~jurafsky/slp3/)

### Libraries & Tools
- [LangChain Text Splitters documentation](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [bm25s — fast BM25 in Python](https://github.com/xhluca/bm25s)
- [ChromaDB documentation](https://docs.trychroma.com/)
- [sentence-transformers documentation](https://www.sbert.net/)

### AI Usage
AI assistance (Claude / Gemini / ChatGPT) was used for the following tasks during this project:
- **Pair programming**: brainstorming application architecture, performing code reviews.
- **Docstring writing**: generating docstrings for public methods from their type signatures and logic.
- **README drafting**: structuring and wording the present document based on the subject requirements.

All AI-generated code was reviewed, tested, and integrated manually.
