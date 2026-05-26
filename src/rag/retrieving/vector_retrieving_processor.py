from typing import cast

import chromadb
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.util import semantic_search, dot_score
from transformers import AutoModelForCausalLM, AutoTokenizer, TokenizersBackend

from rag.models.minimal_source import MinimalSource
from rag.models.question import UnansweredQuestion
from rag.models.search_result import MinimalSearchResults, StudentSearchResults
from rag.retrieving.retrieving_processor import RetrievingProcessor
import torch

class VectorRetrievingProcessor(RetrievingProcessor):
    COLLECTION = "rag_vllm_repository"

    def __init__(self) -> None:
        self._embedder = SentenceTransformer(
            "sentence-transformers/msmarco-bert-base-dot-v5"
        )
        self._store = chromadb.PersistentClient("data/processed")

    def _enhance_query(self, query: str) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = cast(
            TokenizersBackend,
            AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-0.6B",
                padding_side="left",
            ),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-0.6B",
                torch_dtype="auto",
                device_map="auto" if device == "cuda" else "cpu",
            )
        model = model.to(device)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a search assistant. Your job is to rewrite the user's query by adding "
                    "synonyms, related terms, and their English translations to improve semantic retrieval. "
                    "Return ONLY the variations separated by commas, nothing else. "
                    "No introduction, no conclusion, no conversational filler."
                )
            },
            {
                "role": "user",
                "content": query
            }
        ]
        prompt_tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )
        prompt_tokens = prompt_tokens.to(device)
        input_length = prompt_tokens["input_ids"].shape[1]
        generated_ids = model.generate(
            **prompt_tokens, max_new_tokens=256
        )
        new_tokens = generated_ids[:, input_length:]
        output = tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )
        return output[0]     



    def retrieve(
        self, queries: list[UnansweredQuestion], k: int
    ) -> StudentSearchResults:
        collection = self._store.get_collection(self.COLLECTION)
        corpus = collection.get(include=["embeddings", "metadatas", "documents"])
        corpus_embeddings = np.array(corpus["embeddings"], dtype=np.float32)
        results = []
        for i, query in enumerate(queries):
            print(i)
            new_q = self._enhance_query(query.question)
            query_embedding = self._embedder.encode_query(
                query.question,  # convert_to_numpy=True, precision="float32"
            )
            similarity_scores = semantic_search(
                query_embedding, corpus_embeddings, top_k=100, score_function=dot_score
            )[0]
            # similarity_scores = self._embedder.similarity(
            #     query_embedding, corpus_embeddings
            # )[0]
            # scores, indices = torch.topk(similarity_scores, k=k)
            # print(scores, indices)
            documents = [(corpus["documents"][score["corpus_id"]], corpus["metadatas"][score["corpus_id"]]) for score in similarity_scores]
            model_inputs = [(query.question, document[0]) for document in documents]
            model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2")
            scores = model.predict(model_inputs)
            results_cross = [{"document": doc, "score": score} for doc, score in zip(documents, scores)]
            results_cross = sorted(results_cross, key=lambda x: x["score"], reverse=True)
            results.append([res["document"][1] for res in results_cross][:k])
            # results.append([corpus["metadatas"][idx] for idx in indices])

        search_result = []
        for i, result in enumerate(results):
            search_result.append(
                MinimalSearchResults(
                    question_id=queries[i].question_id,
                    question=queries[i].question,
                    retrieved_sources=[
                        MinimalSource(
                            file_path=source["source"],
                            first_character_index=source["start_index"],
                            last_character_index=source["end_index"],
                        )
                        for source in result
                    ],
                )
            )
        return StudentSearchResults(search_results=search_result, k=k)
