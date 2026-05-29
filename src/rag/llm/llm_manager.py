from typing import cast

import torch
from more_itertools import batched
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    TokenizersBackend,
)
from transformers._typing import GenerativePreTrainedModel

from rag.config.llm import LLMConfig
from rag.exceptions import RAGException
from rag.tui import TUI

ChatMessages = list[list[dict[str, str]]]


class LLMChatProcessorError(RAGException):
    pass


class LLMManager:
    def __init__(self, tui: TUI, config: LLMConfig) -> None:
        self._config = config
        self._tui = tui
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self._tokenizer = cast(
                TokenizersBackend,
                AutoTokenizer.from_pretrained(
                    self._config.model,
                    padding_side="left",
                ),
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = cast(
                GenerativePreTrainedModel,
                AutoModelForCausalLM.from_pretrained(
                    self._config.model,
                    torch_dtype="auto",
                    device_map="auto" if self._device == "cuda" else "cpu",
                ),
            )
        except OSError as e:
            raise LLMChatProcessorError(
                f"Invalid model name: '{self._config.model}'"
            ) from e

    def answer_queries(self, queries: ChatMessages) -> list[str]:
        with self._tui.progress(
            "Processing queries", len(queries), "query"
        ) as progress:
            output: list[str] = []
            for batch in batched(queries, self._config.batch_size):
                output.extend(self._process_batch(list(batch)))
                progress.update(len(batch))
            return output

    @torch.inference_mode()
    def _process_batch(self, queries: ChatMessages) -> list[str]:
        prompt_tokens = self._tokenizer.apply_chat_template(
            queries,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )
        assert isinstance(prompt_tokens, BatchEncoding)
        prompt_tokens = prompt_tokens.to(self._device)
        input_length = prompt_tokens["input_ids"].shape[1]
        if input_length > self._tokenizer.model_max_length:
            raise LLMChatProcessorError(
                "Model input length exceeds model max length: "
                f"{input_length} > {self._tokenizer.model_max_length}"
            )
        generated_ids = self._model.generate(
            **prompt_tokens, max_new_tokens=self._config.max_new_tokens
        )
        new_tokens = generated_ids[:, input_length:]
        output = self._tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )
        return output
