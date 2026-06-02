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
    """Manager class responsible for loading the LLM model and generating
    query responses in batches.
    """

    def __init__(self, tui: TUI, config: LLMConfig) -> None:
        """Initializes the LLMManager, loading the causal language model and
        tokenizer.

        Args:
            tui: A TUI instance to handle user interface / progress output.
            config: An LLMConfig configuration object.

        Raises:
            LLMChatProcessorError: If loading the tokenizer or model fails.
        """
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

    def answer_queries(
        self, queries: ChatMessages, description: str = "Processing_queries"
    ) -> list[str]:
        """Generates natural language answers for multiple formatted chat
        queries.

        Args:
            queries: A list of formatted chat conversation messages.

        Returns:
            A list of answer strings from the LLM.
        """
        with self._tui.progress(
            description, len(queries), "query"
        ) as progress:
            output: list[str] = []
            for batch in batched(queries, self._config.batch_size):
                output.extend(self._process_batch(list(batch)))
                progress.update(len(batch))
            return output

    @torch.inference_mode()
    def _process_batch(self, queries: ChatMessages) -> list[str]:
        """Generates answers for a single batch of queries under inference
        mode.

        Args:
            queries: A list of formatted chat messages for a single batch.

        Returns:
            A list of decoded answer strings from the LLM.

        Raises:
            LLMChatProcessorError: If the token length of the query batch
                exceeds the model's capacity limit.
        """
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
