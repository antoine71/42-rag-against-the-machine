from typing import cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TokenizersBackend,
)
from transformers._typing import GenerativePreTrainedModel

from rag.exceptions import RAGException
from rag.llm.prompt_generator import (
    ChatTemplatePromptGenerator,
    PromptGenerator,
    SimplePromptGenerator,
)
from rag.models.search_result import MinimalSearchResults
from rag.utils.files_manager import FilesManager


class LLMChatProcessorError(RAGException):
    pass


class LLMChatProcessor:
    def __init__(
        self, files_manager: FilesManager, model_name: str, k: int
    ) -> None:
        self._files_manager = files_manager
        self._k = k
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self._tokenizer = cast(
                TokenizersBackend,
                AutoTokenizer.from_pretrained(
                    model_name,
                    padding_side="left",
                ),
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = cast(
                GenerativePreTrainedModel,
                AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="auto" if self._device == "cuda" else "cpu",
                ),
            )
            self._model = self._model.to(self._device)
        except OSError as e:
            raise LLMChatProcessorError(
                f"Invalid model name: '{model_name}'"
            ) from e
        prompt_generator: type[PromptGenerator] = (
            ChatTemplatePromptGenerator
            if self._tokenizer.chat_template is not None
            else SimplePromptGenerator
        )
        print(prompt_generator.__name__)
        self._prompt_generator = prompt_generator(
            self._tokenizer, self._files_manager, k
        )

    def answer_batch_query(
        self, input: tuple[MinimalSearchResults, ...]
    ) -> list[str]:
        prompt_tokens = self._prompt_generator.generate_tokens(input)
        prompt_tokens = prompt_tokens.to(self._device)
        input_length = prompt_tokens["input_ids"].shape[1]
        if input_length > self._tokenizer.model_max_length:
            raise LLMChatProcessorError(
                f"Model input length exceeds model max length: {input_length} > {self._tokenizer.model_max_length}"
            )
        generated_ids = self._model.generate(
            **prompt_tokens, max_new_tokens=256
        )
        new_tokens = generated_ids[:, input_length:]
        output = self._tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )
        return output
