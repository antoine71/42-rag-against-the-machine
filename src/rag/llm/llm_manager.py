from typing import cast

from transformers import AutoModelForCausalLM, AutoTokenizer, TokenizersBackend


class LLMManager:
    def __init__(self, model: str) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            model, cache_dir="/tmp/hf"
        )
        self.tokenizer = cast(
            TokenizersBackend, AutoTokenizer.from_pretrained(model)
        )

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def tokenize_batch(self, texts: list[str]) -> list[list[int]]:
        return self.tokenizer(texts)["input_ids"]

    def get_vocab(self) -> dict[str, int]:
        return self.tokenizer.get_vocab()
