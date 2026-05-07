import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """

    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        # Add special tokens with fixed IDs
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]

        for idx, token in enumerate(special_tokens):
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token

        self.vocab_size = len(special_tokens)

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        unique_words = set()

        # Collect all unique lowercase words
        for text in texts:
            words = text.lower().split()
            unique_words.update(words)

        # Add words in sorted order
        sorted_words = sorted(unique_words)

        current_id = 4

        for word in sorted_words:
            if word not in self.word_to_id:
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1

        self.vocab_size = current_id

    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        words = text.lower().split()

        return [
            self.word_to_id.get(word, self.word_to_id[self.unk_token])
            for word in words
        ]

    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        words = []

        for idx in ids:
            words.append(
                self.id_to_word.get(idx, self.unk_token)
            )

        return " ".join(words)