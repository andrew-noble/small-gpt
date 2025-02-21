from typing import List
from sentencepiece import SentencePieceProcessor

# Default tokenizer model path if none is provided
TOKENIZER_MODEL = "tokenizer.model"


class Tokenizer:
    """
    Wrapper class for SentencePiece tokenizer that handles text encoding and decoding.
    Provides convenient access to special tokens and vocabulary information.
    """
    def __init__(self, tokenizer_model):
        """
        Initialize the tokenizer with a SentencePiece model.
        Args:
            tokenizer_model: Path to the SentencePiece model file. If None, uses default path.
        """
        model_path = tokenizer_model if tokenizer_model else TOKENIZER_MODEL
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

        # Cache special token IDs and vocabulary size for convenient access. this avoids repeated lookups, for efficiency
        self.n_words = self.sp_model.vocab_size()
        self.bos_id = self.sp_model.bos_id()  # Beginning of sequence token ID
        self.eos_id = self.sp_model.eos_id()  # End of sequence token ID
        self.pad_id = self.sp_model.pad_id()  # Padding token ID

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """
        Encode a text string into a list of token IDs.
        Args:
            s (str): Input text to tokenize
            bos (bool): Whether to add beginning-of-sequence token
            eos (bool): Whether to add end-of-sequence token
        Returns:
            List[int]: List of token IDs
        """
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, tokens: List[int]) -> str:
        """
        Decode a list of token IDs back into a text string.
        Args:
            tokens (List[int]): List of token IDs to decode
        Returns:
            str: Decoded text
        """
        return self.sp_model.decode(tokens)