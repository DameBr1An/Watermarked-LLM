import hashlib
import math
import numpy as np
from scipy.stats import norm
import torch
from transformers import LogitsWarper # type: ignore

class WatermarkBase:
    """
    Base class for watermarking distributions with fixed grouped green-listed tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """

    def __init__(self, fraction: float = 0.5, strength: float = 2.0, vocab_size: int = 50257, watermark_key: int = 0):
        rng = np.random.default_rng(self._hash_fn(watermark_key))
        mask = rng.choice([True, False], size=vocab_size, p=[fraction, 1-fraction])
        self.green_list_mask = torch.tensor(mask, dtype=torch.float32)
        self.strength = strength
        self.fraction = fraction

    @staticmethod
    def _hash_fn(x: int) -> int:
        """solution from https://stackoverflow.com/questions/67219691/python-hash-function-that-returns-32-or-64-bits"""
        x = np.int64(x)
        return int.from_bytes(hashlib.sha256(x).digest()[:4], 'little')


class WatermarkLogitsWarper(WatermarkBase, LogitsWarper):
    """
    LogitsWarper for watermarking distributions with fixed-group green-listed tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_losses = []
        self.all_input_ids = []
        self.all_logits = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        """Add the watermark to the logits and return."""
        scores[:, self.green_list_mask == 1] += self.strength
        # scores = torch.exp(scores)    # softmax
        # scores = scores/(1+torch.exp(scores))     #swish
        return scores

    def record_loss(self, loss):
        """Record the loss for each step."""
        self.all_losses.append(loss.item())

    def calculate_perplexity(self):
        """Calculate the overall perplexity."""
        avg_loss = sum(self.all_losses) / len(self.all_losses)
        perplexity = torch.exp(torch.tensor(avg_loss))
        return perplexity.item()

    def get_top_n_perplexity_words(self, tokenizer, n=5):
        """Get the top N words with the highest perplexity."""
        word_losses = []
        for input_id, logits in zip(self.all_input_ids, self.all_logits):
            tokens = tokenizer.convert_ids_to_tokens(input_id.squeeze().tolist())
            for token, logit in zip(tokens, logits):
                word_losses.append((token, logit))
        # Sort by loss and get top N words
        word_losses.sort(key=lambda x: x[1], reverse=True)
        return word_losses[:n]

class WatermarkDetector(WatermarkBase):
    """
    Class for detecting watermarks in a sequence of tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _z_score(num_green: int, total: int, fraction: float) -> float:
        """Calculate and return the z-score of the number of green tokens in a sequence."""
        return (num_green - fraction * total) / np.sqrt(fraction * (1 - fraction) * total)
    
    def _score_sequence(
        self,
        input_ids: torch.Tensor,
        num_green_tokens: int,
        sequence_size: int,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_z_score: bool = True,
        return_p_value: bool = True,
        return_green_token_mask: bool = True
    ):

        score_dict = dict()
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=num_green_tokens))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=(num_green_tokens / sequence_size)))
        if return_z_score:
            score_dict.update(dict(z_score=self._z_score(num_green_tokens, sequence_size, self.fraction)))
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._z_score(num_green_tokens, sequence_size)
            score_dict.update(dict(p_value=norm.sf(z_score)))
        if return_green_token_mask:
            green_index = torch.nonzero(self.green_list_mask == 1).squeeze()
            green_list_mask = [item for item in input_ids if item in green_index]
            score_dict.update(dict(green_token_mask=green_list_mask))
        return score_dict
    
    def detect(self, sequence: list[int], z_threshold: float = None, **kwargs,) -> dict:
        """Detect the watermark in a sequence of tokens and return the z value."""

        num_green_tokens = int(sum(self.green_list_mask[i] for i in sequence))
        sequence_size = len(sequence)
        output_dict = {}
        score_dict = self._score_sequence(sequence, num_green_tokens, sequence_size, **kwargs)
        output_dict.update(score_dict)
        output_dict["prediction"] = (score_dict["z_score"] > z_threshold)
        if output_dict["prediction"]:
            output_dict["confidence"] = 1 - score_dict["p_value"]
        return output_dict