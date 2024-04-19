import hashlib
from typing import List

import scipy
import numpy as np
from scipy.stats import norm
import torch
from transformers import LogitsWarper


class WatermarkBase:
    """
    Base class for watermarking distributions with fixed-group green-listed tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """

    def __init__(self, fraction: float = 0.5, strength: float = 2.0, vocab_size: int = 50257, watermark_key: int = 0):
        rng = np.random.default_rng(self._hash_fn(watermark_key))
        mask = np.array([True] * int(fraction * vocab_size) + [False] * (vocab_size - int(fraction * vocab_size)))
        rng.shuffle(mask)
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

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        """Add the watermark to the logits and return new logits."""
        # watermark = self.strength * self.green_list_mask
        # new_logits = scores + watermark.to(scores.device)
        watermark = self.green_list_mask
        new_logits = scores + watermark.to(scores.device)
        for gindex in torch.nonzero(self.green_list_mask == 1).squeeze():
            green_sum = torch.sum(torch.exp(new_logits[0,gindex]+self.strength))
        for rindex in torch.nonzero(self.green_list_mask == 0).squeeze():
            red_sum = torch.sum(torch.exp(new_logits[0,rindex]+self.strength))
        new_logits = torch.exp(new_logits+self.strength)/(green_sum+red_sum)
        print(new_logits)
        return new_logits


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
    
    @staticmethod
    def _compute_tau(m: int, N: int, alpha: float) -> float:
        """
        Compute the threshold tau for the dynamic thresholding.

        Args:
            m: The number of unique tokens in the sequence.
            N: Vocabulary size.
            alpha: The false positive rate to control.
        Returns:
            The threshold tau.
        """
        factor = np.sqrt(1 - (m - 1) / (N - 1))
        tau = factor * norm.ppf(1 - alpha)
        return tau
        
    def _compute_p_value(self, z):
        p_value = scipy.stats.norm.sf(z)
        return p_value
    
    def _score_sequence(
        self,
        input_ids: torch.Tensor,
        green_tokens,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_z_at_T: bool = False,
        return_p_value: bool = True,
    ):
        # ngram_to_watermark_lookup, frequencies_table = self._score_ngrams_in_passage(input_ids)
        # green_token_mask, green_unique, offsets = self._get_green_at_T_booleans(input_ids, ngram_to_watermark_lookup)

        # # Count up scores over all ngrams
        # if self.ignore_repeated_ngrams:
        #     # Method that only counts a green/red hit once per unique ngram.
        #     # New num total tokens scored (T) becomes the number unique ngrams.
        #     # We iterate over all unqiue token ngrams in the input, computing the greenlist
        #     # induced by the context in each, and then checking whether the last
        #     # token falls in that greenlist.
        #     num_tokens_scored = len(frequencies_table.keys())
        #     green_token_count = sum(ngram_to_watermark_lookup.values())
        # else:
        #     num_tokens_scored = sum(frequencies_table.values())
        #     assert num_tokens_scored == len(input_ids) - self.context_width + self.self_salt
        #     green_token_count = sum(freq * outcome for freq, outcome in zip(frequencies_table.values(), ngram_to_watermark_lookup.values()))
        # assert green_token_count == green_unique.sum()

        # HF-style output dictionary
        score_dict = dict()
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=len(input_ids)))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_tokens))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=(green_tokens / len(input_ids))))
        if return_z_score:
            score_dict.update(dict(z_score=self._z_score(green_tokens, len(input_ids), self.fraction)))
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._z_score(green_tokens, len(input_ids))
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))
        # if return_green_token_mask:
        #     score_dict.update(dict(green_token_mask=green_token_mask.tolist()))
        # if return_z_at_T:
        #     # Score z_at_T separately:
        #     sizes = torch.arange(1, len(green_unique) + 1)
        #     seq_z_score_enum = torch.cumsum(green_unique, dim=0) - self.fraction * sizes
        #     seq_z_score_denom = torch.sqrt(sizes * self.fraction * (1 - self.fraction))
        #     z_score_at_effective_T = seq_z_score_enum / seq_z_score_denom
        #     z_score_at_T = z_score_at_effective_T[offsets]
        #     assert torch.isclose(z_score_at_T[-1], torch.tensor(z_score))

        #     score_dict.update(dict(z_score_at_T=z_score_at_T))

        return score_dict
    
    def detect(self, sequence: List[int], z_threshold: float = None, **kwargs,) -> dict:

        """Detect the watermark in a sequence of tokens and return the z value."""
        green_tokens = int(sum(self.green_list_mask[i] for i in sequence))
        output_dict = {}
        score_dict = self._score_sequence(sequence, green_tokens, **kwargs)
        output_dict.update(score_dict)
        assert z_threshold is not None, "Need a threshold in order to decide outcome of detection test"
        output_dict["prediction"] = score_dict["z_score"] > z_threshold
        if output_dict["prediction"]:
            output_dict["confidence"] = 1 - score_dict["p_value"]

        return output_dict

    # def unidetect(self, sequence: List[int]) -> float:
    #     """Detect the watermark in a sequence of tokens and return the z value. Just for unique tokens."""
    #     sequence = list(set(sequence))
    #     green_tokens = int(sum(self.green_list_mask[i] for i in sequence))
    #     return self._z_score(green_tokens, len(sequence), self.fraction)
    
    # def dynamic_threshold(self, sequence: List[int], alpha: float, vocab_size: int) -> (bool, float):
    #     """Dynamic thresholding for watermark detection. True if the sequence is watermarked, False otherwise."""
    #     z_score = self.unidetect(sequence)
    #     tau = self._compute_tau(len(list(set(sequence))), vocab_size, alpha)
    #     return z_score > tau, z_score
